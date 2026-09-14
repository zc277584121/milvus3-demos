from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest
import torch

from embedding_list_demo.cache import PageEmbeddingCache
from embedding_list_demo.config import COLLECTION_NAME, PAGE_COUNT, RuntimeConfig
from embedding_list_demo.manual import ManualManifest
from embedding_list_demo.model import EmbeddingBatchReport, ModelIdentity, SnapshotRecord
from embedding_list_demo.repository import MilvusHit
from embedding_list_demo.service import EmbeddingListService, ScoreComparisonError


def identity() -> ModelIdentity:
    snapshot = SnapshotRecord("test", "a" * 40, "b" * 40, "/test")
    return ModelIdentity(
        model_id="test",
        adapter=snapshot,
        base=snapshot,
        inference_dtype="float32",
        inference_device="cpu",
        vector_dimension=128,
        colpali_engine_version="test",
        torch_version="test",
        transformers_version="test",
    )


def page_scores() -> np.ndarray:
    scores = np.linspace(0.01, 0.4, PAGE_COUNT, dtype=np.float64)
    scores[1] = 0.9
    return scores


class FakeModel:
    def __init__(self) -> None:
        self.identity = identity()
        self.loaded = False
        self.page_calls = 0
        self.query_calls = 0
        self.explanation_calls = 0

    def runtime_dict(self) -> dict[str, object]:
        return {
            **self.identity.public_dict(),
            "loaded": self.loaded,
            "configured_device": "cpu",
            "device_type": "cpu",
            "cpu_only": True,
        }

    def embed_pages(self, **_: Any) -> tuple[list[torch.Tensor], EmbeddingBatchReport]:
        self.loaded = True
        self.page_calls += 1
        page_vectors = [
            torch.full((index + 2, 128), float(index + 1)) for index in range(PAGE_COUNT)
        ]
        return page_vectors, EmbeddingBatchReport(
            tuple(range(2, PAGE_COUNT + 2)), 128, 1.0, 1.0, "cpu", "float32"
        )

    def embed_query(self, query: str) -> tuple[torch.Tensor, EmbeddingBatchReport]:
        assert query
        self.loaded = True
        self.query_calls += 1
        return torch.ones((3, 128)), EmbeddingBatchReport((3,), 128, 1.0, 1.0, "cpu", "float32")

    def score_pages(self, *_: Any) -> np.ndarray:
        return page_scores()

    def explain_page(self, **kwargs: Any) -> dict[str, object]:
        self.explanation_calls += 1
        return {
            "page_id": kwargs["page_id"],
            "source": "local_colsmol_query_page_multi_vector",
            "grid": {"columns": 1, "rows": 1, "patch_count": 1},
            "concepts": [{"label": "stakeholder", "peak_similarity": 0.8}],
            "patches": [{"patch_index": 0, "x": 0, "y": 0, "intensity": 1}],
        }

    def unload(self) -> None:
        self.loaded = False


@dataclass
class FakeAudit:
    target_exists: bool
    server_version: str = "3.0.0"

    def public_dict(self) -> dict[str, object]:
        return {
            "server_version": self.server_version,
            "target_exists": self.target_exists,
            "raw_collection_names": [COLLECTION_NAME] if self.target_exists else [],
        }


class PublicValue:
    def __init__(self, value: dict[str, object]) -> None:
        self.value = value

    def public_dict(self) -> dict[str, object]:
        return self.value


class FakeRepository:
    def __init__(self) -> None:
        self.owns_collection = False
        self.prepare_calls = 0
        self.manual: ManualManifest | None = None

    def raw_audit(self) -> FakeAudit:
        return FakeAudit(target_exists=self.owns_collection)

    def adopt(self, page_count: int) -> bool:
        # The fake never has a pre-existing, unowned Collection; always prepare.
        return False

    def prepare(self, manual: ManualManifest, vectors: Any) -> PublicValue:
        assert manual.page_count == PAGE_COUNT
        assert len(vectors) == PAGE_COUNT
        self.manual = manual
        self.owns_collection = True
        self.prepare_calls += 1
        return PublicValue({"status": "prepared", "metric_type": "MAX_SIM_COSINE"})

    def search(self, query_vectors: torch.Tensor, *, limit: int) -> tuple[MilvusHit, ...]:
        assert query_vectors.shape == (3, 128)
        assert limit == PAGE_COUNT
        assert self.manual is not None
        scores = page_scores()
        order = sorted(range(PAGE_COUNT), key=lambda index: float(scores[index]), reverse=True)
        return tuple(
            MilvusHit(
                page_id=self.manual.pages[index].page_id,
                pdf_page_index=self.manual.pages[index].pdf_page_index,
                printed_page=self.manual.pages[index].printed_page,
                title=self.manual.pages[index].title,
                section=self.manual.pages[index].section,
                document_identifier=self.manual.document_identifier,
                ntrs_id=self.manual.ntrs_id,
                ntrs_record_url=self.manual.ntrs_record_url,
                distribution=self.manual.distribution,
                rights_determination=self.manual.rights_determination,
                contains_third_party_material=self.manual.contains_third_party_material,
                pdf_sha256=self.manual.pdf_sha256,
                image_sha256=self.manual.pages[index].image_sha256,
                score=float(scores[index]) + 0.000001,
            )
            for index in order
        )

    def cleanup(self) -> PublicValue:
        self.owns_collection = False
        return PublicValue({"status": "clean", "target_absent_after": True})


def test_service_prepare_cache_search_status_timings_and_cleanup(
    runtime_config: RuntimeConfig,
) -> None:
    model = FakeModel()
    repository = FakeRepository()
    service = EmbeddingListService(config=runtime_config, model=model, repository=repository)

    initial = service.status()
    first = service.prepare()
    second = service.prepare()
    status = service.status()
    query = first["query_presets"][0]["text"]
    search = service.search(str(query))
    cleanup = service.cleanup()

    assert initial["implementation_status"] == "implemented"
    assert initial["runtime_ready"] is False
    assert initial["dataset"]["revision"] == "NASA/SP-2016-6105 Rev 2"
    assert initial["dataset"]["rights_determination"] == "PUBLIC_USE_PERMITTED"
    assert len(initial["query_presets"]) == 8
    assert first["cache"]["hit"] is False
    assert second["cache"]["hit"] is True
    assert first["timings"]["page_inference_ms"] >= 0
    assert second["timings"]["page_inference_ms"] == 0
    assert status["prepare_timings"]["cold"] is not None
    assert status["prepare_timings"]["warm"] is not None
    assert model.page_calls == 1
    assert repository.prepare_calls == 1
    assert search["milvus_page_order"] == search["local_page_order"]
    assert search["ranking_equal"] is True
    assert search["score_within_tolerance"] is True
    assert len(search["local_explanations"]) == 3
    assert model.explanation_calls == 3
    assert "ground_truth" not in search
    assert search["results"][0]["page_id"] == "nasa-seh-printed-053"
    assert "is_expected_page" not in search["results"][0]
    assert search["results"][0]["pdf_page_index"] == 66
    assert search["results"][0]["printed_page"] == 53
    assert search["timings"]["query_inference_ms"] >= 0
    assert search["results"][0]["evidence_label"].startswith("Unmodified 144-DPI")
    assert repository.manual is not None
    first_page = repository.manual.pages[0]
    first_result = next(item for item in search["results"] if item["page_id"] == first_page.page_id)
    assert first_result["image_width"] == first_page.width
    assert first_result["image_height"] == first_page.height
    assert cleanup["status"] == "clean"
    assert cleanup["dataset_source_preserved"] is True
    assert cleanup["runtime_root_exists"] is False
    assert not PageEmbeddingCache(runtime_config.embedding_cache_root).root.exists()


def test_service_fails_closed_when_complete_local_ranking_differs(
    runtime_config: RuntimeConfig,
) -> None:
    model = FakeModel()
    repository = FakeRepository()
    service = EmbeddingListService(config=runtime_config, model=model, repository=repository)
    prepared = service.prepare()
    original_search = repository.search

    def reversed_search(query_vectors: torch.Tensor, *, limit: int) -> tuple[MilvusHit, ...]:
        return tuple(reversed(original_search(query_vectors, limit=limit)))

    repository.search = reversed_search  # type: ignore[method-assign]
    with pytest.raises(ScoreComparisonError, match="complete local scorer"):
        service.search(str(prepared["query_presets"][0]["text"]))
