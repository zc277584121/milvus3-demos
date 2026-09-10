from __future__ import annotations

import pytest

from structarray_hybrid_demo.data import DatasetBundle
from structarray_hybrid_demo.embedding import EmbeddingBatchReport, ModelIdentity
from structarray_hybrid_demo.repository import ChildHit, FusionHit, ParentHit
from structarray_hybrid_demo.service import SearchContractError, StructArrayHybridService


class FakeEmbedder:
    def __init__(self) -> None:
        self.loaded = True
        self.cache_available = True
        self.identity = ModelIdentity(
            model_id="test/model",
            revision="abc",
            vector_dimension=4,
            dense_output_name="dense_vecs",
            device="cpu",
        )
        self.calls: list[list[str]] = []

    def load(self) -> None:
        self.loaded = True

    def unload(self) -> None:
        self.loaded = False

    def embed(self, texts: list[str]) -> tuple[list[list[float]], EmbeddingBatchReport]:
        self.calls.append(texts)
        vectors = [[1.0, 0.0, 0.0, 0.0] for _ in texts]
        report = EmbeddingBatchReport(
            vector_count=len(vectors), vector_dimension=4, norm_min=1.0, norm_max=1.0
        )
        return vectors, report


class FakeRepository:
    def __init__(self) -> None:
        self.status_result: dict[str, object] = {
            "milvus_uri": "http://127.0.0.1:49530",
            "server_version": "3.0.0",
            "collection_name": "c",
            "collection_exists": True,
            "raw_collection_names": ["c"],
            "row_count": 30,
            "index_names": ["i1", "i2"],
        }
        self.last_prepare_args: tuple | None = None

    def status(self) -> dict[str, object]:
        return self.status_result

    def prepare(self, bundle, parent_vectors, child_vectors):
        self.last_prepare_args = (bundle, parent_vectors, child_vectors)

        class Report:
            def public_dict(self) -> dict[str, object]:
                return {"status": "prepared"}

        return Report()

    def parent_search(self, *, query_vector, videos_by_id, limit):
        first = list(videos_by_id.values())[0]
        return (ParentHit(rank=1, score=0.9, video=first),)

    def child_search(self, *, query_vector, videos_by_id, limit):
        first = list(videos_by_id.values())[0]
        return (
            ChildHit(
                rank=1,
                score=0.8,
                video=first,
                offset=0,
                observation=first.observations[0],
            ),
        )

    def fusion_search(
        self,
        *,
        query_vector,
        videos_by_id,
        limit,
        parent_weight,
        child_weight,
        collapse_strategy,
        collapse_topk,
    ):
        first = list(videos_by_id.values())[0]
        return (FusionHit(rank=1, score=0.85, video=first),)

    def cleanup(self) -> dict[str, object]:
        return {"status": "clean"}


@pytest.fixture
def service(bundle: DatasetBundle) -> StructArrayHybridService:
    embedder = FakeEmbedder()
    repository = FakeRepository()
    svc = StructArrayHybridService(embedder=embedder, repository=repository)
    # Seed the dataset so search() does not read the real JSON again.
    svc._bundle = bundle
    svc._videos_by_id = {video.video_id: video for video in bundle.videos}
    return svc


def test_search_returns_three_paths_and_derived_child_weight(
    service: StructArrayHybridService,
) -> None:
    report = service.search(query="a parked sedan", limit=4, parent_weight=0.7)
    assert report.public_dict()["status"] == "passed"
    assert report.parent_weight == 0.7
    assert report.child_weight == pytest.approx(0.3)
    assert report.collapse_strategy == "topk_sum"
    assert report.collapse_topk == 3
    paths = report.public_dict()["paths"]
    assert set(paths) == {"parent", "child", "fusion"}
    assert paths["parent"]["results"][0]["rank"] == 1
    assert paths["child"]["results"][0]["offset"] == 0
    assert paths["fusion"]["results"][0]["rank"] == 1


def test_search_annotates_preset_matches(service: StructArrayHybridService) -> None:
    report = service.search(
        query="a black suv in a residential neighborhood",
        limit=4,
        parent_weight=0.5,
    )
    paths = report.public_dict()["paths"]
    for path_key in ("parent", "child", "fusion"):
        result = paths[path_key]["results"][0]
        assert "scene_match" in result
        assert "object_match" in result
        assert "color_match" in result
        # Real detected values back every boolean so the UI can strike out words
        # that contradict the intent instead of only highlighting matches.
        assert "actual_scene" in result
        assert "actual_object" in result
        assert "actual_color" in result


def test_search_reports_ground_truth_and_path_recall(
    service: StructArrayHybridService,
) -> None:
    report = service.search(
        query="a black suv in a residential neighborhood", limit=4, parent_weight=0.5
    )
    payload = report.public_dict()
    assert payload["ground_truth"]["matched_count"] == 5
    assert payload["ground_truth"]["color_terms"] == ["black"]
    assert payload["path_recall"]["parent"]["gt_count"] == 5
    # FakeRepository returns only the first video (local_residential, no suv),
    # so it is not among the 5 residential+black-suv ground-truth videos.
    assert payload["path_recall"]["parent"]["matched"] == 0
    assert payload["path_recall"]["fusion"]["matched"] == 0


def test_search_omits_match_flags_for_freeform_query(
    service: StructArrayHybridService,
) -> None:
    report = service.search(query="any freeform text", limit=4, parent_weight=0.5)
    paths = report.public_dict()["paths"]
    result = paths["parent"]["results"][0]
    assert "scene_match" not in result
    assert "object_match" not in result
    assert "color_match" not in result
    assert "ground_truth" not in report.public_dict()
    assert "path_recall" not in report.public_dict()


def test_search_rejects_out_of_range_weight(service: StructArrayHybridService) -> None:
    with pytest.raises(SearchContractError, match="parent_weight"):
        service.search(query="x", parent_weight=1.5)
    with pytest.raises(SearchContractError, match="parent_weight"):
        service.search(query="x", parent_weight=-0.1)


def test_search_rejects_out_of_range_limit(service: StructArrayHybridService) -> None:
    with pytest.raises(SearchContractError, match="limit"):
        service.search(query="x", limit=0)
    with pytest.raises(SearchContractError, match="limit"):
        service.search(query="x", limit=21)


def test_search_rejects_empty_query(service: StructArrayHybridService) -> None:
    with pytest.raises(SearchContractError, match="empty"):
        service.search(query="   ")


def test_prepare_embeds_parents_then_children(service: StructArrayHybridService) -> None:
    result = service.prepare()
    assert result["status"] == "prepared"
    embedder = service.embedder
    assert len(embedder.calls) == 2
    parent_batch, child_batch = embedder.calls
    assert len(parent_batch) == 30
    assert len(child_batch) == service._bundle.observation_count
