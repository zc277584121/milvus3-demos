"""Dataset, model, cache, Milvus, and local-oracle orchestration."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Protocol

import numpy as np
import torch

from embedding_list_demo.cache import CacheLoad, PageEmbeddingCache
from embedding_list_demo.config import (
    ANNS_FIELD,
    COLLECTION_NAME,
    EXECUTION_PATH,
    INDEX_NAME,
    METRIC_TYPE,
    SCORE_ABSOLUTE_TOLERANCE,
    RuntimeConfig,
)
from embedding_list_demo.manual import (
    ManualManifest,
    QueryGroundTruth,
    load_manual,
    load_queries,
    resolve_page_image,
)
from embedding_list_demo.model import ColSmolModel, EmbeddingBatchReport, ModelIdentity
from embedding_list_demo.repository import EmbeddingListRepository, MilvusHit


class ServiceContractError(RuntimeError):
    """Raised when the end-to-end backend state is not ready for an operation."""


class ScoreComparisonError(ServiceContractError):
    """Raised when Milvus output differs from the complete local scoring contract."""


class ModelLike(Protocol):
    @property
    def identity(self) -> ModelIdentity: ...

    @property
    def loaded(self) -> bool: ...

    def runtime_dict(self) -> dict[str, object]: ...

    def embed_pages(
        self, *, dataset_root: Path, manifest: ManualManifest
    ) -> tuple[list[torch.Tensor], EmbeddingBatchReport]: ...

    def embed_query(self, query: str) -> tuple[torch.Tensor, EmbeddingBatchReport]: ...

    def score_pages(
        self, query_vectors: torch.Tensor, page_vectors: list[torch.Tensor]
    ) -> np.ndarray: ...

    def explain_page(
        self,
        *,
        query: str,
        query_vectors: torch.Tensor,
        page_vectors: torch.Tensor,
        page_image: Path,
        page_id: str,
    ) -> dict[str, object]: ...

    def unload(self) -> None: ...


class RepositoryLike(Protocol):
    @property
    def owns_collection(self) -> bool: ...

    def raw_audit(self): ...

    def prepare(self, manual: ManualManifest, vectors: list[torch.Tensor]): ...

    def search(self, query_vectors: torch.Tensor, *, limit: int) -> tuple[MilvusHit, ...]: ...

    def cleanup(self): ...


class EmbeddingListService:
    """Own the complete backend lifecycle within one process."""

    def __init__(
        self,
        *,
        config: RuntimeConfig | None = None,
        model: ModelLike | None = None,
        repository: RepositoryLike | None = None,
    ) -> None:
        self.config = config or RuntimeConfig.from_environment()
        self.model = model or ColSmolModel(self.config)
        self.repository = repository or EmbeddingListRepository()
        self.cache = PageEmbeddingCache(self.config.embedding_cache_root)
        self._manual: ManualManifest | None = None
        self._queries: tuple[QueryGroundTruth, ...] | None = None
        self._page_vectors: list[torch.Tensor] | None = None
        self._cache_manifest: dict[str, object] | None = None
        self._last_prepare: dict[str, object] | None = None
        self._prepare_timings: dict[str, dict[str, float] | None] = {
            "cold": None,
            "warm": None,
        }

    def _verified_manual(self) -> ManualManifest:
        if self._manual is None:
            self._manual = load_manual(self.config.dataset_root)
        return self._manual

    def _verified_queries(self) -> tuple[QueryGroundTruth, ...]:
        if self._queries is None:
            self._queries = load_queries(self.config.dataset_root, self._verified_manual())
        return self._queries

    @staticmethod
    def _dataset_summary(manifest: ManualManifest) -> dict[str, object]:
        return {
            "dataset_id": manifest.dataset_id,
            "title": manifest.title,
            "revision": manifest.revision,
            "document_identifier": manifest.document_identifier,
            "ntrs_id": manifest.ntrs_id,
            "ntrs_record_url": manifest.ntrs_record_url,
            "official_pdf_url": manifest.official_pdf_url,
            "distribution": manifest.distribution,
            "rights_determination": manifest.rights_determination,
            "contains_third_party_material": manifest.contains_third_party_material,
            "attribution": manifest.attribution,
            "authors": [author.public_dict() for author in manifest.authors],
            "language": manifest.language,
            "render_statement": manifest.render_statement,
            "endorsement_statement": manifest.endorsement_statement,
            "renderer": manifest.renderer,
            "renderer_version": manifest.renderer_version,
            "render_dpi": manifest.render_dpi,
            "pdf_page_index_base": manifest.pdf_page_index_base,
            "source_pdf_page_count": manifest.source_pdf_page_count,
            "source_pdf_bytes": manifest.source_pdf_bytes,
            "page_count": manifest.page_count,
            "manifest_sha256": manifest.manifest_sha256,
            "pdf_sha256": manifest.pdf_sha256,
            "queries_sha256": manifest.queries_sha256,
        }

    def status(self) -> dict[str, object]:
        """Inspect local state and the exact project-GA namespace without model inference."""

        audit = self.repository.raw_audit()
        manual = self._verified_manual()
        queries = self._verified_queries()
        cache_exists = self.config.embedding_cache_root.is_dir()
        ready = self.repository.owns_collection
        return {
            "status": "ready" if ready else "not_prepared",
            "implementation_status": "implemented",
            "runtime_ready": ready,
            "execution_path": EXECUTION_PATH,
            "collection_name": COLLECTION_NAME,
            "index_name": INDEX_NAME,
            "anns_field": ANNS_FIELD,
            "metric_type": METRIC_TYPE,
            "dataset_exists": True,
            "embedding_cache_exists": cache_exists,
            "dataset": self._dataset_summary(manual),
            "query_presets": [query.public_dict() for query in queries],
            "last_prepare": self._last_prepare,
            "prepare_timings": self._prepare_timings,
            "model": self.model.runtime_dict(),
            "milvus": audit.public_dict(),
        }

    def manual(self) -> dict[str, object]:
        """Return the verified dataset manifest and public natural-query presets."""

        manifest = self._verified_manual()
        return {
            "generated": False,
            **manifest.public_dict(),
            "query_presets": [query.public_dict() for query in self._verified_queries()],
        }

    def page_image(self, page_id: str) -> Path:
        """Resolve one allowlisted generated page image."""

        manifest = self._verified_manual()
        return resolve_page_image(self.config.dataset_root, manifest, page_id)

    def prepare(self) -> dict[str, object]:
        """Verify inputs, cache real page vectors, and prepare the exact Collection."""

        started = time.perf_counter()
        dataset_started = time.perf_counter()
        manifest = load_manual(self.config.dataset_root)
        queries = load_queries(self.config.dataset_root, manifest)
        self._manual = manifest
        self._queries = queries
        dataset_verify_ms = (time.perf_counter() - dataset_started) * 1000

        identity_started = time.perf_counter()
        identity = self.model.identity
        model_identity_ms = (time.perf_counter() - identity_started) * 1000

        cache_read_started = time.perf_counter()
        cache_load: CacheLoad = self.cache.load(manifest, identity)
        cache_read_ms = (time.perf_counter() - cache_read_started) * 1000
        page_inference_ms = 0.0
        cache_write_ms = 0.0
        if cache_load.hit:
            assert cache_load.vectors is not None and cache_load.manifest is not None
            vectors = cache_load.vectors
            cache_manifest = cache_load.manifest
            page_inference = cache_manifest["inference"]
        else:
            inference_started = time.perf_counter()
            vectors, report = self.model.embed_pages(
                dataset_root=self.config.dataset_root,
                manifest=manifest,
            )
            page_inference_ms = (time.perf_counter() - inference_started) * 1000
            page_inference = report.public_dict()
            cache_write_started = time.perf_counter()
            cache_manifest = self.cache.save(
                manual=manifest,
                identity=identity,
                vectors=vectors,
                inference_report=page_inference,
            )
            cache_write_ms = (time.perf_counter() - cache_write_started) * 1000
        self._page_vectors = vectors
        self._cache_manifest = cache_manifest

        milvus_started = time.perf_counter()
        if self.repository.owns_collection:
            repository_report = {
                "status": "already_prepared_by_this_process",
                "audit": self.repository.raw_audit().public_dict(),
            }
        else:
            adopted = self.repository.adopt(page_count=len(manifest.pages))
            if adopted:
                repository_report = {
                    "status": "adopted_pre_existing_collection",
                    "audit": self.repository.raw_audit().public_dict(),
                }
            else:
                repository_report = self.repository.prepare(manifest, vectors).public_dict()
        milvus_prepare_ms = (time.perf_counter() - milvus_started) * 1000
        total_ms = (time.perf_counter() - started) * 1000
        timings = {
            "dataset_verify_ms": round(dataset_verify_ms, 3),
            "model_identity_ms": round(model_identity_ms, 3),
            "cache_read_ms": round(cache_read_ms, 3),
            "page_inference_ms": round(page_inference_ms, 3),
            "cache_write_ms": round(cache_write_ms, 3),
            "milvus_prepare_ms": round(milvus_prepare_ms, 3),
            "total_ms": round(total_ms, 3),
        }
        self._last_prepare = {"cache_hit": cache_load.hit, "timings": timings}
        self._prepare_timings["warm" if cache_load.hit else "cold"] = timings
        return {
            "status": "prepared",
            "implementation_status": "implemented",
            "runtime_ready": True,
            "manual_generated": False,
            "manual": manifest.public_dict(),
            "query_presets": [query.public_dict() for query in queries],
            "cache": {
                "hit": cache_load.hit,
                "cache_key": cache_manifest["cache_key"],
                "tensor_sha256": cache_manifest["tensor_sha256"],
                "pages": cache_manifest["pages"],
            },
            "model": self.model.runtime_dict(),
            "page_inference": page_inference,
            "milvus": repository_report,
            "timings": timings,
            "prepare_timings": self._prepare_timings,
            "latency_ms": timings["total_ms"],
        }

    def search(self, query: str) -> dict[str, object]:
        """Search with real EmbeddingList and compare against the local ColPali scorer."""

        text = query.strip()
        if not text or len(text) > 512:
            raise ServiceContractError("Query must contain between 1 and 512 characters")
        if not self.repository.owns_collection or self._page_vectors is None:
            raise ServiceContractError("Backend must be prepared before search")
        manifest = self._verified_manual()
        started = time.perf_counter()
        query_started = time.perf_counter()
        query_vectors, query_report = self.model.embed_query(text)
        query_inference_ms = (time.perf_counter() - query_started) * 1000
        milvus_started = time.perf_counter()
        milvus_hits = self.repository.search(query_vectors, limit=manifest.page_count)
        milvus_search_ms = (time.perf_counter() - milvus_started) * 1000
        local_started = time.perf_counter()
        local_scores = self.model.score_pages(query_vectors, self._page_vectors)
        local_score_ms = (time.perf_counter() - local_started) * 1000
        if len(local_scores) != manifest.page_count:
            raise ScoreComparisonError("Local scorer result count differs from dataset page count")
        if not np.isfinite(local_scores).all():
            raise ScoreComparisonError("Local scorer returned a non-finite page score")

        local_order = sorted(
            range(manifest.page_count),
            key=lambda index: float(local_scores[index]),
            reverse=True,
        )
        local_rank_by_page = {
            manifest.pages[page_index].page_id: rank
            for rank, page_index in enumerate(local_order, 1)
        }
        local_score_by_page = {
            page.page_id: float(local_scores[index]) for index, page in enumerate(manifest.pages)
        }
        milvus_page_ids = [hit.page_id for hit in milvus_hits]
        local_page_ids = [manifest.pages[index].page_id for index in local_order]
        if (
            len(milvus_page_ids) != manifest.page_count
            or len(set(milvus_page_ids)) != manifest.page_count
            or set(milvus_page_ids) != set(local_page_ids)
        ):
            raise ScoreComparisonError("Milvus result page set differs from the dataset")
        page_by_id = {page.page_id: page for page in manifest.pages}
        results: list[dict[str, object]] = []
        score_deltas: list[float] = []
        for milvus_rank, hit in enumerate(milvus_hits, 1):
            page = page_by_id[hit.page_id]
            metadata_matches = (
                hit.pdf_page_index == page.pdf_page_index
                and hit.printed_page == page.printed_page
                and hit.title == page.title
                and hit.section == page.section
                and hit.document_identifier == manifest.document_identifier
                and hit.ntrs_id == manifest.ntrs_id
                and hit.ntrs_record_url == manifest.ntrs_record_url
                and hit.distribution == manifest.distribution
                and hit.rights_determination == manifest.rights_determination
                and hit.contains_third_party_material == manifest.contains_third_party_material
                and hit.pdf_sha256 == manifest.pdf_sha256
                and hit.image_sha256 == page.image_sha256
            )
            if not metadata_matches:
                raise ScoreComparisonError(
                    f"Milvus metadata differs from the dataset: {hit.page_id}"
                )
            local_score = local_score_by_page[hit.page_id]
            delta = abs(hit.score - local_score)
            if not np.isfinite(hit.score) or not np.isfinite(delta):
                raise ScoreComparisonError(f"Milvus returned a non-finite score: {hit.page_id}")
            score_deltas.append(delta)
            results.append(
                {
                    **hit.public_dict(),
                    "milvus_rank": milvus_rank,
                    "local_rank": local_rank_by_page[hit.page_id],
                    "local_score": local_score,
                    "score_delta": delta,
                    "page_image_url": f"/api/v1/pages/{hit.page_id}",
                    "image_width": page.width,
                    "image_height": page.height,
                    "evidence_label": ("Unmodified 144-DPI render from the official NTRS PDF"),
                }
            )
        max_delta = max(score_deltas, default=float("inf"))
        ranking_equal = milvus_page_ids == local_page_ids
        score_within_tolerance = max_delta <= SCORE_ABSOLUTE_TOLERANCE
        if not ranking_equal or not score_within_tolerance:
            raise ScoreComparisonError(
                "Milvus MAX_SIM output differs from the complete local scorer comparison"
            )
        explanation_started = time.perf_counter()
        page_index_by_id = {page.page_id: index for index, page in enumerate(manifest.pages)}
        explanations = [
            self.model.explain_page(
                query=text,
                query_vectors=query_vectors,
                page_vectors=self._page_vectors[page_index_by_id[hit.page_id]],
                page_image=resolve_page_image(
                    self.config.dataset_root,
                    manifest,
                    hit.page_id,
                ),
                page_id=hit.page_id,
            )
            for hit in milvus_hits[:3]
        ]
        local_explanation_ms = (time.perf_counter() - explanation_started) * 1000
        total_ms = (time.perf_counter() - started) * 1000
        timings = {
            "query_inference_ms": round(query_inference_ms, 3),
            "milvus_search_ms": round(milvus_search_ms, 3),
            "local_score_ms": round(local_score_ms, 3),
            "local_explanation_ms": round(local_explanation_ms, 3),
            "total_ms": round(total_ms, 3),
        }
        return {
            "status": "passed",
            "execution_path": EXECUTION_PATH,
            "query": text,
            "collection_name": COLLECTION_NAME,
            "index_name": INDEX_NAME,
            "anns_field": ANNS_FIELD,
            "metric_type": METRIC_TYPE,
            "dataset": self._dataset_summary(manifest),
            "query_embedding": query_report.public_dict(),
            "page_vector_counts": [int(item.shape[0]) for item in self._page_vectors],
            "milvus_page_order": milvus_page_ids,
            "local_page_order": local_page_ids,
            "ranking_equal": ranking_equal,
            "score_absolute_tolerance": SCORE_ABSOLUTE_TOLERANCE,
            "max_score_delta": max_delta,
            "score_within_tolerance": score_within_tolerance,
            "results": results,
            "local_explanations": explanations,
            "model": self.model.runtime_dict(),
            "timings": timings,
            "latency_ms": timings["total_ms"],
        }

    def cleanup(self) -> dict[str, object]:
        """Clean the exact Collection and derived cache while preserving source data."""

        milvus = self.repository.cleanup()
        self.model.unload()
        cache_removed = self.cache.purge()
        self._manual = None
        self._queries = None
        self._page_vectors = None
        self._cache_manifest = None
        if self.config.runtime_root.is_dir() and not any(self.config.runtime_root.iterdir()):
            self.config.runtime_root.rmdir()
        return {
            "status": "clean",
            "milvus": milvus.public_dict(),
            "embedding_cache_removed": cache_removed,
            "dataset_source_preserved": self.config.dataset_root.is_dir(),
            "runtime_root_exists": self.config.runtime_root.exists(),
            "model_loaded": self.model.loaded,
        }
