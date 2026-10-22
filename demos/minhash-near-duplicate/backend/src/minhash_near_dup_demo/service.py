"""Dataset, Milvus, and diff orchestration for the MinHash demo."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

from minhash_near_dup_demo.config import (
    CAPABILITY,
    COLLECTION_NAME,
    CONTENT_MAX_LENGTH,
    DEFAULT_THRESHOLD,
    DEFAULT_TOP_K,
    EXECUTION_PATH,
    INDEX_NAME,
    MAX_TOP_K,
    METRIC_TYPE,
    SIGNATURE_FIELD,
    TEXT_FIELD,
    THRESHOLD_MAX,
    THRESHOLD_MIN,
    RuntimeConfig,
)
from minhash_near_dup_demo.data import DatasetBundle, PresetRecord, build_dataset
from minhash_near_dup_demo.diff import compute_diff
from minhash_near_dup_demo.repository import (
    CleanupResult,
    MilvusHit,
    MinHashRepository,
    PrepareResult,
    RepositoryContractError,
    SearchContractError,
)


class ServiceContractError(RuntimeError):
    """Raised when the end-to-end backend state is not ready for an operation."""


class RepositoryLike(Protocol):
    @property
    def owns_collection(self) -> bool: ...

    def raw_audit(self): ...

    def prepare(self, documents) -> PrepareResult: ...

    def adopt(self, document_count: int) -> bool: ...

    def search(self, query_text: str, *, limit: int) -> tuple[MilvusHit, ...]: ...

    def cleanup(self) -> CleanupResult: ...


@dataclass(frozen=True)
class SearchReport:
    query: str
    threshold: float
    limit: int
    above_threshold_count: int
    results: tuple[MilvusHit, ...]
    diffs: tuple[dict[str, object], ...]
    latency_ms: float
    semantics: str

    def public_dict(self) -> dict[str, object]:
        return {
            "status": "passed",
            "execution_path": EXECUTION_PATH,
            "query": self.query,
            "collection_name": COLLECTION_NAME,
            "index_name": INDEX_NAME,
            "text_field": TEXT_FIELD,
            "signature_field": SIGNATURE_FIELD,
            "metric_type": METRIC_TYPE,
            "threshold": self.threshold,
            "limit": self.limit,
            "above_threshold_count": self.above_threshold_count,
            "results": [hit.public_dict(threshold=self.threshold) for hit in self.results],
            "diffs": list(self.diffs),
            "semantics": self.semantics,
            "latency_ms": round(self.latency_ms, 2),
        }


class MinHashService:
    """Own the complete backend lifecycle within one process."""

    def __init__(
        self,
        *,
        config: RuntimeConfig | None = None,
        repository: RepositoryLike | None = None,
    ) -> None:
        self.config = config or RuntimeConfig.from_environment()
        self.repository = repository or MinHashRepository()
        self._bundle: DatasetBundle | None = None
        self._last_prepare: dict[str, object] | None = None

    def _dataset(self) -> DatasetBundle:
        if self._bundle is None:
            self._bundle = build_dataset(self.config)
        return self._bundle

    def _presets(self) -> tuple[PresetRecord, ...]:
        return self._dataset().presets

    def status(self) -> dict[str, object]:
        dataset = self._dataset()
        audit = self.repository.raw_audit()
        return {
            "status": "ready" if audit.target_exists else "not_prepared",
            "implementation_status": "implemented",
            "execution_path": EXECUTION_PATH,
            "capability": CAPABILITY,
            "collection_name": COLLECTION_NAME,
            "index_name": INDEX_NAME,
            "text_field": TEXT_FIELD,
            "signature_field": SIGNATURE_FIELD,
            "metric_type": METRIC_TYPE,
            "dataset": {
                "dataset_id": dataset.dataset_id,
                "dataset_version": dataset.dataset_version,
                "document_count": dataset.document_count,
                "documents_sha256": dataset.documents_sha256,
                "presets_sha256": dataset.presets_sha256,
            },
            "query_presets": [preset.public_dict() for preset in self._presets()],
            "milvus": audit.public_dict(),
            "last_prepare": self._last_prepare,
        }

    def prepare(self) -> dict[str, object]:
        dataset = self._dataset()
        result = self.repository.prepare(dataset.documents)
        self._last_prepare = result.public_dict()
        return self._last_prepare

    def adopt(self) -> bool:
        """Adopt a pre-existing exact Collection on startup when present."""
        return self.repository.adopt(self._dataset().document_count)

    def _resolve_query(self, query: str) -> str:
        query = query.strip()
        if not query:
            raise SearchContractError("Query text must not be empty")
        if len(query) > CONTENT_MAX_LENGTH:
            raise SearchContractError(f"Query text must be at most {CONTENT_MAX_LENGTH} characters")
        return query

    def _resolve_threshold(self, threshold: float) -> float:
        if not THRESHOLD_MIN <= threshold <= THRESHOLD_MAX:
            raise SearchContractError(
                f"Threshold must be between {THRESHOLD_MIN} and {THRESHOLD_MAX}"
            )
        return round(threshold, 3)

    def _resolve_limit(self, limit: int) -> int:
        if limit < 1 or limit > MAX_TOP_K:
            raise SearchContractError(f"Limit must be between 1 and {MAX_TOP_K}")
        return limit

    def search(
        self,
        *,
        query: str,
        threshold: float = DEFAULT_THRESHOLD,
        limit: int = DEFAULT_TOP_K,
        selected_doc_id: str | None = None,
    ) -> SearchReport:
        query = self._resolve_query(query)
        threshold = self._resolve_threshold(threshold)
        limit = self._resolve_limit(limit)

        if not self.repository.owns_collection:
            # Startup adopt keeps a restart search-ready; only prepare from
            # scratch when no exact Collection exists.
            if not self.repository.adopt(self._dataset().document_count):
                self.prepare()

        started = time.perf_counter()
        hits = self.repository.search(query, limit=limit)
        latency_ms = (time.perf_counter() - started) * 1000

        dataset = self._dataset()
        by_id = {document.doc_id: document for document in dataset.documents}
        diffs: list[dict[str, object]] = []
        for hit in hits:
            document = by_id.get(hit.doc_id)
            if document is None:
                raise RepositoryContractError(f"Unknown doc_id in search hit: {hit.doc_id}")
            if document.content != hit.content:
                raise RepositoryContractError(
                    f"Retrieved content differs from the dataset for {hit.doc_id}"
                )
            diffs.append(
                compute_diff(
                    incoming_text=query,
                    document_id=hit.doc_id,
                    document_text=document.content,
                ).public_dict()
            )

        above_threshold_count = sum(1 for hit in hits if hit.similarity() >= threshold)
        return SearchReport(
            query=query,
            threshold=threshold,
            limit=limit,
            above_threshold_count=above_threshold_count,
            results=hits,
            diffs=tuple(diffs),
            latency_ms=latency_ms,
            semantics=(
                "Estimated Jaccard similarity between MinHash signatures. "
                "Milvus's MHJACCARD metric returns the estimated Jaccard "
                "similarity directly (larger is more similar; identical "
                "signatures score 1.0). It is not a probability or a model "
                "confidence. LSH may miss near-duplicates, so an empty result "
                "never proves that no duplicate exists."
            ),
        )

    def cleanup(self) -> dict[str, object]:
        return self.repository.cleanup().public_dict()
