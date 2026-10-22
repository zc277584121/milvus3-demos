"""Orchestrate data, embedding, and the two Milvus search paths."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

from movie_discovery_demo.config import (
    AGG_TOP_HITS_SIZE,
    DEFAULT_FACET,
    DEFAULT_TOP_K,
    FACET_FIELDS,
    MAX_RUNTIME_FILTER,
    MAX_TOP_K,
    RuntimeConfig,
)
from movie_discovery_demo.data import DatasetBundle, build_dataset
from movie_discovery_demo.embedding import BgeM3OnnxEmbedder, EmbeddingBatchReport, ModelIdentity
from movie_discovery_demo.repository import (
    GroupedBucket,
    MovieHit,
    MovieRepository,
    SearchContractError,
)


class ServiceContractError(RuntimeError):
    """Raised when the end-to-end backend state is not ready for an operation."""


class EmbedderLike(Protocol):
    @property
    def loaded(self) -> bool: ...

    @property
    def identity(self) -> ModelIdentity: ...

    def load(self) -> None: ...

    def embed(self, texts: list[str]) -> tuple[list[list[float]], EmbeddingBatchReport]: ...

    def unload(self) -> None: ...


class RepositoryLike(Protocol):
    def status(self) -> dict[str, object]: ...

    def prepare(self, movies, synopsis_vectors): ...

    def list_search(
        self, *, query_vector, movies_by_id, limit, filter=None
    ) -> tuple[MovieHit, ...]: ...

    def aggregated_search(
        self,
        *,
        query_vector,
        movies_by_id,
        facet,
        filter=None,
        top_hits_size=AGG_TOP_HITS_SIZE,
        root_size=6,
    ) -> tuple[GroupedBucket, ...]: ...

    def cleanup(self) -> dict[str, object]: ...


@dataclass(frozen=True)
class SearchReport:
    query: str
    view: str
    facet: str
    facet_field: str
    filter: str | None
    limit: int
    latency_ms: float
    list_results: tuple[MovieHit, ...]
    grouped_results: tuple[GroupedBucket, ...]
    retained_candidate_count: int
    semantics: str

    def public_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "status": "passed",
            "query": self.query,
            "view": self.view,
            "facet": self.facet,
            "facet_field": self.facet_field,
            "filter": self.filter,
            "limit": self.limit,
            "latency_ms": round(self.latency_ms, 2),
            "score_semantics": "COSINE similarity, not a probability",
            "list": {"results": [hit.public_dict() for hit in self.list_results]},
            "grouped": {
                "buckets": [bucket.public_dict() for bucket in self.grouped_results],
            },
            "retained_candidate_count": self.retained_candidate_count,
            "semantics": self.semantics,
        }
        return payload


class MovieDiscoveryService:
    """Own the complete backend lifecycle within one process."""

    def __init__(
        self,
        *,
        config: RuntimeConfig | None = None,
        embedder: EmbedderLike | None = None,
        repository: RepositoryLike | None = None,
    ) -> None:
        self.config = config or RuntimeConfig.from_environment()
        self.embedder = embedder or BgeM3OnnxEmbedder(self.config)
        self.repository = repository or MovieRepository()
        self._bundle: DatasetBundle | None = None
        self._movies_by_id: dict[str, object] | None = None
        self._last_prepare: dict[str, object] | None = None

    def _dataset(self) -> DatasetBundle:
        if self._bundle is None:
            self._bundle = build_dataset(self.config)
        return self._bundle

    def _movies_map(self):
        if self._movies_by_id is None:
            self._movies_by_id = {movie.movie_id: movie for movie in self._dataset().movies}
        return self._movies_by_id

    def model_identity(self) -> dict[str, object]:
        return self.embedder.identity.public_dict()

    def status(self) -> dict[str, object]:
        repository_status = self.repository.status()
        dataset = self._dataset()
        return {
            "status": "ready" if repository_status.get("collection_exists") else "not_prepared",
            "movie_count": dataset.movie_count,
            "model": {
                **self.model_identity(),
                "loaded": self.embedder.loaded,
                "cache_available": getattr(self.embedder, "cache_available", False),
            },
            "dataset": {
                "dataset_id": dataset.dataset_id,
                "dataset_version": dataset.dataset_version,
                "movie_count": dataset.movie_count,
                "movies_sha256": dataset.movies_sha256,
                "presets_sha256": dataset.presets_sha256,
            },
            "milvus": repository_status,
            "facets": [
                {"id": facet, "field": field[0], "label": field[1]}
                for facet, field in FACET_FIELDS.items()
            ],
            "query_presets": [preset.public_dict() for preset in dataset.presets],
            "poster_placeholders": True,
        }

    def prepare(self) -> dict[str, object]:
        dataset = self._dataset()
        self.embedder.load()

        synopsis_texts = [movie.synopsis for movie in dataset.movies]
        started = time.perf_counter()
        synopsis_vectors, report = self.embedder.embed(synopsis_texts)
        embed_seconds = time.perf_counter() - started

        provisioning = self.repository.prepare(dataset.movies, synopsis_vectors)
        self._movies_by_id = {movie.movie_id: movie for movie in dataset.movies}
        self._last_prepare = {
            "status": "prepared",
            "embedding": {
                "vector_count": len(synopsis_vectors),
                "vector_dimension": report.vector_dimension,
                "embedding_seconds": round(embed_seconds, 3),
            },
            "milvus": provisioning.public_dict(),
        }
        return self._last_prepare

    def _resolve_query(self, query: str) -> str:
        query = query.strip()
        if not query:
            raise SearchContractError("Query text must not be empty")
        if len(query) > 2000:
            raise SearchContractError("Query text is too long")
        return query

    def _resolve_view(self, view: str) -> str:
        if view not in {"grouped", "list"}:
            raise SearchContractError("View must be 'grouped' or 'list'")
        return view

    def _resolve_facet(self, facet: str) -> str:
        if facet not in FACET_FIELDS:
            raise SearchContractError(f"Unknown facet: {facet}")
        return facet

    def _resolve_limit(self, limit: int) -> int:
        if not 1 <= limit <= MAX_TOP_K:
            raise SearchContractError(f"Search limit must be between 1 and {MAX_TOP_K}")
        return limit

    def _resolve_filter(self, max_runtime: int | None) -> str | None:
        if max_runtime is None:
            return None
        if not 1 <= max_runtime <= MAX_RUNTIME_FILTER:
            raise SearchContractError(
                f"max_runtime must be between 1 and {MAX_RUNTIME_FILTER} minutes"
            )
        return f"runtime_minutes <= {max_runtime}"

    def search(
        self,
        *,
        query: str,
        view: str = "grouped",
        facet: str = DEFAULT_FACET,
        limit: int = DEFAULT_TOP_K,
        max_runtime: int | None = None,
    ) -> SearchReport:
        resolved_query = self._resolve_query(query)
        resolved_view = self._resolve_view(view)
        resolved_facet = self._resolve_facet(facet)
        resolved_limit = self._resolve_limit(limit)
        filter_expr = self._resolve_filter(max_runtime)

        movies_by_id = self._movies_map()

        started = time.perf_counter()
        vectors, report = self.embedder.embed([resolved_query])
        query_vector = vectors[0]
        embed_ms = (time.perf_counter() - started) * 1000.0

        search_started = time.perf_counter()
        if resolved_view == "list":
            list_results = self.repository.list_search(
                query_vector=query_vector,
                movies_by_id=movies_by_id,
                limit=resolved_limit,
                filter=filter_expr,
            )
            grouped_results: tuple[GroupedBucket, ...] = ()
            retained = len(list_results)
        else:
            grouped_results = self.repository.aggregated_search(
                query_vector=query_vector,
                movies_by_id=movies_by_id,
                facet=resolved_facet,
                filter=filter_expr,
            )
            list_results = ()
            retained = sum(bucket.count for bucket in grouped_results)
        total_ms = (time.perf_counter() - search_started) * 1000.0 + embed_ms

        facet_field = FACET_FIELDS[resolved_facet][0]
        semantics = (
            "Scores are COSINE similarity between BGE-M3 query and synopsis vectors. "
            "Group counts and candidate lists cover only the candidates retained "
            "during retrieval, not every movie in the catalog. When Search "
            "Aggregation is enabled, the plain hit list is empty by design."
        )
        return SearchReport(
            query=resolved_query,
            view=resolved_view,
            facet=resolved_facet,
            facet_field=facet_field,
            filter=filter_expr,
            limit=resolved_limit,
            latency_ms=total_ms,
            list_results=list_results,
            grouped_results=grouped_results,
            retained_candidate_count=retained,
            semantics=semantics,
        )

    def cleanup(self) -> dict[str, object]:
        milvus = self.repository.cleanup()
        self.embedder.unload()
        self._bundle = None
        self._movies_by_id = None
        self._last_prepare = None
        return {
            "status": "clean",
            "milvus": milvus,
            "model_loaded": self.embedder.loaded,
        }
