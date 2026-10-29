"""Orchestrate data, embedding, and the two Milvus search paths."""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from movie_discovery_demo.config import (
    AGG_TOP_HITS_SIZE,
    BROWSE_MAX_PAGE_SIZE,
    BROWSE_PAGE_SIZE,
    BROWSE_SORT_DIRECTIONS,
    BROWSE_SORT_FIELDS,
    DEFAULT_BROWSE_DIRECTION,
    DEFAULT_BROWSE_SORT,
    DEFAULT_EXACT_METRICS,
    DEFAULT_FACET,
    DEFAULT_RANKED_SORT,
    DEFAULT_TOP_K,
    EXACT_METRICS,
    FACET_FIELDS,
    MAX_RUNTIME_FILTER,
    MAX_TOP_K,
    RANKED_SORT_FIELDS,
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

    def browse_sorted(
        self,
        *,
        movies_by_id,
        sort_field,
        direction,
        limit,
        offset,
        filter=None,
    ) -> tuple[tuple[object, ...], int]: ...

    def query_aggregate(
        self,
        *,
        facet_field,
        metric_fields,
        filter=None,
    ) -> tuple[dict[str, object], ...]: ...

    def ranked_search(
        self,
        *,
        query_vector,
        movies_by_id,
        limit,
        sort_field=None,
        direction="desc",
        filter=None,
    ) -> tuple[MovieHit, ...]: ...

    def cleanup(self) -> dict[str, object]: ...


@dataclass(frozen=True)
class ExploreReport:
    """One cell of the 2x2 capability matrix.

    `api` is which Milvus entry point ran, `mode` is which capability it
    exercised. The payload is a discriminated union keyed on those two fields so
    the page never has to guess which shape it received.
    """

    api: str
    mode: str
    filter: str | None
    latency_ms: float
    semantics: str
    payload: dict[str, object]

    def public_dict(self) -> dict[str, object]:
        return {
            "status": "passed",
            "api": self.api,
            "mode": self.mode,
            "filter": self.filter,
            "latency_ms": round(self.latency_ms, 2),
            "score_semantics": "COSINE similarity, not a probability",
            "semantics": self.semantics,
            **self.payload,
        }


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
        posters_dir = self.config.data_root / "posters"
        poster_count = len(list(posters_dir.glob("m*.jpg"))) if posters_dir.is_dir() else 0
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
            "poster_placeholders": poster_count < dataset.movie_count,
            "poster_count": poster_count,
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

    # ------------------------------------------------------------------
    # The 2x2 capability matrix
    # ------------------------------------------------------------------
    def explore_query_sort(
        self,
        *,
        sort: str = DEFAULT_BROWSE_SORT,
        direction: str = DEFAULT_BROWSE_DIRECTION,
        page: int = 1,
        page_size: int = BROWSE_PAGE_SIZE,
        max_runtime: int | None = None,
    ) -> ExploreReport:
        """query + ORDER BY: an exact, paged listing of the whole catalog."""
        if sort not in BROWSE_SORT_FIELDS:
            raise SearchContractError(f"Unknown browse sort: {sort}")
        if direction not in BROWSE_SORT_DIRECTIONS:
            raise SearchContractError(f"Unsupported sort direction: {direction}")
        if page < 1:
            raise SearchContractError("page must be >= 1")
        if not 1 <= page_size <= BROWSE_MAX_PAGE_SIZE:
            raise SearchContractError(f"page_size must be between 1 and {BROWSE_MAX_PAGE_SIZE}")
        filter_expr = self._resolve_filter(max_runtime)
        sort_field, sort_label = BROWSE_SORT_FIELDS[sort]
        offset = (page - 1) * page_size

        started = time.perf_counter()
        movies, total = self.repository.browse_sorted(
            movies_by_id=self._movies_map(),
            sort_field=sort_field,
            direction=direction,
            limit=page_size,
            offset=offset,
            filter=filter_expr,
        )
        elapsed = (time.perf_counter() - started) * 1000.0

        total_pages = max(1, (total + page_size - 1) // page_size)
        return ExploreReport(
            api="query",
            mode="sort",
            filter=filter_expr,
            latency_ms=elapsed,
            semantics=(
                "Server-side ORDER BY over the whole catalog; no vector is used. "
                "This is a strict sort that also returns a paged listing and an "
                "exact total, which a search-path sort cannot do."
            ),
            payload={
                "sort": sort,
                "sort_field": sort_field,
                "sort_label": sort_label,
                "direction": direction,
                "page": page,
                "page_size": page_size,
                "total_pages": total_pages,
                "total": total,
                "results": [
                    {
                        "movie_id": movie.movie_id,
                        "title": movie.title,
                        "synopsis": movie.synopsis,
                        "primary_genre": movie.primary_genre,
                        "release_year": movie.release_year,
                        "decade": movie.decade,
                        "region": movie.region,
                        "runtime_minutes": movie.runtime_minutes,
                        "poster_path": movie.poster_path,
                        "poster_palette": list(movie.poster_palette),
                    }
                    for movie in movies
                ],
            },
        )

    def explore_query_group(
        self,
        *,
        facet: str = DEFAULT_FACET,
        metrics: Sequence[str] = DEFAULT_EXACT_METRICS,
        max_runtime: int | None = None,
    ) -> ExploreReport:
        """query + GROUP BY: exact aggregates over the whole catalog."""
        facet = self._resolve_facet(facet)
        if not metrics:
            raise SearchContractError("At least one metric is required")
        unknown = [name for name in metrics if name not in EXACT_METRICS]
        if unknown:
            raise SearchContractError(f"Unknown metrics: {', '.join(unknown)}")
        filter_expr = self._resolve_filter(max_runtime)
        facet_field = FACET_FIELDS[facet][0]
        metric_fields = [EXACT_METRICS[name][0] for name in metrics]

        started = time.perf_counter()
        rows = self.repository.query_aggregate(
            facet_field=facet_field,
            metric_fields=metric_fields,
            filter=filter_expr,
        )
        elapsed = (time.perf_counter() - started) * 1000.0

        groups = []
        for row in rows:
            values = row.get("values") or {}
            cells = {}
            for name in metrics:
                raw = values.get(EXACT_METRICS[name][0])
                # Aggregates come back as numpy scalars or averages; normalise
                # to JSON-safe numbers, keeping integers integral.
                if raw is None:
                    cells[name] = None
                elif isinstance(raw, float) and raw.is_integer():
                    cells[name] = int(raw)
                elif isinstance(raw, float):
                    cells[name] = round(raw, 2)
                else:
                    cells[name] = raw
            groups.append({"key": row.get("key"), "metrics": cells})

        return ExploreReport(
            api="query",
            mode="group",
            filter=filter_expr,
            latency_ms=elapsed,
            semantics=(
                "GROUP BY with count/sum/avg/min/max runs over the entire match "
                "set, so these figures are exact. Each group carries aggregates "
                "only; it does not retrieve individual films."
            ),
            payload={
                "facet": facet,
                "facet_field": facet_field,
                "metrics": [
                    {"id": name, "field": EXACT_METRICS[name][0], "label": EXACT_METRICS[name][1]}
                    for name in metrics
                ],
                "groups": groups,
                "group_count": len(groups),
            },
        )

    def explore_search_sort(
        self,
        *,
        query: str,
        sort: str = DEFAULT_RANKED_SORT,
        direction: str = "desc",
        limit: int = DEFAULT_TOP_K,
        max_runtime: int | None = None,
    ) -> ExploreReport:
        """search + ORDER BY: re-order the retained top-K only."""
        if sort not in RANKED_SORT_FIELDS:
            raise SearchContractError(f"Unknown ranked sort: {sort}")
        resolved_query = self._resolve_query(query)
        resolved_limit = self._resolve_limit(limit)
        filter_expr = self._resolve_filter(max_runtime)

        started = time.perf_counter()
        vectors, _ = self.embedder.embed([resolved_query])
        embed_ms = (time.perf_counter() - started) * 1000.0

        search_started = time.perf_counter()
        if sort == "score":
            hits = self.repository.ranked_search(
                query_vector=vectors[0],
                movies_by_id=self._movies_map(),
                limit=resolved_limit,
                filter=filter_expr,
            )
        else:
            sort_field = RANKED_SORT_FIELDS[sort][0]
            hits = self.repository.ranked_search(
                query_vector=vectors[0],
                movies_by_id=self._movies_map(),
                limit=resolved_limit,
                sort_field=sort_field,
                direction=direction,
                filter=filter_expr,
            )
        elapsed = (time.perf_counter() - search_started) * 1000.0 + embed_ms

        scores = [round(hit.score, 4) for hit in hits]
        monotonic = all(a >= b for a, b in zip(scores, scores[1:], strict=False))
        sort_field, sort_label = RANKED_SORT_FIELDS[sort]

        return ExploreReport(
            api="search",
            mode="sort",
            filter=filter_expr,
            latency_ms=elapsed,
            semantics=(
                "The vector search retains a fixed top-K and the scalar sort "
                "only re-orders those retained hits: the candidate set is "
                "unchanged. This is a strict sort, not a blended score, and it "
                "cannot produce a sorted list of every match."
            ),
            payload={
                "query": resolved_query,
                "sort": sort,
                "sort_field": sort_field,
                "sort_label": sort_label,
                "direction": direction,
                "limit": resolved_limit,
                "retained_candidate_count": len(hits),
                "score_monotonic": monotonic,
                "results": [hit.public_dict() for hit in hits],
            },
        )

    def explore_search_group(
        self,
        *,
        query: str,
        facet: str = DEFAULT_FACET,
        limit: int = DEFAULT_TOP_K,
        max_runtime: int | None = None,
    ) -> ExploreReport:
        """search + aggregation: facet buckets with metrics and best hits."""
        facet = self._resolve_facet(facet)
        resolved_query = self._resolve_query(query)
        resolved_limit = self._resolve_limit(limit)
        filter_expr = self._resolve_filter(max_runtime)
        facet_field = FACET_FIELDS[facet][0]

        started = time.perf_counter()
        vectors, _ = self.embedder.embed([resolved_query])
        embed_ms = (time.perf_counter() - started) * 1000.0

        search_started = time.perf_counter()
        buckets = self.repository.aggregated_search_with_metrics(
            query_vector=vectors[0],
            movies_by_id=self._movies_map(),
            facet=facet,
            filter=filter_expr,
        )
        elapsed = (time.perf_counter() - search_started) * 1000.0 + embed_ms
        retained = sum(int(bucket["count"]) for bucket in buckets)

        return ExploreReport(
            api="search",
            mode="group",
            filter=filter_expr,
            latency_ms=elapsed,
            semantics=(
                "Facet buckets are built over the candidates retained during "
                "vector retrieval, so counts are approximate: each bucket can "
                "only hold as many retained hits as its TopHits budget. Use the "
                "query aggregation cell for exact figures."
            ),
            payload={
                "query": resolved_query,
                "facet": facet,
                "facet_field": facet_field,
                "limit": resolved_limit,
                "retained_candidate_count": retained,
                "buckets": buckets,
            },
        )
