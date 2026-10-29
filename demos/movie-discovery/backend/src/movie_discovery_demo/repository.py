"""Milvus 3.0 provisioning, plain ANN search, and Search Aggregation."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from pymilvus import DataType, MilvusClient, SearchAggregation, TopHits

from movie_discovery_demo.config import (
    AGG_ROOT_SIZE,
    AGG_TOP_HITS_SIZE,
    ANNS_FIELD,
    APPROX_METRICS,
    BROWSE_TIEBREAK_FIELD,
    COLLECTION_NAME,
    FACET_FIELDS,
    INDEX_BUILD_POLL_SECONDS,
    INDEX_BUILD_TIMEOUT_SECONDS,
    INDEX_EF_CONSTRUCTION,
    INDEX_M,
    INDEX_METRIC,
    INDEX_NAME,
    INDEX_TYPE,
    MILVUS_EXPECTED_VERSION,
    MILVUS_TIMEOUT_SECONDS,
    MILVUS_URI,
    SEARCH_EF,
    VECTOR_DIMENSION,
)
from movie_discovery_demo.data import MovieRecord


class RepositoryContractError(RuntimeError):
    """Raised when the real Milvus state violates the fixed contract."""


class SearchContractError(ValueError):
    """Raised when a search cannot be executed honestly against the prepared collection."""


@dataclass(frozen=True)
class MovieHit:
    rank: int
    score: float
    movie: MovieRecord

    def public_dict(self) -> dict[str, object]:
        return {
            "rank": self.rank,
            "score": round(self.score, 6),
            "movie_id": self.movie.movie_id,
            "title": self.movie.title,
            "synopsis": self.movie.synopsis,
            "primary_genre": self.movie.primary_genre,
            "release_year": self.movie.release_year,
            "decade": self.movie.decade,
            "region": self.movie.region,
            "runtime_minutes": self.movie.runtime_minutes,
            "poster_path": self.movie.poster_path,
            "poster_palette": list(self.movie.poster_palette),
        }


@dataclass(frozen=True)
class GroupedBucket:
    key_value: str
    count: int
    hits: tuple[MovieHit, ...]

    def public_dict(self) -> dict[str, object]:
        return {
            "key": self.key_value,
            "count": self.count,
            "candidates": [hit.public_dict() for hit in self.hits],
        }


@dataclass(frozen=True)
class ProvisioningReport:
    server_version: str
    collection_name: str
    movie_count: int
    index_names: tuple[str, ...]

    def public_dict(self) -> dict[str, object]:
        return {
            "status": "prepared",
            "milvus_uri": MILVUS_URI,
            "server_version": self.server_version,
            "collection_name": self.collection_name,
            "movie_count": self.movie_count,
            "index_names": list(self.index_names),
            "vector_field": ANNS_FIELD,
            "vector_dimension": VECTOR_DIMENSION,
        }


class MovieRepository:
    """Own the fixed collection: schema, index, prepare, and two search paths."""

    def __init__(self, client_factory=None) -> None:
        self._client_factory = client_factory or default_client_factory

    def _client(self):
        return self._client_factory()

    def _checked_version(self, client) -> str:
        version = client.get_server_version()
        if version != MILVUS_EXPECTED_VERSION:
            raise RepositoryContractError(
                f"Expected Milvus {MILVUS_EXPECTED_VERSION}, got {version}"
            )
        return version

    def status(self) -> dict[str, object]:
        client = self._client()
        try:
            version = self._checked_version(client)
            exists = client.has_collection(COLLECTION_NAME)
            names = [str(name) for name in client.list_collections()]
            stats = client.get_collection_stats(COLLECTION_NAME) if exists else None
            indexes = client.list_indexes(COLLECTION_NAME) if exists else []
        finally:
            client.close()
        return {
            "milvus_uri": MILVUS_URI,
            "server_version": version,
            "collection_name": COLLECTION_NAME,
            "collection_exists": exists,
            "raw_collection_names": names,
            "row_count": int(stats.get("row_count", 0)) if isinstance(stats, Mapping) else None,
            "index_names": [str(index) for index in indexes],
        }

    def _schema(self, client) -> Any:
        schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("movie_id", DataType.VARCHAR, max_length=32, is_primary=True)
        schema.add_field("title", DataType.VARCHAR, max_length=128)
        schema.add_field("synopsis", DataType.VARCHAR, max_length=1024)
        schema.add_field("primary_genre", DataType.VARCHAR, max_length=32)
        schema.add_field("release_year", DataType.INT64)
        schema.add_field("decade", DataType.VARCHAR, max_length=16)
        schema.add_field("region", DataType.VARCHAR, max_length=64)
        schema.add_field("runtime_minutes", DataType.INT64)
        schema.add_field("poster_path", DataType.VARCHAR, max_length=128)
        schema.add_field(ANNS_FIELD, DataType.FLOAT_VECTOR, dim=VECTOR_DIMENSION)
        return schema

    def _index_params(self, client) -> Any:
        params = client.prepare_index_params()
        params.add_index(
            field_name=ANNS_FIELD,
            index_name=INDEX_NAME,
            index_type=INDEX_TYPE,
            metric_type=INDEX_METRIC,
            params={"M": INDEX_M, "efConstruction": INDEX_EF_CONSTRUCTION},
        )
        return params

    def _wait_for_finished_indexes(self, client) -> tuple[str, ...]:
        deadline = time.monotonic() + INDEX_BUILD_TIMEOUT_SECONDS
        while True:
            raw_names = client.list_indexes(COLLECTION_NAME)
            index_names = tuple(sorted(str(name) for name in raw_names))
            if index_names != (INDEX_NAME,):
                raise RepositoryContractError(
                    "Milvus index names differ: "
                    f"expected={[INDEX_NAME]}, actual={list(index_names)}"
                )
            descriptions = [client.describe_index(COLLECTION_NAME, INDEX_NAME)]
            states = {str(item.get("index_name")): str(item.get("state")) for item in descriptions}
            if all(str(item.get("state")) == "Finished" for item in descriptions):
                return index_names
            if time.monotonic() >= deadline:
                raise RepositoryContractError(
                    f"Milvus index did not finish within {INDEX_BUILD_TIMEOUT_SECONDS}s: {states}"
                )
            time.sleep(INDEX_BUILD_POLL_SECONDS)

    def prepare(
        self,
        movies: Sequence[MovieRecord],
        synopsis_vectors: Sequence[Sequence[float]],
    ) -> ProvisioningReport:
        if len(synopsis_vectors) != len(movies):
            raise RepositoryContractError("Synopsis vector count does not match movie count")

        client = self._client()
        try:
            version = self._checked_version(client)
            if client.has_collection(COLLECTION_NAME):
                raise RepositoryContractError("Collection already exists; refuse to replace it")
            client.create_collection(
                collection_name=COLLECTION_NAME,
                schema=self._schema(client),
                index_params=self._index_params(client),
                consistency_level="Strong",
            )

            rows: list[dict[str, Any]] = []
            for movie, vector in zip(movies, synopsis_vectors, strict=True):
                rows.append(
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
                        ANNS_FIELD: list(vector),
                    }
                )
            client.insert(collection_name=COLLECTION_NAME, data=rows)
            client.flush(COLLECTION_NAME)
            client.load_collection(COLLECTION_NAME)

            stats = client.get_collection_stats(COLLECTION_NAME)
            row_count = int(stats.get("row_count", 0)) if isinstance(stats, Mapping) else 0
            if row_count != len(movies):
                raise RepositoryContractError(f"Unexpected row count: {row_count} != {len(movies)}")
            index_names = self._wait_for_finished_indexes(client)
        finally:
            client.close()

        return ProvisioningReport(
            server_version=version,
            collection_name=COLLECTION_NAME,
            movie_count=len(movies),
            index_names=tuple(index_names),
        )

    def _hydrate(self, movie_id: object, movies_by_id: Mapping[str, MovieRecord]) -> MovieRecord:
        if not isinstance(movie_id, str):
            raise RepositoryContractError("Search hit is missing the movie_id primary key")
        movie = movies_by_id.get(movie_id)
        if movie is None:
            raise RepositoryContractError(f"Unknown movie_id in search hit: {movie_id}")
        return movie

    def list_search(
        self,
        *,
        query_vector: Sequence[float],
        movies_by_id: Mapping[str, MovieRecord],
        limit: int,
        filter: str | None = None,
    ) -> tuple[MovieHit, ...]:
        client = self._client()
        try:
            self._checked_version(client)
            if not client.has_collection(COLLECTION_NAME):
                raise RepositoryContractError("Collection is not prepared")
            kwargs: dict[str, Any] = {
                "collection_name": COLLECTION_NAME,
                "data": [list(query_vector)],
                "anns_field": ANNS_FIELD,
                "search_params": {"metric_type": "COSINE", "params": {"ef": SEARCH_EF}},
                "limit": limit,
                "output_fields": ["movie_id"],
            }
            if filter is not None:
                kwargs["filter"] = filter
            raw = client.search(**kwargs)
        finally:
            client.close()
        if len(raw) != 1:
            raise RepositoryContractError("Milvus search returned an unexpected batch shape")
        hits: list[MovieHit] = []
        for rank, hit in enumerate(raw[0], start=1):
            movie_id = hit.get("movie_id") or hit.get("id")
            movie = self._hydrate(movie_id, movies_by_id)
            hits.append(MovieHit(rank=rank, score=float(hit["distance"]), movie=movie))
        return tuple(hits)

    def _facet_field(self, facet: str) -> str:
        field = FACET_FIELDS.get(facet)
        if field is None:
            raise SearchContractError(f"Unknown facet: {facet}")
        return field[0]

    def aggregated_search(
        self,
        *,
        query_vector: Sequence[float],
        movies_by_id: Mapping[str, MovieRecord],
        facet: str,
        filter: str | None = None,
        top_hits_size: int = AGG_TOP_HITS_SIZE,
        root_size: int = AGG_ROOT_SIZE,
    ) -> tuple[GroupedBucket, ...]:
        facet_field = self._facet_field(facet)
        aggregation = SearchAggregation(
            fields=[facet_field],
            size=root_size,
            top_hits=TopHits(size=top_hits_size, sort=[{"_score": "desc"}]),
        )
        client = self._client()
        try:
            self._checked_version(client)
            if not client.has_collection(COLLECTION_NAME):
                raise RepositoryContractError("Collection is not prepared")
            kwargs: dict[str, Any] = {
                "collection_name": COLLECTION_NAME,
                "data": [list(query_vector)],
                "anns_field": ANNS_FIELD,
                "search_params": {"metric_type": "COSINE", "params": {"ef": SEARCH_EF}},
                "limit": 1,  # ignored; bucket count comes from SearchAggregation.size
                "output_fields": ["movie_id"],
                "search_aggregation": aggregation,
            }
            if filter is not None:
                kwargs["filter"] = filter
            raw = client.search(**kwargs)
        finally:
            client.close()
        if len(raw) != 1:
            raise RepositoryContractError("Milvus search returned an unexpected batch shape")
        buckets = getattr(raw, "agg_buckets", None)
        if buckets is None:
            raise RepositoryContractError("Milvus search did not return agg_buckets")
        if not isinstance(buckets, list) or len(buckets) != 1:
            raise RepositoryContractError("Unexpected agg_buckets shape")

        grouped: list[GroupedBucket] = []
        for bucket in buckets[0]:
            key_value = self._bucket_key_value(bucket, facet_field)
            hits: list[MovieHit] = []
            for rank, hit in enumerate(bucket.hits, start=1):
                movie = self._hydrate(hit.pk, movies_by_id)
                hits.append(MovieHit(rank=rank, score=float(hit.score), movie=movie))
            grouped.append(
                GroupedBucket(
                    key_value=key_value,
                    count=int(bucket.count),
                    hits=tuple(hits),
                )
            )
        return tuple(grouped)

    @staticmethod
    def _bucket_key_value(bucket, facet_field: str) -> str:
        key = bucket.key
        if not isinstance(key, list) or not key:
            raise RepositoryContractError("Aggregation bucket is missing its key")
        first = key[0]
        value = first.get("value") if isinstance(first, Mapping) else None
        if not isinstance(value, str) or not value:
            raise RepositoryContractError("Aggregation bucket key value is not a string")
        return value

    # ------------------------------------------------------------------
    # query + sort : server-side ORDER BY over the whole catalog
    # ------------------------------------------------------------------
    def browse_sorted(
        self,
        *,
        movies_by_id: Mapping[str, MovieRecord],
        sort_field: str,
        direction: str,
        limit: int,
        offset: int,
        filter: str | None = None,
    ) -> tuple[tuple[MovieRecord, ...], int]:
        """Return one deterministic page plus the exact total match count."""
        if direction not in ("asc", "desc"):
            raise SearchContractError(f"Unsupported sort direction: {direction}")
        if offset < 0:
            raise SearchContractError("offset must not be negative")

        # A duplicate sort value would otherwise make offset paging ambiguous,
        # so a unique secondary key pins the total order.
        order_by = [
            {"field": sort_field, "order": direction},
            {"field": BROWSE_TIEBREAK_FIELD, "order": "asc"},
        ]
        output_fields = [
            "movie_id",
            "title",
            "primary_genre",
            "release_year",
            "decade",
            "region",
            "runtime_minutes",
        ]
        client = self._client()
        try:
            self._checked_version(client)
            if not client.has_collection(COLLECTION_NAME):
                raise RepositoryContractError("Collection is not prepared")
            kwargs: dict[str, Any] = {
                "collection_name": COLLECTION_NAME,
                "output_fields": output_fields,
                "order_by": order_by,
                "limit": limit,
                "offset": offset,
            }
            if filter is not None:
                kwargs["filter"] = filter
            rows = client.query(**kwargs)
            # count(*) must not be combined with limit/offset.
            count_kwargs: dict[str, Any] = {
                "collection_name": COLLECTION_NAME,
                "output_fields": ["count(*)"],
            }
            if filter is not None:
                count_kwargs["filter"] = filter
            counted = client.query(**count_kwargs)
        finally:
            client.close()

        movies: list[MovieRecord] = []
        for row in rows:
            movie_id = row.get("movie_id")
            movies.append(self._hydrate(movie_id, movies_by_id))

        total = 0
        if counted:
            first = counted[0]
            raw_total = first.get("count(*)") if isinstance(first, Mapping) else None
            if isinstance(raw_total, int):
                total = raw_total
        return tuple(movies), total

    # ------------------------------------------------------------------
    # query + group : GROUP BY with exact aggregate functions
    # ------------------------------------------------------------------
    def query_aggregate(
        self,
        *,
        facet_field: str,
        metric_fields: Sequence[str],
        filter: str | None = None,
    ) -> tuple[dict[str, object], ...]:
        """Return one row per group with exact aggregates over the full catalog.

        `group_by_fields` (plural) is required: the singular `group_by_field`
        argument accepts the request but drops the aggregate expressions.
        """
        if not metric_fields:
            raise SearchContractError("At least one aggregate metric is required")
        output_fields = [facet_field, *metric_fields]
        client = self._client()
        try:
            self._checked_version(client)
            if not client.has_collection(COLLECTION_NAME):
                raise RepositoryContractError("Collection is not prepared")
            kwargs: dict[str, Any] = {
                "collection_name": COLLECTION_NAME,
                "output_fields": output_fields,
                "group_by_fields": [facet_field],
            }
            if filter is not None:
                kwargs["filter"] = filter
            rows = client.query(**kwargs)
        finally:
            client.close()

        groups: list[dict[str, object]] = []
        for row in rows:
            key = row.get(facet_field)
            if not isinstance(key, str) or not key:
                raise RepositoryContractError("Aggregated row is missing its group key")
            values: dict[str, object] = {}
            for metric in metric_fields:
                values[metric] = row.get(metric)
            groups.append({"key": key, "values": values})
        return tuple(groups)

    # ------------------------------------------------------------------
    # search + sort : re-order the retained top-K only
    # ------------------------------------------------------------------
    def ranked_search(
        self,
        *,
        query_vector: Sequence[float],
        movies_by_id: Mapping[str, MovieRecord],
        limit: int,
        sort_field: str | None = None,
        direction: str = "desc",
        filter: str | None = None,
    ) -> tuple[MovieHit, ...]:
        """Plain ANN search, optionally re-ordered by one scalar field.

        Sort/distance is intentionally returned untouched so the page can show
        that the score sequence stops being monotonic once a scalar sort is
        applied -- the candidate set itself never changes.
        """
        if sort_field is not None and direction not in ("asc", "desc"):
            raise SearchContractError(f"Unsupported sort direction: {direction}")
        client = self._client()
        try:
            self._checked_version(client)
            if not client.has_collection(COLLECTION_NAME):
                raise RepositoryContractError("Collection is not prepared")
            kwargs: dict[str, Any] = {
                "collection_name": COLLECTION_NAME,
                "data": [list(query_vector)],
                "anns_field": ANNS_FIELD,
                "search_params": {"metric_type": "COSINE", "params": {"ef": SEARCH_EF}},
                "limit": limit,
                "output_fields": ["movie_id"],
            }
            if sort_field is not None:
                # The search path only reads `order_by_fields`; passing
                # `order_by` here is silently ignored by pymilvus.
                kwargs["order_by_fields"] = [{"field": sort_field, "order": direction}]
            if filter is not None:
                kwargs["filter"] = filter
            raw = client.search(**kwargs)
        finally:
            client.close()
        if len(raw) != 1:
            raise RepositoryContractError("Milvus search returned an unexpected batch shape")
        hits: list[MovieHit] = []
        for rank, hit in enumerate(raw[0], start=1):
            movie_id = hit.get("movie_id") or hit.get("id")
            movie = self._hydrate(movie_id, movies_by_id)
            hits.append(MovieHit(rank=rank, score=float(hit["distance"]), movie=movie))
        return tuple(hits)

    def aggregated_search_with_metrics(
        self,
        *,
        query_vector: Sequence[float],
        movies_by_id: Mapping[str, MovieRecord],
        facet: str,
        filter: str | None = None,
        root_size: int = AGG_ROOT_SIZE,
        top_hits_size: int = AGG_TOP_HITS_SIZE,
    ) -> tuple[dict[str, object], ...]:
        """Facet buckets that also carry per-bucket metrics and best hits.

        The bucket `count` is the number of hits retained for that bucket, which
        is bounded by `top_hits_size`; it is NOT a count over the whole catalog.
        """
        facet_field = self._facet_field(facet)
        # Already in pymilvus' {alias: {aggregate_op: field_name}} shape.
        metric_spec: dict[str, dict[str, str]] = {
            alias: dict(spec) for alias, spec in APPROX_METRICS.items()
        }

        aggregation = SearchAggregation(
            fields=[facet_field],
            size=root_size,
            metrics=metric_spec,
            top_hits=TopHits(size=top_hits_size, sort=[{"_score": "desc"}]),
        )
        client = self._client()
        try:
            self._checked_version(client)
            if not client.has_collection(COLLECTION_NAME):
                raise RepositoryContractError("Collection is not prepared")
            kwargs: dict[str, Any] = {
                "collection_name": COLLECTION_NAME,
                "data": [list(query_vector)],
                "anns_field": ANNS_FIELD,
                "search_params": {"metric_type": "COSINE", "params": {"ef": SEARCH_EF}},
                "limit": 1,  # ignored; bucket count comes from SearchAggregation.size
                "output_fields": ["movie_id"],
                "search_aggregation": aggregation,
            }
            if filter is not None:
                kwargs["filter"] = filter
            raw = client.search(**kwargs)
        finally:
            client.close()
        if len(raw) != 1:
            raise RepositoryContractError("Milvus search returned an unexpected batch shape")
        buckets = getattr(raw, "agg_buckets", None)
        if buckets is None:
            raise RepositoryContractError("Milvus search did not return agg_buckets")
        if not isinstance(buckets, list) or len(buckets) != 1:
            raise RepositoryContractError("Unexpected agg_buckets shape")

        out: list[dict[str, object]] = []
        for bucket in buckets[0]:
            key_value = self._bucket_key_value(bucket, facet_field)
            hits: list[MovieHit] = []
            for rank, hit in enumerate(bucket.hits, start=1):
                movie = self._hydrate(hit.pk, movies_by_id)
                hits.append(MovieHit(rank=rank, score=float(hit.score), movie=movie))
            raw_metrics = getattr(bucket, "metrics", None) or {}
            clean_metrics: dict[str, object] = {}
            for alias in APPROX_METRICS:
                value = raw_metrics.get(alias)
                if isinstance(value, float):
                    clean_metrics[alias] = round(value, 2)
                elif value is not None:
                    clean_metrics[alias] = value
            out.append(
                {
                    "key": key_value,
                    "count": int(bucket.count),
                    "metrics": clean_metrics,
                    "candidates": [hit.public_dict() for hit in hits],
                }
            )
        return tuple(out)

    def cleanup(self) -> dict[str, object]:
        client = self._client()
        try:
            version = self._checked_version(client)
            existed = client.has_collection(COLLECTION_NAME)
            indexes = (
                [str(index) for index in client.list_indexes(COLLECTION_NAME)] if existed else []
            )
            if existed:
                client.drop_collection(COLLECTION_NAME)
            absent = not client.has_collection(COLLECTION_NAME)
        finally:
            client.close()
        return {
            "status": "clean",
            "server_version": version,
            "collection_name": COLLECTION_NAME,
            "dropped": existed,
            "indexes_deleted_with_collection": list(indexes),
            "collection_absent_after": absent,
        }


def default_client_factory():
    return MilvusClient(uri=MILVUS_URI, timeout=MILVUS_TIMEOUT_SECONDS)
