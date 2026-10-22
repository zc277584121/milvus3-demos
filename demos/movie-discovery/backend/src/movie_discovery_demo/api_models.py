"""Strict HTTP models for the Movie Discovery backend."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

DEMO_ID_LITERAL = Literal["movie-discovery"]
FACET_LITERAL = Literal["genre", "decade", "region"]
VIEW_LITERAL = Literal["grouped", "list"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class LiveHealthResponse(StrictModel):
    status: Literal["healthy"] = "healthy"
    demo_id: DEMO_ID_LITERAL = "movie-discovery"


class ReadyHealthResponse(StrictModel):
    status: Literal["healthy", "unhealthy"]
    model_cache_available: bool
    device_type: Literal["cpu"] = "cpu"
    inference_dtype: Literal["float32"] = "float32"
    cpu_only: Literal[True] = True
    collection_exists: bool
    server_version: str | None
    vector_dimension: int | None


class QueryPresetModel(StrictModel):
    id: str
    label: str
    text: str
    description: str


class ModelModel(StrictModel):
    model_id: str
    revision: str
    vector_dimension: int
    dense_output_name: str
    device: Literal["cpu"]
    loaded: bool
    cache_available: bool


class DatasetModel(StrictModel):
    dataset_id: str
    dataset_version: int
    movie_count: int
    movies_sha256: str
    presets_sha256: str


class MilvusModel(StrictModel):
    milvus_uri: str
    server_version: str
    collection_name: str
    collection_exists: bool
    raw_collection_names: list[str]
    row_count: int | None
    index_names: list[str]


class FacetModel(StrictModel):
    id: FACET_LITERAL
    field: str
    label: str


class StatusResponse(StrictModel):
    status: Literal["ready", "not_prepared"]
    movie_count: int
    model: ModelModel
    dataset: DatasetModel
    milvus: MilvusModel
    facets: list[FacetModel]
    query_presets: list[QueryPresetModel]
    poster_placeholders: bool


class EmbeddingModel(StrictModel):
    vector_count: int
    vector_dimension: int
    embedding_seconds: float


class PreparedMilvusModel(StrictModel):
    status: Literal["prepared"]
    milvus_uri: str
    server_version: str
    collection_name: str
    movie_count: int
    index_names: list[str]
    vector_field: str
    vector_dimension: int


class PrepareResponse(StrictModel):
    status: Literal["prepared"]
    embedding: EmbeddingModel
    milvus: PreparedMilvusModel


class SearchRequest(StrictModel):
    query: str = Field(min_length=1, max_length=2000)
    view: VIEW_LITERAL = "grouped"
    facet: FACET_LITERAL = "genre"
    limit: int = Field(default=12, ge=1, le=20)
    max_runtime: int | None = Field(default=None, ge=1, le=120)


class MovieResultModel(StrictModel):
    rank: int
    score: float
    movie_id: str
    title: str
    synopsis: str
    primary_genre: str
    release_year: int
    decade: str
    region: str
    runtime_minutes: int
    poster_path: str
    poster_palette: list[str]


class ListModel(StrictModel):
    results: list[MovieResultModel]


class GroupedBucketModel(StrictModel):
    key: str
    count: int
    candidates: list[MovieResultModel]


class GroupedModel(StrictModel):
    buckets: list[GroupedBucketModel]


class SearchResponse(StrictModel):
    status: Literal["passed"]
    query: str
    view: VIEW_LITERAL
    facet: FACET_LITERAL
    facet_field: str
    filter: str | None
    limit: int
    latency_ms: float
    score_semantics: str
    list: ListModel
    grouped: GroupedModel
    retained_candidate_count: int
    semantics: str


class CleanupMilvusModel(StrictModel):
    status: Literal["clean"]
    server_version: str
    collection_name: str
    dropped: bool
    indexes_deleted_with_collection: list[str]
    collection_absent_after: bool


class CleanupResponse(StrictModel):
    status: Literal["clean"]
    milvus: CleanupMilvusModel
    model_loaded: bool
