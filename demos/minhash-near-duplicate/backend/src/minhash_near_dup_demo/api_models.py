"""Strict HTTP models for the MinHash near-duplicate backend."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

DEMO_ID_LITERAL = Literal["minhash-near-duplicate"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LiveHealthResponse(StrictModel):
    status: Literal["healthy"] = "healthy"
    demo_id: DEMO_ID_LITERAL = "minhash-near-duplicate"


class ReadyHealthResponse(StrictModel):
    status: Literal["healthy", "unhealthy"]
    collection_exists: bool
    server_version: str | None


class PresetModel(StrictModel):
    id: str
    label: str
    title: str
    text: str
    description: str


class StatusResponse(StrictModel):
    status: Literal["ready", "not_prepared"]
    implementation_status: Literal["implemented"]
    execution_path: str
    capability: str
    collection_name: str
    index_name: str
    text_field: str
    signature_field: str
    metric_type: str
    dataset: dict[str, object]
    query_presets: list[PresetModel]
    milvus: dict[str, object]
    last_prepare: dict[str, object] | None


class PrepareResponse(StrictModel):
    status: Literal["prepared"]
    collection_name: str
    index_name: str
    text_field: str
    signature_field: str
    metric_type: str
    server_version: str
    insert_result: dict[str, object]
    audit: dict[str, object]
    latency_ms: dict[str, float]


class SearchRequest(StrictModel):
    query: str = Field(min_length=1, max_length=2000)
    threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    limit: int = Field(default=5, ge=1, le=20)
    selected_doc_id: str | None = None


class SearchResultModel(StrictModel):
    doc_id: str
    title: str
    content: str
    estimated_jaccard: float
    raw_distance: float
    above_threshold: bool


class DiffSegmentModel(StrictModel):
    kind: Literal["equal", "removed", "added"]
    text: str


class DiffModel(StrictModel):
    document_id: str
    removed_count: int
    added_count: int
    segments: list[DiffSegmentModel]


class SearchResponse(StrictModel):
    status: Literal["passed"]
    execution_path: str
    query: str
    collection_name: str
    index_name: str
    text_field: str
    signature_field: str
    metric_type: str
    threshold: float
    limit: int
    above_threshold_count: int
    results: list[SearchResultModel]
    diffs: list[DiffModel]
    semantics: str
    latency_ms: float


class CleanupMilvusModel(StrictModel):
    status: Literal["clean"]
    collection_name: str
    dropped: bool
    target_absent_after: bool
    unknown_collections_unchanged: bool


class CleanupResponse(StrictModel):
    status: Literal["clean"]
    collection_name: str
    dropped: bool
    target_absent_after: bool
    unknown_collections_unchanged: bool
