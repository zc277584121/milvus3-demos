"""Strict public API models for the EmbeddingList demo."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LiveHealthResponse(StrictModel):
    status: str = "healthy"
    demo_id: str = "embedding-list-max-sim"


class ReadyHealthResponse(StrictModel):
    status: str
    model_cache_available: bool
    device_type: str
    inference_dtype: str
    cpu_only: bool
    collection_exists: bool
    server_version: str | None


class SearchRequest(StrictModel):
    query: str = Field(min_length=1, max_length=512)


class StatusResponse(StrictModel):
    status: str
    implementation_status: str
    runtime_ready: bool
    execution_path: str
    collection_name: str
    index_name: str
    anns_field: str
    metric_type: str
    dataset_exists: bool
    embedding_cache_exists: bool
    dataset: dict[str, Any]
    query_presets: list[dict[str, str]]
    last_prepare: dict[str, Any] | None
    prepare_timings: dict[str, dict[str, float] | None]
    model: dict[str, Any]
    milvus: dict[str, Any]


class ManualResponse(StrictModel):
    generated: bool
    schema_version: int
    dataset_id: str
    title: str
    revision: str
    document_identifier: str
    ntrs_id: int
    ntrs_record_url: str
    official_pdf_url: str
    distribution: str
    rights_determination: str
    contains_third_party_material: bool
    attribution: str
    authors: list[dict[str, str]]
    language: str
    render_statement: str
    endorsement_statement: str
    generator: str
    generator_version: str
    renderer: str
    renderer_version: str
    source_spec_sha256: str
    source_pdf_page_count: int
    source_pdf_bytes: int
    pdf_page_index_base: int
    page_count: int
    pdf_filename: str
    pdf_sha256: str
    render_dpi: int
    queries_filename: str
    queries_sha256: str
    manifest_sha256: str
    pages: list[dict[str, Any]]
    query_presets: list[dict[str, str]]


class PrepareResponse(StrictModel):
    status: str
    implementation_status: str
    runtime_ready: bool
    manual_generated: bool
    manual: dict[str, Any]
    query_presets: list[dict[str, str]]
    cache: dict[str, Any]
    model: dict[str, Any]
    page_inference: dict[str, Any]
    milvus: dict[str, Any]
    timings: dict[str, float]
    prepare_timings: dict[str, dict[str, float] | None]
    latency_ms: float


class SearchResponse(StrictModel):
    status: str
    execution_path: str
    query: str
    collection_name: str
    index_name: str
    anns_field: str
    metric_type: str
    dataset: dict[str, Any]
    query_embedding: dict[str, Any]
    page_vector_counts: list[int]
    milvus_page_order: list[str]
    local_page_order: list[str]
    ranking_equal: bool
    score_absolute_tolerance: float
    max_score_delta: float
    score_within_tolerance: bool
    results: list[dict[str, Any]]
    local_explanations: list[dict[str, Any]]
    model: dict[str, Any]
    timings: dict[str, float]
    latency_ms: float


class CleanupResponse(StrictModel):
    status: str
    milvus: dict[str, Any]
    embedding_cache_removed: bool
    dataset_source_preserved: bool
    runtime_root_exists: bool
    model_loaded: bool
