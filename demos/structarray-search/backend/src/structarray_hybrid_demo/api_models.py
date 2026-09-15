"""Strict HTTP models for the StructArray hybrid backend."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

DEMO_ID_LITERAL = Literal["structarray-search"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class LiveHealthResponse(StrictModel):
    status: Literal["healthy"] = "healthy"
    demo_id: DEMO_ID_LITERAL = "structarray-search"


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
    text: str
    scene_terms: list[str] = []
    object_terms: list[str] = []
    color_terms: list[str] = []


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
    video_count: int
    observation_count: int
    evidence_frame_count: int
    manifest_sha256: str


class MilvusModel(StrictModel):
    milvus_uri: str
    server_version: str
    collection_name: str
    collection_exists: bool
    raw_collection_names: list[str]
    row_count: int | None
    index_names: list[str]


class StatusResponse(StrictModel):
    status: Literal["ready", "not_prepared"]
    sample_size: int
    model: ModelModel
    dataset: DatasetModel
    milvus: MilvusModel
    query_presets: list[QueryPresetModel]


class EmbeddingModel(StrictModel):
    parent_vectors: int
    child_vectors: int
    child_vector_dimension: int
    embedding_seconds: float


class PreparedSchemaModel(StrictModel):
    parent_fields: list[str]
    child_array: str
    child_vector_field: str
    child_text_field: str
    max_capacity: int


class PreparedMilvusModel(StrictModel):
    status: Literal["prepared"]
    milvus_uri: str
    server_version: str
    collection_name: str
    parent_video_count: int
    element_count: int
    index_names: list[str]
    schema_info: PreparedSchemaModel = Field(alias="schema")


class PrepareResponse(StrictModel):
    status: Literal["prepared"]
    embedding: EmbeddingModel
    milvus: PreparedMilvusModel


class SearchRequest(StrictModel):
    query: str = Field(min_length=1, max_length=2000)
    limit: int = Field(default=8, ge=1, le=20)
    parent_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    collapse_strategy: Literal["max", "sum", "avg", "topk_sum", "topk_avg"] = "topk_sum"


class ObservationModel(StrictModel):
    description: str
    object_type: str
    frame_id: int
    image_id: str
    bbox: list[int]
    vehicle_type: str
    color: str
    orientation: str
    lights_on: str
    v_ego: float
    a_ego: float
    clip_id: str
    raw_frame: str | None
    annotated_frame: str | None


class PreviewModel(StrictModel):
    frame_id: int
    annotated_frame: str | None
    raw_frame: str | None


class ParentResultModel(StrictModel):
    rank: int
    score: float
    video_id: str
    video_summary: str
    source_ordinal: int
    preview: PreviewModel | None
    scene_match: bool = False
    object_match: bool = False
    color_match: bool = False
    actual_scene: str = ""
    actual_object: str = ""
    actual_color: str = ""


class ChildResultModel(StrictModel):
    rank: int
    score: float
    video_id: str
    video_summary: str
    source_ordinal: int
    offset: int
    observation: ObservationModel
    scene_match: bool = False
    object_match: bool = False
    color_match: bool = False
    actual_scene: str = ""
    actual_object: str = ""
    actual_color: str = ""


class FusionResultModel(StrictModel):
    rank: int
    score: float
    video_id: str
    video_summary: str
    source_ordinal: int
    preview: PreviewModel | None
    child_observation: ObservationModel | None = None
    scene_match: bool = False
    object_match: bool = False
    color_match: bool = False
    actual_scene: str = ""
    actual_object: str = ""
    actual_color: str = ""


class ParentPathModel(StrictModel):
    anns_field: str
    label: str
    results: list[ParentResultModel]


class ChildPathModel(StrictModel):
    anns_field: str
    group_by_field: str
    label: str
    results: list[ChildResultModel]


class FusionPathModel(StrictModel):
    label: str
    ranker: str
    parent_anns_field: str
    child_anns_field: str
    results: list[FusionResultModel]


class PathsModel(StrictModel):
    parent: ParentPathModel
    child: ChildPathModel
    fusion: FusionPathModel


class GroundTruthModel(StrictModel):
    scene_terms: list[str]
    object_terms: list[str]
    color_terms: list[str] = []
    matched_video_ids: list[str]
    matched_count: int


class PathRecallModel(StrictModel):
    matched: int
    gt_count: int
    recall: float
    ndcg: float


class SearchResponse(StrictModel):
    status: Literal["passed"]
    query: str
    query_vector_dimension: int
    limit: int
    parent_weight: float
    child_weight: float
    collapse_strategy: str
    collapse_topk: int
    latency_ms: float
    score_semantics: str
    paths: PathsModel
    ground_truth: GroundTruthModel | None = None
    path_recall: dict[str, PathRecallModel] | None = None


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
