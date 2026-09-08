"""Public request and response models for the Function Chain demo."""

from typing import Literal

from pydantic import BaseModel, Field


class LiveHealthResponse(BaseModel):
    status: Literal["healthy"] = "healthy"
    demo_id: Literal["function-chain-rerank"] = "function-chain-rerank"


class ReadyHealthResponse(BaseModel):
    status: Literal["healthy", "unhealthy"]
    milvus_version: str | None
    collection_name: str
    model_sha256: str | None


class DatasetIdentityResponse(BaseModel):
    dataset_id: Literal["synthetic-commerce-catalog"]
    revision: Literal["synthetic-commerce-catalog-r1"]
    manifest_sha256: str
    dataset_name: Literal["Synthetic Commerce Catalog"]
    publisher_and_data_credit: Literal["milvus3-demos project (no external publisher)"]
    license: Literal["CC0-1.0"]
    license_url: str
    license_conflict_record: Literal["none"]
    publication_review_required: Literal[False]
    source_type: Literal["synthetic_catalog_with_simulated_operations"]
    synthetic: Literal[True]
    real_product_metadata: Literal[False]
    real_product_photos: Literal[False]
    real_transaction_data: Literal[False]
    contains_simulated_operational_signals: Literal[True]
    simulated_signal_label: Literal["deterministic_simulated_operational_signal"]
    as_of_date: str
    product_count: int
    product_type_count: int
    image_count: int
    embedding: dict[str, object]
    semantic_score_preprocessing: dict[str, object]
    relevance_ground_truth: dict[str, object]
    field_provenance: dict[str, str]


class DemoStatusResponse(BaseModel):
    demo_id: str
    capability: str
    implementation_status: str
    expected_milvus_version: str
    execution_path: str
    model_version: str
    feature_order: list[str]
    dataset: DatasetIdentityResponse


class QueryOptionResponse(BaseModel):
    id: str
    query_text: str
    story: str
    split: Literal["train", "validation"]
    dataset: DatasetIdentityResponse


class SearchRequest(BaseModel):
    query_text: str = Field(min_length=3, max_length=240)
    query_id: str | None = Field(default=None, min_length=1, max_length=64)


class RankedProductResponse(BaseModel):
    id: int
    rank: int
    score: float
    item_id: str
    title: str | None
    product_type: str
    brand: str | None
    color: str | None
    material: str | None
    style: str | None
    node_name: str | None
    description: str | None
    bullet_points: list[str]
    main_image_id: str
    selected_image_id: str
    image_role: str
    source_object_path: str
    source_url: str
    display_price_usd: float
    rating_value: float
    clicks_30d: int
    sales_30d: int
    inventory_units: int
    inventory_capacity: int
    release_date: str
    release_epoch: int
    rating: float
    inventory: float
    return_rate: float
    freshness: float
    image_path: str
    image_mime: Literal["image/jpeg"]
    image_width: int
    image_height: int
    image_sha256: str
    operational_signal_provenance: Literal["deterministic_simulated_operational_signal"]
    field_provenance: dict[str, str]


class ChainStepResponse(BaseModel):
    name: str
    operation: str
    output: str
    description: str
    code: str


class FunctionChainExecutionResponse(BaseModel):
    stage: Literal["L0_RERANK"]
    operation: Literal["xgboost"]
    feature_order: list[str]
    parallel_inputs: Literal[False]
    intermediate_feature_orders: Literal[False]
    vector_order_score: Literal["semantic_score"]
    business_order_score: Literal["business_score"]
    chain_steps: list[ChainStepResponse]


class DataModelResponse(BaseModel):
    fields: list[dict[str, object]]
    chain_trace: dict[str, object]


class SearchResponse(BaseModel):
    query_id: str | None
    query_text: str
    execution_path: Literal["milvus_l0_xgboost_function_chain"]
    model_resource_name: str
    function_chain: FunctionChainExecutionResponse
    dataset: DatasetIdentityResponse
    data_model: DataModelResponse
    vector_order: list[RankedProductResponse]
    business_order: list[RankedProductResponse]
