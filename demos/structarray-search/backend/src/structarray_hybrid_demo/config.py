"""Fixed CPU, model, Milvus, and filesystem boundaries for the hybrid demo."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DEMO_ID = "structarray-search"
CAPABILITY = "Parent + child semantic hybrid search"

# Deployable against a shared Milvus by setting MILVUS_URI; defaults to the
# local standalone endpoint for development.
MILVUS_URI = os.environ.get("MILVUS_URI", "http://127.0.0.1:49530")
MILVUS_EXPECTED_VERSION = "3.0.0"
MILVUS_TIMEOUT_SECONDS = 30.0

COLLECTION_NAME = "milvus3_demos_structarray_hybrid_covla"
CHILD_ARRAY_FIELD = "observations"
CHILD_TEXT_FIELD = "description"
CHILD_VECTOR_FIELD = "description_vector"
CHILD_ANNS_FIELD = f"{CHILD_ARRAY_FIELD}[{CHILD_VECTOR_FIELD}]"

PARENT_VECTOR_FIELD = "summary_vector"
PARENT_ANNS_FIELD = PARENT_VECTOR_FIELD

GROUP_BY_FIELD = "video_id"

EXECUTION_PATH = "milvus_parent_child_semantic_hybrid"

METRIC_TYPE = "COSINE"
VECTOR_DIMENSION = 1024
INDEX_METRIC = "COSINE"
INDEX_TYPE = "HNSW"
INDEX_M = 16
INDEX_EF_CONSTRUCTION = 128
SEARCH_EF = 128
INDEX_BUILD_POLL_SECONDS = 2.0
INDEX_BUILD_TIMEOUT_SECONDS = 120.0

PARENT_INDEX_NAME = "milvus3_demos_structarray_hybrid_summary_vector"
CHILD_INDEX_NAME = "milvus3_demos_structarray_hybrid_description_vector"

OBSERVATIONS_MAX_CAPACITY = 128
DEFAULT_TOP_K = 8
MAX_TOP_K = 20
DEFAULT_COLLAPSE_STRATEGY = "topk_sum"
DEFAULT_COLLAPSE_TOPK = 3
DEFAULT_PARENT_WEIGHT = 0.5
WEIGHT_MIN = 0.0
WEIGHT_MAX = 1.0

MODEL_ID = "gpahal/bge-m3-onnx-int8"
MODEL_REVISION = "2b34e84df040034d4b9eabb62383a87c18955822"
MODEL_DIMENSION = VECTOR_DIMENSION
MODEL_BATCH_SIZE = 32

SAMPLE_SIZE = 30
SAMPLE_OFFSET = 0
EXPECTED_VIDEO_COUNT = 30
EXPECTED_MIN_OBSERVATIONS = 500
EXPECTED_MAX_OBSERVATIONS = 900

RUNTIME_RELATIVE_ROOT = Path("artifacts/runtime/structarray-search")
DEFAULT_COVLA_DATA_ROOT = Path("~/.cache/covla-dataset").expanduser()
SAMPLE_RELATIVE_PATH = Path("video_ready/video_data_100_samples.json")
PREFIX_RELATIVE_PATH = Path("video_ready/video_data_10_samples.json")
RAW_FRAME_DIRECTORY = Path("yolo_result/behavior_frames_output/frames")
ANNOTATED_FRAME_DIRECTORY = Path("yolo_result/draw_boxes/frame_wise")
EXPECTED_SAMPLE_SHA256 = "c7accf8cd63c77ba860242aa6415b9a3c44089731312818a50e6d89cde673b0a"
EXPECTED_PREFIX_SHA256 = "4dbbc43b9ccb97602f28a93acc10428bf51ae62a2b6e8503abe200d54d0524f1"

# Each preset pairs a scene term (matched against the environment segment of
# video_summary) with an exact object_type AND an exact color, both matched
# against the SAME observation. The parent summary only carries object counts
# ("4 truck"), never colors, so a "color + object + scene" query is precisely
# where the child path owns the discriminating signal and the fused column can
# beat BOTH single paths on ranking (NDCG) while reaching full recall. The four
# presets below are chosen so at least one single path misses and the fused
# column both fills the gap and ranks the hits ahead of either single path.
QUERY_PRESETS = (
    (
        "white-truck-intersection",
        "a white truck waiting at an intersection",
        ("intersection",),
        ("truck",),
        ("white",),
    ),
    (
        "white-van-bridge",
        "a white van on a bridge",
        ("bridge",),
        ("van",),
        ("white",),
    ),
    (
        "white-truck-ramp",
        "a white truck on a highway ramp",
        ("ramp",),
        ("truck",),
        ("white",),
    ),
    (
        "black-suv-residential",
        "a black suv in a residential neighborhood",
        ("local_residential",),
        ("suv",),
        ("black",),
    ),
)


class ConfigurationError(ValueError):
    """Raised when a runtime setting would cross the demo boundary."""


def repository_root() -> Path:
    """Return the repository root from the independently packaged backend."""
    return Path(__file__).resolve().parents[5]


def default_runtime_root() -> Path:
    """Return the only default root for generated demo artifacts."""
    return (repository_root() / RUNTIME_RELATIVE_ROOT).resolve()


def default_hf_hub_cache() -> Path:
    """Resolve the Hugging Face hub cache without initiating a download."""
    configured = os.environ.get("HF_HUB_CACHE")
    if configured:
        return Path(configured).expanduser().resolve()
    hf_home = Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser()
    return (hf_home / "hub").resolve()


def default_covla_data_root() -> Path:
    """Return the read-only CoVLA source root."""
    configured = os.environ.get("COVLA_DATA_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    return DEFAULT_COVLA_DATA_ROOT.resolve()


@dataclass(frozen=True)
class RuntimeConfig:
    """Resolved local runtime inputs for one backend process."""

    runtime_root: Path
    hf_hub_cache: Path
    data_root: Path
    sample_path: Path
    prefix_path: Path

    def __post_init__(self) -> None:
        if self.sample_path.is_relative_to(self.data_root) is False:
            raise ConfigurationError("Sample path must stay under the CoVLA data root")

    @classmethod
    def from_environment(cls) -> RuntimeConfig:
        allowed_root = default_runtime_root()
        configured_root = Path(
            os.environ.get("STRUCTARRAY_RUNTIME_DIR", str(allowed_root))
        ).resolve()
        if configured_root != allowed_root and not configured_root.is_relative_to(allowed_root):
            raise ConfigurationError(
                "STRUCTARRAY_RUNTIME_DIR must stay under artifacts/runtime/structarray-search"
            )
        data_root = default_covla_data_root()
        sample_path = (data_root / SAMPLE_RELATIVE_PATH).resolve()
        prefix_path = (data_root / PREFIX_RELATIVE_PATH).resolve()
        return cls(
            runtime_root=configured_root,
            hf_hub_cache=default_hf_hub_cache(),
            data_root=data_root,
            sample_path=sample_path,
            prefix_path=prefix_path,
        )

    @property
    def embedding_cache_root(self) -> Path:
        return self.runtime_root / "embedding-cache"
