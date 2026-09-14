"""Fixed CPU and filesystem boundaries for the ColSmol EmbeddingList demo."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DEMO_ID = "embedding-list-max-sim"
CAPABILITY = "EmbeddingList + MAX_SIM_COSINE"

# Deployable against a shared Milvus by setting MILVUS_URI; defaults to the
# local standalone endpoint for development.
MILVUS_URI = os.environ.get("MILVUS_URI", "http://127.0.0.1:49530")
MILVUS_EXPECTED_VERSION = "3.0.0"
MILVUS_TIMEOUT_SECONDS = 30.0
COLLECTION_NAME = "milvus3_demos_nasa_seh_colsmol_pages"
INDEX_NAME = "milvus3_demos_nasa_seh_patch_embeddings"
ANNS_FIELD = "patches[patch_embedding]"
METRIC_TYPE = "MAX_SIM_COSINE"
EXECUTION_PATH = "milvus_embedding_list_max_sim_cosine"

MODEL_ID = "vidore/colSmol-256M"
MODEL_ADAPTER_REVISION = "ff628001884d6bfb066658e86a32b5f233906bf5"
MODEL_BASE_ID = "vidore/ColSmolVLM-Instruct-256M-base"
MODEL_BASE_REVISION = "99ca96f1f6b95b3a69e6abef74a2416cb738fed0"
MODEL_GIT_HASH = "8b4f75e476bddc6c34a02722761bd3ada6ac0d3d"
MODEL_DTYPE = "float32"
VECTOR_DIMENSION = 128
MAX_PATCHES_PER_PAGE = 4096
PAGE_COUNT = 40
SCORE_ABSOLUTE_TOLERANCE = 0.005

RUNTIME_RELATIVE_ROOT = Path("artifacts/runtime/embedding-list-max-sim")
DATASET_RELATIVE_ROOT = Path(
    "demos/embedding-list-max-sim/data/nasa-systems-engineering-handbook-rev2"
)
SUPPORTED_DEVICE = "cpu"


class ConfigurationError(ValueError):
    """Raised when a runtime setting would cross the demo boundary."""


def repository_root() -> Path:
    """Return the repository root from the independently packaged backend."""

    return Path(__file__).resolve().parents[5]


def default_runtime_root() -> Path:
    """Return the only default root for generated demo artifacts."""

    return (repository_root() / RUNTIME_RELATIVE_ROOT).resolve()


def default_dataset_root() -> Path:
    """Return the tracked, read-only NASA handbook dataset root."""

    return (repository_root() / DATASET_RELATIVE_ROOT).resolve()


def default_hf_hub_cache() -> Path:
    """Resolve the Hugging Face hub cache without initiating a download."""

    configured = os.environ.get("HF_HUB_CACHE")
    if configured:
        return Path(configured).expanduser().resolve()
    hf_home = Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser()
    return (hf_home / "hub").resolve()


@dataclass(frozen=True)
class RuntimeConfig:
    """Resolved local runtime inputs for one backend process."""

    runtime_root: Path
    hf_hub_cache: Path
    dataset_root: Path = default_dataset_root()
    device: str = SUPPORTED_DEVICE

    def __post_init__(self) -> None:
        if self.device != SUPPORTED_DEVICE:
            raise ConfigurationError("RuntimeConfig device must be cpu")

    @classmethod
    def from_environment(cls) -> RuntimeConfig:
        """Load strict runtime, dataset, model-cache, and CPU-only settings."""

        allowed_root = default_runtime_root()
        configured_root = Path(
            os.environ.get("EMBEDDING_LIST_RUNTIME_DIR", str(allowed_root))
        ).resolve()
        if configured_root != allowed_root and not configured_root.is_relative_to(allowed_root):
            raise ConfigurationError(
                "EMBEDDING_LIST_RUNTIME_DIR must stay under "
                "artifacts/runtime/embedding-list-max-sim"
            )
        dataset_root = Path(
            os.environ.get("EMBEDDING_LIST_DATASET_DIR", str(default_dataset_root()))
        ).resolve()
        if dataset_root != default_dataset_root():
            raise ConfigurationError(
                "EMBEDDING_LIST_DATASET_DIR must identify the tracked NASA handbook dataset"
            )
        device = os.environ.get("COLSMOL_DEVICE", SUPPORTED_DEVICE)
        if device != SUPPORTED_DEVICE:
            raise ConfigurationError("COLSMOL_DEVICE must be cpu")
        return cls(
            runtime_root=configured_root,
            hf_hub_cache=default_hf_hub_cache(),
            dataset_root=dataset_root,
            device=device,
        )

    @property
    def embedding_cache_root(self) -> Path:
        return self.runtime_root / "embedding-cache"
