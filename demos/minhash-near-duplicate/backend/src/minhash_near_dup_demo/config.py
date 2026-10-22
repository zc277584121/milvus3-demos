"""Fixed MinHash + LSH boundaries for the near-duplicate detection demo."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DEMO_ID = "minhash-near-duplicate"
CAPABILITY = "MinHash Function + MINHASH_LSH"

# Deployable against a shared Milvus by setting MILVUS_URI; defaults to the
# local standalone endpoint for development.
MILVUS_URI = os.environ.get("MILVUS_URI", "http://127.0.0.1:49530")
MILVUS_EXPECTED_VERSION = "3.0.0"
MILVUS_TIMEOUT_SECONDS = 30.0

COLLECTION_NAME = "milvus3_demos_minhash_near_duplicate"
TEXT_FIELD = "content"
SIGNATURE_FIELD = "minhash_signature"
INDEX_NAME = "milvus3_demos_minhash_signature"
EXECUTION_PATH = "milvus_minhash_lsh_near_duplicate"

# MinHash function contract: output dim must equal 32 * num_hashes.
NUM_HASHES = 256
# Word-level shingle_size=1 is verified against the checked-in short texts:
# a two-word light edit keeps exact Jaccard near the 0.85 threshold (~0.85),
# while a 3-shingle drops the same edit to ~0.70 and would mislabel it as
# non-duplicate. The design requires light edits to land above the default
# threshold, so 1 is the correct choice for this corpus.
SHINGLE_SIZE = 1
HASH_FUNCTION = "xxhash"
TOKEN_LEVEL = "word"
SIGNATURE_DIMENSION = 32 * NUM_HASHES

# MINHASH_LSH index + MHJACCARD metric. mh_element_bit_width must be 32 for
# built-in function output, and with_raw_data stores raw signatures so Jaccard
# refinement can rank candidates.
INDEX_TYPE = "MINHASH_LSH"
METRIC_TYPE = "MHJACCARD"
MH_ELEMENT_BIT_WIDTH = 32
MH_LSH_BAND = 64
WITH_RAW_DATA = True

# Search: enable Jaccard refinement. Milvus's MHJACCARD metric returns the
# estimated Jaccard similarity directly in its distance field, so the app
# reports it verbatim and never applies 1 - distance.
SEARCH_WITH_JACCARD = True
REFINE_K = 10
DEFAULT_TOP_K = 5
MAX_TOP_K = 20

DEFAULT_THRESHOLD = 0.85
THRESHOLD_MIN = 0.0
THRESHOLD_MAX = 1.0

INDEX_BUILD_POLL_SECONDS = 0.1

CONTENT_MAX_LENGTH = 2000
TITLE_MAX_LENGTH = 256

RUNTIME_RELATIVE_ROOT = Path("artifacts/runtime/minhash-near-duplicate")
DATASET_RELATIVE_ROOT = Path("demos/minhash-near-duplicate/data/original-short-stories-r1")
DOCUMENTS_FILE_NAME = "documents.json"
PRESETS_FILE_NAME = "presets.json"

EXPECTED_DATASET_ID = "original-short-stories-r1"
EXPECTED_DATASET_VERSION = 1
EXPECTED_DOCUMENT_COUNT = 36


class ConfigurationError(ValueError):
    """Raised when a runtime setting would cross the demo boundary."""


def repository_root() -> Path:
    """Return the repository root from the independently packaged backend."""
    return Path(__file__).resolve().parents[5]


def default_runtime_root() -> Path:
    """Return the only default root for generated demo artifacts."""
    return (repository_root() / RUNTIME_RELATIVE_ROOT).resolve()


def default_dataset_root() -> Path:
    """Return the checked-in original-short-stories dataset root."""
    return (repository_root() / DATASET_RELATIVE_ROOT).resolve()


@dataclass(frozen=True)
class RuntimeConfig:
    """Resolved local runtime inputs for one backend process."""

    runtime_root: Path
    dataset_root: Path

    @classmethod
    def from_environment(cls) -> RuntimeConfig:
        allowed_root = default_runtime_root()
        configured_root = Path(os.environ.get("MINHASH_RUNTIME_DIR", str(allowed_root))).resolve()
        if configured_root != allowed_root and not configured_root.is_relative_to(allowed_root):
            raise ConfigurationError(
                "MINHASH_RUNTIME_DIR must stay under artifacts/runtime/minhash-near-duplicate"
            )
        dataset_root = Path(
            os.environ.get("MINHASH_DATASET_DIR", str(default_dataset_root()))
        ).resolve()
        if dataset_root != default_dataset_root():
            raise ConfigurationError(
                "MINHASH_DATASET_DIR must identify the tracked original-short-stories dataset"
            )
        return cls(runtime_root=configured_root, dataset_root=dataset_root)
