"""Fixed CPU, model, Milvus, and filesystem boundaries for the movie demo."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DEMO_ID = "movie-discovery"
CAPABILITY = "Faceted semantic search with Search Aggregation"

# Deployable against a shared Milvus by setting MILVUS_URI; defaults to the
# local standalone endpoint for development.
MILVUS_URI = os.environ.get("MILVUS_URI", "http://127.0.0.1:49530")
MILVUS_EXPECTED_VERSION = "3.0.0"
MILVUS_TIMEOUT_SECONDS = 30.0

COLLECTION_NAME = "milvus3_demos_movie_catalog"
VECTOR_FIELD = "synopsis_vector"
ANNS_FIELD = VECTOR_FIELD
INDEX_NAME = "milvus3_demos_movie_synopsis_vector"

METRIC_TYPE = "COSINE"
VECTOR_DIMENSION = 1024
INDEX_METRIC = "COSINE"
INDEX_TYPE = "HNSW"
INDEX_M = 16
INDEX_EF_CONSTRUCTION = 128
SEARCH_EF = 128
INDEX_BUILD_POLL_SECONDS = 2.0
INDEX_BUILD_TIMEOUT_SECONDS = 120.0

# Facet fields must be single-value scalar fields: the design uses one primary
# genre and a precomputed decade, never an ARRAY<...> multi-label field.
FACET_FIELDS = {
    "genre": ("primary_genre", "Genre"),
    "decade": ("decade", "Decade"),
    "region": ("region", "Region"),
}
DEFAULT_FACET = "genre"
GENRES = (
    "Adventure",
    "Animation",
    "Comedy",
    "Drama",
    "Sci-Fi",
    "Thriller",
)

# Search Aggregation budget: the maximum TopHits.size decides each full group
# key's candidate budget. Keeping it modest keeps groups scannable.
AGG_TOP_HITS_SIZE = 3
AGG_ROOT_SIZE = 10
DEFAULT_TOP_K = 12
MAX_TOP_K = 20

MAX_RUNTIME_FILTER = 120

# ---------------------------------------------------------------------------
# The 2x2 capability matrix. Each cell is one (api, mode) pair:
#
#            sort                              group
#   query    server-side ORDER BY over the      GROUP BY with exact aggregate
#            whole catalog (no vector)         functions (no vector)
#   search   re-orders the retained top-K       facet buckets carrying metrics
#            only (candidate set unchanged)    and best-matching hits
#
# These are deliberately separate code paths so the page can state each cell's
# statistical contract without blurring them together.
# ---------------------------------------------------------------------------
PANEL_APIS = ("query", "search")
PANEL_MODES = ("sort", "group")

# query+sort: server-side ORDER BY. `movie_id` is appended as a tiebreaker so
# offset pagination stays deterministic when the sort key has duplicate values.
BROWSE_SORT_FIELDS = {
    "year": ("release_year", "Year"),
    "runtime": ("runtime_minutes", "Runtime"),
    "title": ("title", "Title"),
}
DEFAULT_BROWSE_SORT = "year"
BROWSE_SORT_DIRECTIONS = ("asc", "desc")
DEFAULT_BROWSE_DIRECTION = "desc"
BROWSE_PAGE_SIZE = 12
BROWSE_MAX_PAGE_SIZE = 50
# Always-appended secondary key; guarantees a total order for pagination.
BROWSE_TIEBREAK_FIELD = "movie_id"

# search+sort: which scalar the retained top-K is re-ordered by. "score" is the
# untouched relevance order and therefore the honest baseline for comparison.
RANKED_SORT_FIELDS = {
    "score": ("distance", "Relevance"),
    "year": ("release_year", "Year"),
    "runtime": ("runtime_minutes", "Runtime"),
}
DEFAULT_RANKED_SORT = "score"

# query+group: exact aggregate functions. `group_by_fields` (plural) is required
# for pymilvus to aggregate per group; the singular form silently ignores them.
EXACT_METRICS = {
    "count": ("count(*)", "Count"),
    "avg_runtime": ("avg(runtime_minutes)", "Avg runtime"),
    "min_year": ("min(release_year)", "Earliest"),
    "max_year": ("max(release_year)", "Latest"),
    "sum_runtime": ("sum(runtime_minutes)", "Total runtime"),
}
DEFAULT_EXACT_METRICS = ("count", "avg_runtime", "min_year", "max_year")

# search+group: approximate per-bucket metrics computed over retained hits only.
# Shape is pymilvus' own {alias: {aggregate_op: field_name}}.
APPROX_METRICS = {
    "avg_runtime": {"avg": "runtime_minutes"},
    "max_year": {"max": "release_year"},
}

MODEL_ID = "gpahal/bge-m3-onnx-int8"
MODEL_REVISION = "2b34e84df040034d4b9eabb62383a87c18955822"
MODEL_DIMENSION = VECTOR_DIMENSION
MODEL_BATCH_SIZE = 32

EXPECTED_MOVIE_COUNT = 90

RUNTIME_RELATIVE_ROOT = Path("artifacts/runtime/movie-discovery")
DATASET_RELATIVE_ROOT = Path("demos/movie-discovery/data/movie-catalog-r1")
MOVIES_FILE_NAME = "movies.json"
PRESETS_FILE_NAME = "presets.json"
EXPECTED_DATASET_ID = "movie-catalog-r1"
EXPECTED_DATASET_VERSION = 1


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


def default_dataset_root() -> Path:
    """Return the checked-in movie catalog dataset root."""
    return (repository_root() / DATASET_RELATIVE_ROOT).resolve()


@dataclass(frozen=True)
class RuntimeConfig:
    """Resolved local runtime inputs for one backend process."""

    runtime_root: Path
    hf_hub_cache: Path
    data_root: Path
    movies_path: Path
    presets_path: Path

    @classmethod
    def from_environment(cls) -> RuntimeConfig:
        allowed_root = default_runtime_root()
        configured_root = Path(os.environ.get("MOVIE_RUNTIME_DIR", str(allowed_root))).resolve()
        if configured_root != allowed_root and not configured_root.is_relative_to(allowed_root):
            raise ConfigurationError(
                "MOVIE_RUNTIME_DIR must stay under artifacts/runtime/movie-discovery"
            )
        data_root = default_dataset_root()
        return cls(
            runtime_root=configured_root,
            hf_hub_cache=default_hf_hub_cache(),
            data_root=data_root,
            movies_path=(data_root / MOVIES_FILE_NAME).resolve(),
            presets_path=(data_root / PRESETS_FILE_NAME).resolve(),
        )

    @property
    def embedding_cache_root(self) -> Path:
        return self.runtime_root / "embedding-cache"
