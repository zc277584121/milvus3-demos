# StructArray hybrid backend

This package provides deterministic checks for the checked-in synthetic
driving dataset, CPU BGE-M3 ONNX embedding, Milvus 3.0 StructArray
provisioning, and the parent-only, child-only, and weighted-fusion search
paths.

## Local validation

```bash
UV_OFFLINE=1 uv lock --check --offline --project demos/structarray-search/backend
UV_OFFLINE=1 uv run --offline --project demos/structarray-search/backend \
  ruff format --check demos/structarray-search/backend/src \
  demos/structarray-search/backend/tests
UV_OFFLINE=1 uv run --offline --project demos/structarray-search/backend \
  ruff check demos/structarray-search/backend/src \
  demos/structarray-search/backend/tests
UV_OFFLINE=1 uv run --offline --project demos/structarray-search/backend \
  pytest demos/structarray-search/backend/tests -q
```

Validate the manifest and all 30 representative frames:

```bash
make structarray-search-data-check
```

The loader reads
`demos/structarray-search/data/synthetic-driving-scenes-r1/scenes.json`,
validates the dataset identity, ordered video IDs, image references, object
boxes, fixed counts, and StructArray capacity, then deterministically expands
six objects across three observation phases per video.

## Fixed contract

- collection: `milvus3_demos_structarray_hybrid_synthetic`
- 30 parent rows
- 540 child observations
- 30 generated representative JPEGs
- vectors: 1024-dimensional CPU BGE-M3 embeddings
- indexes: HNSW/COSINE on parent and nested child vector fields
- child grouping: `video_id`
- fusion: `topk_sum(3)` followed by `WeightedRanker`

The synthetic data is repository-local and read-only at runtime. Only
embedding caches and other generated runtime state are written below
`artifacts/runtime/structarray-search`.
