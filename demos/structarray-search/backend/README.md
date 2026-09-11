# StructArray Search Backend

This package provides the deterministic CoVLA data checks, the local BGE-M3 ONNX
embedding backend, the FastAPI application, and the real PyMilvus StructArray
parent + child hybrid lifecycle for the parent demo. See
[`../README.md`](../README.md) for the complete schema, source-data contract,
and data-use restrictions.

## Local checks

Run all routine checks without enabling the opt-in Milvus integration test:

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

The deterministic data gate is available from the repository root:

```bash
make structarray-search-data-check
```

It reads the approved CoVLA source files in place and validates the fixed
30-video slice. It does not connect to Milvus.

## Fixed runtime boundary

Every runtime operation is hard-coded to:

- URI: `http://127.0.0.1:49530`
- collection: `milvus3_demos_structarray_hybrid_covla`
- model: `gpahal/bge-m3-onnx-int8` @
  `2b34e84df040034d4b9eabb62383a87c18955822`
- sample size: exactly `30` (the verified 100-sample prefix)

The CLI and API do not accept URI, collection, or model overrides. `prepare`
refuses to replace an existing exact collection, embeds the 30 summaries and
519 non-empty observations locally, creates the collection, inserts rows,
loads it, and waits for both HNSW indexes to finish. `cleanup` drops only the
exact collection and reports the pre-drop index names.

The stable container entry is:

```bash
make structarray-search-image-check
```

It builds the unified image, starts it, health-checks the live endpoint, and uses
the exact cleanup trap.

## Data and model safety

The CoVLA source root and model cache are read-only inputs mounted into the
container. Do not copy or commit source JSON, model weights, vectors, derived
rows, or credentials.

CoVLA is gated and restricted to academic/non-commercial use by the local
pipeline notice. Review the current authoritative upstream terms before any
redistribution, public deployment, or commercial use.

The BGE-M3 ONNX model is the `gpahal/bge-m3-onnx-int8` quantized export and is
resolved only from the local Hugging Face hub cache; no download occurs at
runtime. The backend reports the exact model id and revision it uses.
