# Milvus 3.0 Demos — Function Chain Rerank

An interactive demo for [Milvus 3.0](https://milvus.io) search capabilities. This repository
currently ships a single demo: an e-commerce product rerank that runs an **L0 Function Chain**
and an **XGBoost UBJ FileResource** entirely inside Milvus.

## What it shows

For one natural-language shopping query, the demo produces two server-side rankings and lets you
compare them:

1. `vector_order` — Milvus COSINE vector recall;
2. `business_order` — the same recall re-ranked in Milvus by a Function Chain that normalizes
   five business features and scores them with an XGBoost model loaded from a UBJ FileResource.

```text
FastAPI → Milvus 3.0 vector recall
                └→ L0 Function Chain → XGBoost UBJ FileResource
```

The backend preserves both returned arrays exactly as Milvus returns them; it never performs the
final sort. The web UI renders the two orders side by side and animates the movement between them.

## Repository layout

```text
demos/function-chain-rerank/   the demo: backend, web UI, and synthetic catalog
packages/python/milvus-demo-common/   shared settings and Milvus health probe
packages/web/demo-ui/                 shared UI primitives (badge, page frame)
infra/function-chain-rerank/          Docker Compose for the isolated backend
```

## Data

The catalog is **fully synthetic**. Its 240 authored, fictional products (20 product types) are
not derived from any real retailer or public dataset, and all business telemetry (price, rating,
clicks, sales, release date) is simulated. Metadata and text are dedicated to the public domain
under CC0 1.0. See
[`demos/function-chain-rerank/README.md`](demos/function-chain-rerank/README.md) for the full
provenance and model contracts.

> **Note:** the 240 solid-color placeholder JPEGs are intentionally **not** committed to this
> repository. The catalog metadata, hash manifests, and training records are included; images
> will be added once final, clearly licensed product photography is available. Until then, the
> strict loader's image validation will not pass.

## Quick start

The demo requires Milvus 3.0.0, a Hugging Face hub cache for the frozen BGE-M3 ONNX encoder, and
MinIO-backed FileResource storage. See the
[backend README](demos/function-chain-rerank/README.md) for the full prerequisites and contracts.

```bash
# Install Python and Node dependencies
make bootstrap

# Verify the synthetic catalog metadata offline
make function-chain-rerank-data-check
```

## License

Project-authored catalog metadata and text: CC0 1.0
(https://creativecommons.org/publicdomain/zero/1.0/). See
[`demos/function-chain-rerank/data/synthetic-commerce-catalog-r1/LICENSE-CONFLICT.md`](demos/function-chain-rerank/data/synthetic-commerce-catalog-r1/LICENSE-CONFLICT.md)
and
[`demos/function-chain-rerank/data/synthetic-commerce-catalog-r1/NOTICE.md`](demos/function-chain-rerank/data/synthetic-commerce-catalog-r1/NOTICE.md)
for the dataset-specific notice and license record.
