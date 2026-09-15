# Milvus 3.0 Demos

A growing collection of interactive, self-contained demos for
[Milvus 3.0](https://milvus.io). Each demo turns one Milvus capability into a concrete, runnable
story — a backend, a web UI, a fixed dataset, and its own image boundary — so you can see what
changes when search understands more than a single vector. More demos will be added over time.

## Demos

### Demo 1 — Function Chain Rerank

An e-commerce product rerank that runs an **L0 Function Chain** and an **XGBoost UBJ
FileResource** entirely inside Milvus.

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

**Data.** The catalog is **fully synthetic**. Its 240 authored, fictional products (20 product
types) are not derived from any real retailer or public dataset, and all business telemetry
(price, rating, clicks, sales, release date) is simulated. Metadata and text are dedicated to
the public domain under CC0 1.0.

> **Note:** the 240 project-generated synthetic catalog images are committed to this repository.
> They are 256×256 hash-pinned synthetic renderings, not photographs of real products, and are
> dedicated to the public domain under CC0 1.0. The strict loader's image validation passes
> against the checked-in `hash-manifest.sha256`.

See [`demos/function-chain-rerank/README.md`](demos/function-chain-rerank/README.md) for the full
provenance, model contracts, and API reference.

### Demo 2 — EmbeddingList MAX_SIM NASA Handbook Retrieval

A 40-page selection from the **NASA Systems Engineering Handbook, NASA/SP-2016-6105 Rev 2**
retrieved with real ColSmol multi-vector embeddings and Milvus 3.0 `EmbeddingList`
`MAX_SIM_COSINE` search.

Page patches and query tokens are both 128-dimensional. Milvus ranks pages by the maximum cosine
similarity between query tokens and page patches; the demo also computes a local spatial
explanation (token matches, patch traces, and heatmaps) from the same real vectors — this
explanation is produced application-side and never returned by Milvus.

```text
FastAPI → ColSmol CPU page/query multi-vectors
        → Milvus EmbeddingList (patches[patch_embedding]) MAX_SIM_COSINE
        → local MaxSim scoring + spatial explanation for the Top 3
```

**Data.** The tracked production dataset is the NASA handbook PDF plus 40 unmodified rendered
pages, recorded with NTRS source metadata, rights determination `PUBLIC_USE_PERMITTED`, and a
complete SHA-256 manifest. NASA does not endorse Milvus or this demo.

See [`demos/embedding-list-max-sim/README.md`](demos/embedding-list-max-sim/README.md) for the
fixed NTRS source contract, model contract, and API reference.

### Demo 3 — StructArray Parent + Child Semantic Hybrid

One query, one model, three answers. A driving-video dataset is stored as a Milvus 3.0
`ARRAY<STRUCT>` row per parent (video) with its fine child records (object observations), and one
semantic query is answered three ways:

1. **Parent-only** — search the `summary_vector` field;
2. **Child-only** — search the nested `observations[description_vector]` element field and group
   every hit back to one `video_id`;
3. **Fused** — collapse each parent's best three child scores (`topk_sum(3)`) and weighted-rerank
   against the parent route with `WeightedRanker`.

The query is embedded at request time by a CPU-only quantized BGE-M3 ONNX model; no external
embedding API and no precomputed companion-vector file are used.

```text
Browser → StructArray FastAPI
        → BGE-M3 ONNX (CPU) query embedding (1024 dims)
        → summary_vector                           parent-only search
        → observations[description_vector] + group_by_field=video_id
        → WeightedRanker + element_scope collapse(topk_sum, 3)
```

**Data.** `synthetic-driving-scenes-r1` contains 30 original fictional video records, 540
deterministic child observations, and 30 generated representative frames. The text and images are
checked into the repository, and no external driving dataset is required. Images are display-only;
retrieval uses the parent summaries and child descriptions.

See [`demos/structarray-search/README.md`](demos/structarray-search/README.md) for the full
schema, synthetic-data contract, and provenance.

## Repository layout

```text
demos/function-chain-rerank/        Demo 1: backend, web UI, and synthetic catalog
demos/embedding-list-max-sim/       Demo 2: backend, web UI, and NASA handbook dataset
demos/structarray-search/           Demo 3: backend, web UI, and synthetic driving dataset
infra/function-chain-rerank/        Docker Compose for the isolated rerank image
infra/embedding-list-max-sim/       Docker Compose for the isolated ColSmol image
infra/structarray-search/           Docker Compose for the isolated hybrid image
```

Each demo is fully self-contained: its backend declares its own Python dependencies, its web
directory declares its own Node dependencies, and its Dockerfile builds the UI and the API into a
single image. The backend serves both the compiled front-end and the `/api/v1/` endpoints from one
port — there is no shared package, portal, or gateway layer between demos.

## Quick start

For step-by-step build and run instructions — starting Milvus, building the three independent
images, and deploying each demo — see [DEPLOYMENT.md](DEPLOYMENT.md).

Each demo has its own prerequisites and fixed runtime boundary. Demo 1 requires Milvus 3.0.0 and
MinIO-backed FileResource storage; Demo 2 requires a local offline ColSmol model cache; Demo 3
uses its checked-in synthetic dataset and requires a local BGE-M3 ONNX model cache. See each
demo's README for the full contracts.

```bash
# Install Python and Node dependencies
make bootstrap

# Verify each demo's dataset metadata offline
make function-chain-rerank-data-check
make embedding-list-max-sim-data-check
make structarray-search-data-check
```

## License

Project-authored catalog metadata, text, and synthetic images: CC0 1.0
(https://creativecommons.org/publicdomain/zero/1.0/). See
[`demos/function-chain-rerank/data/synthetic-commerce-catalog-r1/LICENSE-CONFLICT.md`](demos/function-chain-rerank/data/synthetic-commerce-catalog-r1/LICENSE-CONFLICT.md)
and
[`demos/function-chain-rerank/data/synthetic-commerce-catalog-r1/NOTICE.md`](demos/function-chain-rerank/data/synthetic-commerce-catalog-r1/NOTICE.md)
and
[`demos/structarray-search/data/synthetic-driving-scenes-r1/NOTICE.md`](demos/structarray-search/data/synthetic-driving-scenes-r1/NOTICE.md)
for the dataset-specific notice and license record.

The Demo 2 NASA Systems Engineering Handbook is a U.S. Government work with NTRS rights
determination `PUBLIC_USE_PERMITTED`; it is redistributed as the official, unmodified PDF and
page renders with full source metadata. Demo 3 publishes no third-party data, model weights, or
vectors — only project-authored source code.
