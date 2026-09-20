# Milvus 3.0 Demos

Interactive, self-contained demos for [Milvus 3.0](https://milvus.io). Each demo turns one
Milvus 3.0 capability into a concrete, runnable story — a backend, a web UI, a fixed dataset,
and its own container image — so you can see what changes when search understands more than a
single vector. More demos will be added over time.

## Demos at a glance

| Demo                                                                                 | Milvus 3.0 capability                               | Live demo                                                              |
| ------------------------------------------------------------------------------------ | --------------------------------------------------- | ---------------------------------------------------------------------- |
| [1 · Function Chain Rerank](#demo-1--function-chain-rerank)                          | L0 Function Chain + XGBoost UBJ FileResource rerank | [function-chain-rerank](https://demos.milvus.io/function-chain-rerank) |
| [2 · StructArray Parent + Child](#demo-2--structarray-parent--child-semantic-hybrid) | `ARRAY<STRUCT>` multi-route hybrid search           | [structarray-search](https://demos.milvus.io/structarray-search)       |
| [3 · EmbeddingList MAX_SIM](#demo-3--embeddinglist-max_sim-nasa-handbook-retrieval)  | `EmbeddingList` multi-vector `MAX_SIM_COSINE`       | [embedding-list](https://demos.milvus.io/embedding-list)               |

> Live demos are served from a shared Milvus 3.0 stack behind `demos.milvus.io`. If a link is
> temporarily unavailable, the same demo runs locally — see [Quick start](#quick-start).

## How the demos relate

One shared Milvus 3.0 standalone stack backs all three demos; each demo owns a distinct
collection and connects to the stack with its own `MILVUS_URI`. The demos themselves are fully
independent — each is a single image that serves its compiled front end and its `/api/v1/` API
from one port. There is no shared package, portal, or gateway between them.

```text
                 ┌──────────────────────────────────────────┐
                 │   Shared Milvus 3.0 standalone stack      │
                 │   etcd · MinIO · milvusdb/milvus:v3.0.0   │
                 └────────────────────┬─────────────────────┘
                                      │  MILVUS_URI (one per demo)
        ┌─────────────────┬───────────┴───────────┬─────────────────┐
        ▼                 ▼                       ▼                 ▼
   Demo 1 image       Demo 2 image            Demo 3 image
   Function Chain     StructArray             EmbeddingList
   :48020             :48030                  :48040
```

## Demo 1 — Function Chain Rerank

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

## Demo 2 — StructArray Parent + Child Semantic Hybrid

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

## Demo 3 — EmbeddingList MAX_SIM NASA Handbook Retrieval

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

## Repository layout

```text
demos/function-chain-rerank/        Demo 1: backend, web UI, and synthetic catalog
demos/structarray-search/           Demo 2: backend, web UI, and synthetic driving dataset
demos/embedding-list-max-sim/       Demo 3: backend, web UI, and NASA handbook dataset
infra/milvus/                       Shared Milvus 3.0 stack (etcd + MinIO + standalone)
infra/function-chain-rerank/        Docker Compose for the isolated rerank image
infra/structarray-search/           Docker Compose for the isolated hybrid image
infra/embedding-list-max-sim/       Docker Compose for the isolated ColSmol image
deploy/                             Kubernetes manifests for the shared stack and three apps
```

Each demo is fully self-contained: its backend declares its own Python dependencies, its web
directory declares its own Node dependencies, and its Dockerfile builds the UI and the API into a
single image. The backend serves both the compiled front-end and the `/api/v1/` endpoints from one
port. Model weights are baked into each image at build time, so no external model mount or
embedding API is needed at runtime.

## Quick start

Dependencies: Docker 27+ and Docker Compose v2. (Development-only steps additionally need
`uv` 0.11+ and Node.js 22 — see [Development](#development).)

```bash
# 1. Start the shared Milvus 3.0 stack (etcd + MinIO + standalone)
export MILVUS_RUNTIME_DIR=/path/to/writable/runtime/dir
docker compose -f infra/milvus/docker-compose.yml up -d

# 2. Build and run one demo (repeat for any demo you want)
docker compose -f infra/embedding-list-max-sim/docker-compose.yml up -d --build

# 3. Open it in a browser
#    Demo 1 → http://127.0.0.1:48020/
#    Demo 2 → http://127.0.0.1:48030/
#    Demo 3 → http://127.0.0.1:48040/
```

`MILVUS_RUNTIME_DIR` must point to a writable directory where etcd, MinIO, and Milvus persist
their data. The three demos bind to loopback only.

For step-by-step instructions — starting Milvus, building each independent image, running the
demos, and the production Kubernetes deployment — see [DEPLOYMENT.md](DEPLOYMENT.md).

## Development

Install development dependencies and run the fast, offline checks:

```bash
make bootstrap            # uv sync each backend + npm install
make validate             # data checks + format + lint + typecheck + tests + compose config
```

Per-demo data checks verify each dataset's bytes and metadata without containers:

```bash
make function-chain-rerank-data-check
make structarray-search-data-check
make embedding-list-max-sim-data-check
```

Real container + Milvus integration is opt-in per demo via `<demo-id>-image-check` (see
[`DEPLOYMENT.md`](DEPLOYMENT.md)). Each demo owns its own backend `uv.lock`, web
`package-lock.json`, and Dockerfile, so a change to one demo cannot change the other two images.

## License

Project-authored catalog metadata, text, and synthetic images: CC0 1.0
(https://creativecommons.org/publicdomain/zero/1.0/). See
[`demos/function-chain-rerank/data/synthetic-commerce-catalog-r1/LICENSE-CONFLICT.md`](demos/function-chain-rerank/data/synthetic-commerce-catalog-r1/LICENSE-CONFLICT.md)
and
[`demos/function-chain-rerank/data/synthetic-commerce-catalog-r1/NOTICE.md`](demos/function-chain-rerank/data/synthetic-commerce-catalog-r1/NOTICE.md)
and
[`demos/structarray-search/data/synthetic-driving-scenes-r1/NOTICE.md`](demos/structarray-search/data/synthetic-driving-scenes-r1/NOTICE.md)
for the dataset-specific notice and license record.

The Demo 3 NASA Systems Engineering Handbook is a U.S. Government work with NTRS rights
determination `PUBLIC_USE_PERMITTED`; it is redistributed as the official, unmodified PDF and
page renders with full source metadata. Demo 2 publishes no third-party data, model weights, or
vectors — only project-authored source code.
