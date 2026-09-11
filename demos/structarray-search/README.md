# StructArray Parent + Child Semantic Hybrid

One query, one model, three answers. This demo shows how Milvus 3.0's
StructArray (`ARRAY<STRUCT>`) keeps a coarse parent record (a driving video)
and its fine child records (object observations) in a single row, and how one
semantic query can be answered three ways:

1. **Parent-only** — search the `summary_vector` field for the best matching
   video.
2. **Child-only** — search the nested `observations[description_vector]`
   element field and group every hit back to one parent `video_id`.
3. **Fused** — collapse each parent's best three child scores (`topk_sum(3)`)
   and weighted-rerank the collapsed child route against the parent route with
   `WeightedRanker`.

The query is embedded at request time by a CPU-only, quantized BGE-M3 ONNX
model; the same model embedded the parent summaries and child descriptions at
prepare time. There is no precomputed companion-vector file and no external
embedding API.

```text
Browser /
  → unified StructArray FastAPI static+API server on 127.0.0.1:48030
  → Milvus 3.0.0 on http://127.0.0.1:49530
  → BGE-M3 ONNX (CPU) embeds the query to 1024 dims
  → summary_vector           parent-only search
  → observations[description_vector] + group_by_field=video_id
  → WeightedRanker + element_scope collapse(topk_sum, 3)
```

## Fixed runtime boundaries

- **Collection** `milvus3_demos_structarray_hybrid_covla` (exact, not
  overridable).
- **Milvus** `http://127.0.0.1:49530`, version exactly `3.0.0`.
- **Model** `gpahal/bge-m3-onnx-int8` @ revision
  `2b34e84df040034d4b9eabb62383a87c18955822`, dense output `dense_vecs`,
  1024 dimensions, CPU float32.
- **Slice** the first 30 videos of `video_ready/video_data_100_samples.json`
  (SHA-256 `c7accf8cd63c77ba860242aa6415b9a3c44089731312818a50e6d89cde673b0a`),
  with the 10-record prefix independently verified
  (`4dbbc43b9ccb97602f28a93acc10428bf51ae62a2b6e8503abe200d54d0524f1`).
- **Observations** only non-empty descriptions are searchable. The 30-video
  slice yields 519 searchable observations across 27 of 30 parents, below the
  StructArray capacity of 128.

The model is resolved from the local Hugging Face hub cache at
`${HF_HUB_CACHE:-~/.cache/huggingface/hub}`. No download is performed.

## Milvus schema

| Parent field     | Type                     | Purpose                          |
| ---------------- | ------------------------ | -------------------------------- |
| `video_id`       | `VARCHAR(64)` primary    | Parent identity and group key    |
| `video_summary`  | `VARCHAR(4096)`          | Source video context             |
| `source_ordinal` | `INT64`                  | Deterministic source position    |
| `summary_vector` | `FLOAT_VECTOR(1024)`     | Parent semantic route            |
| `observations`   | `ARRAY<STRUCT>`, max 128 | Searchable child elements        |

Each `observations` element holds `description VARCHAR(512)`,
`description_vector FLOAT_VECTOR(1024)`, `object_type VARCHAR(32)`,
`frame_id INT64`, `image_id VARCHAR(64)`, and `bbox_x1…y2 INT64` scalars.

Indexes are HNSW/COSINE with `M=16`, `efConstruction=128`; search uses `ef=128`.

## API

- `GET /healthz/live`, `GET /healthz/ready`
- `GET /api/v1/status`
- `POST /api/v1/prepare`
- `POST /api/v1/search` — `{ "query": str, "limit": 1..20, "parent_weight": 0..1, "collapse_strategy": "topk_sum" }`
- `POST /api/v1/cleanup`

`parent_weight` and the derived `child_weight = 1 - parent_weight` feed
`WeightedRanker`. The UI slider deliberately stops at 0.1/0.9: WeightedRanker
always unions both routes, so a 0/1 weight would only zero-fill the other side
rather than disable it.

Search responses contain `paths.parent`, `paths.child`, and `paths.fusion`.
The child path returns each hit's `offset` and the matched observation; the
fusion path is entity-level and therefore carries no child offset (documented
Milvus behavior). Scores are COSINE similarity, not probabilities; fused scores
are arctan-normalized per route then summed, so they are not bounded to [0,1].

## Data source and use restrictions

The read-only source root is supplied via the `COVLA_DATA_DIR` environment
variable (defaulting to `~/.cache/covla-dataset`). The approved inputs are:

- `video_ready/video_data_10_samples.json`;
- `video_ready/video_data_100_samples.json`.

The local CoVLA pipeline notice restricts this gated data to
academic/non-commercial use and prohibits unauthorized redistribution of the
dataset or derivative works. Do not commit, copy into images, redistribute, or
publicly serve the source JSON, vectors, model weights, or credentials. Review
the current authoritative CoVLA terms before any public deployment or
commercial use.

## Container and mount boundary

`demos/structarray-search/Dockerfile` installs only the backend's frozen uv
runtime and never copies CoVLA data, model weights, or reports. The Compose
file mounts:

- `${COVLA_DATA_DIR}` at `/data/covla:ro`;
- `${HF_HUB_CACHE}` at `/model-cache/huggingface/hub:ro`;
- the ignored `artifacts/runtime/structarray-search` directory as the only
  writable mount;
- `/tmp` as an ephemeral tmpfs while the image root filesystem remains
  read-only.

The backend uses host networking and binds only `127.0.0.1:48030`, serving both
the browser UI and the `/api/v1/` endpoints from that single address.

## Development checks

```bash
# Offline data contract check
make structarray-search-data-check

# Backend unit tests (embedding and Milvus are stubbed)
UV_OFFLINE=1 uv run --offline --project demos/structarray-search/backend \
  --group dev pytest demos/structarray-search/backend/tests

# Web unit tests and typecheck
npm test --workspace @milvus3-demos/structarray-search
npm run typecheck --workspace @milvus3-demos/structarray-search
```

Real Milvus integration is opt-in:

```bash
make structarray-search-image-check
```

## Known limitations

- This is an explanatory 30-video slice, not a benchmark or a complete driving
  retrieval system.
- Empty-description observations are excluded at prepare time.
- One best matching child element is hydrated per grouped parent in the child
  column; the fusion column is entity-level with no child offset.
- Query input is free-form text embedded by the local model, not an image.
- Authentication, authorization, multi-user ownership, and production
  retention policies are outside this local demo.
