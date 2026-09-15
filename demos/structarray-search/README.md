# StructArray Parent + Child Semantic Hybrid

One query, one model, three answers. This demo shows how Milvus 3.0's
StructArray (`ARRAY<STRUCT>`) keeps a coarse parent record (a synthetic driving
video) and its fine child records (object observations) in a single row, and
how one semantic query can be answered three ways:

1. **Parent-only** searches the `summary_vector` field for the best matching
   video.
2. **Child-only** searches the nested `observations[description_vector]`
   element field and groups every hit back to one parent `video_id`.
3. **Fused** collapses each parent's best three child scores (`topk_sum(3)`)
   and weighted-reranks the child route against the parent route with
   `WeightedRanker`.

The query is embedded at request time by a CPU-only, quantized BGE-M3 ONNX
model. The same model embeds the parent summaries and child descriptions at
prepare time. Images are representative video frames used only for display;
they do not participate in embedding or retrieval.

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

- **Collection** `milvus3_demos_structarray_hybrid_synthetic` (exact, not
  overridable).
- **Milvus** `http://127.0.0.1:49530`, version exactly `3.0.0`.
- **Model** `gpahal/bge-m3-onnx-int8` @ revision
  `2b34e84df040034d4b9eabb62383a87c18955822`, dense output `dense_vecs`,
  1024 dimensions, CPU float32.
- **Dataset** `synthetic-driving-scenes-r1`: 30 fictional video records and 30
  generated representative frames checked into this repository.
- **Observations** 540 deterministic child records: 18 per video and below the
  StructArray capacity of 128.

The model is resolved from the local Hugging Face hub cache at
`${HF_HUB_CACHE:-~/.cache/huggingface/hub}`. No download is performed during
normal local execution.

## Synthetic data design

The manifest lives at
`data/synthetic-driving-scenes-r1/scenes.json`. It defines 30 original scene
summaries, six visible traffic objects per scene, and the prompt used to create
each representative frame. Three observation phases per object expand into
540 searchable children at load time.

The four query presets deliberately split intent across levels:

- the parent summary contains the road environment and object counts but no
  vehicle colors;
- a child description contains the vehicle type and color but no global road
  environment;
- the fused path combines both levels.

Each preset has three positive videos and two same-scene, wrong-color
distractors. This prevents the parent route from inferring the answer from
scene and object count alone.

Parent, child, and fusion result cards all show a full video-level
representative frame. The same video always resolves to the same image in all
three columns, keeping the comparison visually aligned. See
`data/synthetic-driving-scenes-r1/NOTICE.md` for asset provenance.

## Milvus schema

| Parent field     | Type                     | Purpose                          |
| ---------------- | ------------------------ | -------------------------------- |
| `video_id`       | `VARCHAR(64)` primary    | Parent identity and group key    |
| `video_summary`  | `VARCHAR(4096)`          | Video-level context              |
| `source_ordinal` | `INT64`                  | Deterministic source position    |
| `summary_vector` | `FLOAT_VECTOR(1024)`     | Parent semantic route            |
| `observations`   | `ARRAY<STRUCT>`, max 128 | Searchable child elements        |

Each `observations` element holds `description VARCHAR(512)`,
`description_vector FLOAT_VECTOR(1024)`, `object_type VARCHAR(32)`,
`frame_id INT64`, `image_id VARCHAR(64)`, and `bbox_x1…y2 INT64` scalars.

Indexes are HNSW/COSINE with `M=16`, `efConstruction=128`; search uses
`ef=128`.

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
The child path returns each hit's `offset` and matched observation; the fusion
path is entity-level and therefore carries no child offset. Scores are COSINE
similarity, not probabilities. Fused scores are arctan-normalized per route
then summed, so they are not bounded to `[0, 1]`.

## Container boundary

The Docker image includes the synthetic manifest, its 30 generated JPEGs, the
backend runtime, the web build, and the pinned BGE-M3 ONNX snapshot. It does
not require an external dataset or model mount. The only persistent writable
mount is the ignored `artifacts/runtime/structarray-search` directory; `/tmp`
is ephemeral and the application image filesystem remains read-only.

## Development checks

```bash
make structarray-search-data-check

UV_OFFLINE=1 uv run --offline --project demos/structarray-search/backend \
  --group dev pytest demos/structarray-search/backend/tests

npm test --workspace @milvus3-demos/structarray-search
npm run typecheck --workspace @milvus3-demos/structarray-search
```

Real container and Milvus integration is opt-in:

```bash
make structarray-search-image-check
```

## Known limitations

- This is an explanatory 30-video synthetic dataset, not a benchmark or a
  complete driving retrieval system.
- Each video uses one representative still across its observations; the demo
  does not stream or play source video.
- One best matching child element is hydrated per grouped parent in the child
  column; the fusion column is entity-level with no child offset.
- Query input is free-form text embedded by the local model, not an image.
- Authentication, authorization, multi-user ownership, and production
  retention policies are outside this local demo.
