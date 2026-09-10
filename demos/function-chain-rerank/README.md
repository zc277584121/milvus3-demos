# E-commerce XGBoost Function Chain Rerank Demo

This demo shows two server-produced rankings for the same natural-language shopping query.
Milvus first recalls products with COSINE vector search. A second
`MilvusClient.search(..., function_chains=chain)` call executes an L0 Function Chain and an
XGBoost UBJ FileResource inside Milvus. FastAPI preserves both returned arrays as
`vector_order` and `business_order`; it does not perform the final sort.

```text
Portal → FastAPI → Milvus 3.0 vector recall
                         └→ L0 Function Chain → XGBoost UBJ FileResource
```

## Fully synthetic catalog

The default revision is `synthetic-commerce-catalog-r1`. It contains exactly 240 authored,
fictional products across these 20 fixed product types, in this order:

```text
DESK, RUG, BACKPACK, HEADPHONES, UMBRELLA, BED, SOFA, CHAIR, TABLE, LAMP,
HANDBAG, SHOES, BOOT, SANDAL, HAT, SUITCASE, DRINKING_CUP, PILLOW, SHELF,
PLANTER
```

Nothing here is derived from a real retailer or public dataset. Product identifiers
(`SYN-{TYPE}-{NNN}`), brand names, titles, descriptions, bullet points, and colors are all
invented for the demo. The six intent types carry `strong` (six each), `partial` (one each, two
for headphones), and `intra_type_hard_negative` (one each, two for headphones) records selected
by an explicit, text-only curation rule. Fourteen background types carry `catalog_background`
records, and an expansion pass appends six further `catalog_background` products to every one of
the 20 types (the six intent types included), so every query's vector Top-20 fills with same-type
candidates instead of a single cross-category super-item. Every title is authored directly; there
are no localized titles and no derived English title summaries in this revision.

Every product has a unique 256×256 project-generated synthetic catalog image. These images are
hash-pinned and explicitly described as synthetic renderings rather than photographs of real
products. `source-objects.json` freezes the authored fields, selection rank, synthetic URL, byte
count, MD5, SHA-256, and dimensions, plus the hashes of the project's own
`SYNTHETIC-README.md` and CC0 `LICENSE.txt`. `hash-manifest.sha256` makes the checked-in revision
verifiable without network access. The strict loader rejects missing files, extra files,
traversal, symlinks, hash/size changes, invalid JPEG bytes or dimensions, changed source
relationships, inferred authored values, and altered deterministic fields.

Provenance labels distinguish every source of data in the manifest and per-field:

- `synthetic_authored_metadata` — authored catalog text;
- `synthetic_catalog_identity` — synthetic item IDs and image IDs;
- `synthetic_generated_product_image_object` — project-generated synthetic catalog JPEGs;
- `deterministic_simulated_operational_signal` — fictional business telemetry and labels.

### Source preparation

The source-preparation command copies the two static texts and the 240 canonical checked-in
synthetic JPEGs into a bounded cache below
`artifacts/runtime/synthetic-commerce-catalog-r1/cache`, then emits a frozen source manifest. It
never downloads anything. Prepare and regenerate with the backend's uv environment:

```bash
cd demos/function-chain-rerank/backend

uv run --offline python -m function_chain_demo.synthetic_source prepare \
  --cache ../../../artifacts/runtime/synthetic-commerce-catalog-r1/cache \
  --output ../../../artifacts/runtime/synthetic-commerce-catalog-r1/source-manifest.json

uv run --offline python -m function_chain_demo.dataset generate \
  --source-manifest ../../../artifacts/runtime/synthetic-commerce-catalog-r1/source-manifest.json \
  --source-cache ../../../artifacts/runtime/synthetic-commerce-catalog-r1/cache \
  --force

uv run --offline python -m function_chain_demo.dataset validate
```

## License

The authored catalog metadata and project-generated synthetic images are dedicated to the public
domain under CC0 1.0
(https://creativecommons.org/publicdomain/zero/1.0/). No third-party metadata, photography, or
license texts are included, so there is no attribution obligation and no license conflict
record. `publication_review_required` is `false`.

The images remain synthetic renderings, not photographs of real products. This revision must not
be published as if it contained real product photography. See
[`NOTICE.md`](data/synthetic-commerce-catalog-r1/NOTICE.md) and
[`LICENSE-CONFLICT.md`](data/synthetic-commerce-catalog-r1/LICENSE-CONFLICT.md).

## Simulated operations

The catalog does not supply this demo's business telemetry. Every value in the following list is
fictional and labeled `deterministic_simulated_operational_signal` in both the manifest and each
product's field-level provenance:

- display price;
- rating value;
- 30-day clicks and sales;
- inventory units and capacity;
- return rate;
- release date.

The normalized `rating`, `inventory`, and `freshness` model features are labeled as derived from
those simulated signals. Training labels carry the same simulated provenance. None is a
merchant, seller, inventory, transaction, customer, or marketplace fact.

Thirty-six teaching products use one of five fictional operational profiles (`H`, `G`, `S`,
`N`, or `R`). The profile map is keyed only by stable synthetic item ID. It is independent of
query identity and vector rank, and it does not contain the frozen evaluation order. The
remaining 84 products retain deterministic item-level simulated signals. Display price is
always marked `simulated display-only` in the Web UI; it is not a model input and is never used
in a movement reason.

## Text vectors and model

Products and queries use the same deterministic BGE-M3 ONNX int8 encoder
(`gpahal/bge-m3-onnx-int8`, revision `2b34e84df040034d4b9eabb62383a87c18955822`), producing
1024-dimensional L2-normalized dense vectors. The frozen field order is title, product type,
brand, color, material, style, node name, description, and bullet points. The contract, model,
and tokenizer SHA-256 digests are pinned in `embedding-contract.json`.

The six checked natural queries contain no internal product IDs, answer keys, or business
signals. Four complete query groups train the model and two complete groups validate it; no
query group crosses the split. The 1440 training pairs use this fixed feature order:

1. Milvus semantic `$score`, rounded to five decimal places;
2. simulated normalized `rating`;
3. `popularity` = 0.3/12000 × clicks + 0.7/850 × sales;
4. `price_affinity` = max(0.5, 1 − 0.5/500 × price);
5. `freshness` = exp(ln 0.5 / (180 × 86400) × max(0, |epoch − 1787184000|)).

The simulated label is clamped to `[0, 1]` after applying:

```text
0.80 × semantic_score
+ 0.02 × rating
+ 0.08 × popularity
+ 0.06 × price_affinity
+ 0.04 × freshness
```

Training rounds semantic similarity to five decimal places before label construction and model
input. The real Milvus Function Chain applies the same `round_decimal(..., decimal=5)` operation
before XGBoost. This removes the float-boundary mismatch between offline cosine calculations and
Milvus COSINE scores without changing client behavior.

XGBoost uses one thread, seed 42, deterministic histogram training, depth 2, 128 rounds,
`eta=0.08`, no row or column subsampling, and UBJ output. Its monotonic constraints are
`(1,1,1,1,1)`: all five features must be nondecreasing. Two independent generations are
byte-for-byte identical. With the frozen uv environment the UBJ is 115,265 bytes with SHA-256
`500b01b3e6e6965b97f59301b5d78dac047f4f04760b36e0db32d1910534917e`; the pinned training-record
SHA-256 is `f42d937ba3ceb273fe1c35a41c6598c36af04e9c513fae90a61061ac7f07affb`.

### Training and evaluation gates

`train_model` refuses to save a model unless two offline gates pass:

- `direction_gate` probes every feature on a 101-point counterfactual grid over every training
  row and requires each of the five monotonic directions to hold within 1e-7;
- `business_gate` requires, for all six queries, that the rerank hold its expected-type recall
  in the Top-6 (at least as high as vector recall and at least 5 of 6), that a strong candidate
  with at least two favourable operational signals holds a Top-6 slot, and that a repeat
  inference reproduces the identical order.

`relevance-ground-truth.json` assigns every one of the 240 products to `strong`, `partial`,
`intra_type_hard_negative`, `catalog_background`, or `cross_category_hard_negative` for each of
the six natural queries. It also records human-readable intent and judgment rules. It is
evaluation-only, is not inserted into Milvus, and has no production ranking consumer.

With the synthetic revision the model trains to a validation NDCG@10 of 0.99756758 and a
validation RMSE of 0.00470308. All-query and held-out-query relevant Top-1 rates are 1.0, and all
six queries change order inside their type-relevant candidates.

## API contract

The backend exposes:

- `GET /healthz/live`
- `GET /healthz/ready`
- `GET /api/v1/status`
- `GET /api/v1/queries`
- `GET /api/v1/assets/{manifest-listed-jpeg}`
- `POST /api/v1/search`

Example request:

```json
{
  "query_id": "compact-dark-wood-desk",
  "query_text": "a compact dark wood desk for a small home office"
}
```

`query_id` is optional for arbitrary natural text. If supplied, it must match the fixed query
text. `/queries` exposes only `id`, `query_text`, `story`, `split`, and dataset identity; it
never returns evaluation answer keys. Search responses return independent `vector_order` and
`business_order` arrays exactly as received from the two Milvus searches. Product responses
include authored synthetic fields, image identity and hashes, simulated signals, and field-level
provenance.

The response also identifies the actual server execution contract: L0 rerank, XGBoost operation,
the five inputs in order, five-decimal semantic preprocessing, semantic score as the vector-order
score, business score as the business-order score, and `intermediate_feature_orders: false`.

The asset route serves only the 240 manifest-listed JPEG paths. It rejects unknown files and
traversal, returns `image/jpeg`, and includes immutable caching, dataset revision, SHA-256, and
`nosniff` headers.

The Portal keeps this backend contract intact. Nginx maps
`/api/function-chain/v1/{status,queries,search,assets/...}` to the dedicated backend's
`/api/v1/{status,queries,search,assets/...}` without changing request or response JSON. The Web
route submits both the selected `query_id` and its exact `query_text`; it consumes
`vector_order` and `business_order` in response order and performs no `sort`, `reverse`, or
`toSorted`.

The page is a compact single-column shopping-results view. Each row uses a server-returned array
without mutation and shows the actual semantic, rating, clicks, sales, and listed-date inputs.
A replayable phase strip presents one sequence: API `vector_order` → five-decimal semantic
normalization → Milvus XGBoost FileResource → server business scores → API `business_order`.
Only the last step changes DOM order. It uses FLIP position animation while taking both orders
directly from the API; there is no per-feature intermediate ranking. Reduced-motion preference
skips row motion while preserving the same phases and final order.

Movement text uses only observable response inputs, documented thresholds, both server ranks,
and the full-model server score. It is explicitly labeled an input summary and not an
attribution, feature contribution, causal trace, or SHAP explanation. Price is visibly marked as
simulated display-only and is not referenced by this text. Changing a query clears prior results
immediately; loading, error, retry, disabled controls, and reduced-motion states share the same
request lifecycle.

## Isolated runtime and one-command entry points

The default development names remain limited to this demo:

- Collection: `milvus3_demos_function_chain_rerank_products`
- FileResource: `milvus3_demos_function_chain_rerank_model`
- object: `files/milvus3-demos/function-chain-rerank/xgb-reranker.ubj`

The backend checks for exactly Milvus 3.0.0. A shared or unknown runtime must use
`FUNCTION_CHAIN_RUN_NAMESPACE` plus unused ports. The lifecycle derives a unique container,
network, two Compose projects, Collection, FileResource, MinIO object prefix, and UBJ cache name;
preflight refuses any collision. It targets the already-running project Milvus at
`http://127.0.0.1:49530` but must not start, stop, restart, upgrade, or rename that deployment.

```bash
make function-chain-rerank-data-check
make function-chain-rerank-integration
make function-chain-rerank-image-check
make function-chain-rerank-e2e
make function-chain-rerank-regression
```

Example isolated E2E invocation:

```bash
FUNCTION_CHAIN_RUN_NAMESPACE=s20260826-executor-r3 \
FUNCTION_CHAIN_PORT=48120 API_PORT=48100 PORTAL_PORT=4183 \
FUNCTION_CHAIN_ARTIFACT_DIR="$PWD/artifacts/validation/S-20260826-001/executor-r3-repair/e2e" \
  ./scripts/function_chain_rerank.sh e2e
```

All current lifecycle evidence is constrained below
`artifacts/validation/S-20260826-001/executor-r3-repair/`. Preflight refuses to take over any
selected container, Compose project, network, listener, Collection, FileResource, object, or
cache name. Cleanup is gated by per-run ownership flags and removes only resources named in that
run's `command.json`.

## Development verification

```bash
uv lock --check --project demos/function-chain-rerank/backend
uv run --offline --project demos/function-chain-rerank/backend ruff check src tests
uv run --offline --project demos/function-chain-rerank/backend pytest -q
npm test --workspace @milvus3-demos/function-chain-rerank
npm run typecheck --workspace @milvus3-demos/function-chain-rerank
git diff --check
```

The real Milvus test remains explicitly opt-in:

```bash
RUN_FUNCTION_CHAIN_INTEGRATION=1 \
  uv run --project demos/function-chain-rerank/backend \
  pytest demos/function-chain-rerank/backend/tests/test_real_milvus.py
```

## Limitations

This is an explanatory demo, not a production ranker. Its catalog metadata is authored and its
images are synthetic catalog renderings rather than photographs of real products. All business
signals and labels are fictional. The text encoder is a frozen deterministic BGE-M3 ONNX int8
model, not a fine-tuned ranking model. The model is not calibrated for revenue, fairness, safety,
or search quality. Production systems also need authorization, monitoring, realistic evaluation,
controlled model promotion, and an explicit data-retention policy.
