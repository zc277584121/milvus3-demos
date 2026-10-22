# Movie Discovery with Search Aggregation

This demo turns a natural-language movie description into an explorable set of facets with
Milvus 3.0's **Search Aggregation**. You describe the story you want, Milvus embeds the query with
a CPU-only BGE-M3 ONNX model and retrieves related movies by cosine similarity, and the app groups
those hits by `Genre`, `Decade`, or `Region` — with a plain list view alongside for comparison. No
external embedding API and no precomputed companion-vector file are used.

```text
Browser /
  -> unified FastAPI static+API server on 127.0.0.1:48060
  -> BGE-M3 ONNX (CPU) query embedding (1024 dims)
  -> Milvus 3.0.0 GA on 127.0.0.1:49530
  -> grouped path: search(..., search_aggregation=SearchAggregation(...))
  -> list path:   plain ANN search over the same synopsis vector field
```

## Capability contract

- **Schema.** `movie_id` (VARCHAR primary key), `title`, `synopsis`, `primary_genre`,
  `release_year`, `decade`, `region`, `runtime_minutes`, `poster_path` (all scalar, single-value),
  and `synopsis_vector` (`FLOAT_VECTOR`, dim `1024`).
- **Embedding.** The query and every synopsis are embedded at request/prepare time by the pinned
  quantized `gpahal/bge-m3-onnx-int8` snapshot (`revision
  2b34e84df040034d4b9eabb62383a87c18955822`), L2-normalized, CPU only. The dense output is
  `dense_vecs`.
- **Index.** `HNSW` / `COSINE` on `synopsis_vector` (`M=16`, `efConstruction=128`), search `ef=128`.
- **Search.** Two paths over the same vector field:
  - `list` — plain ANN search returning a ranked hit list;
  - `grouped` — `SearchAggregation` over a single-value facet field with a `TopHits(size=3)`
    representative set per bucket.

The collection is `milvus3_demos_movie_catalog`; its index is
`milvus3_demos_movie_synopsis_vector`.

## Search Aggregation semantics

The demo follows the Milvus 3.0 aggregation contract exactly:

- Aggregation results arrive on `result.agg_buckets`; when aggregation is enabled the plain hit
  list is empty by design, so the `grouped` and `list` responses come from two separate requests.
- `count` and the representative `candidates` cover only the candidates retained during retrieval,
  **not** the whole 90-movie catalog and not every semantically related movie. The page footer
  states this explicitly, and the response `semantics` field spells it out.
- Each full group key's candidate budget is set by the largest `TopHits.size` (`3`); changing that
  budget can change counts, statistics, and group ordering.
- Facet fields are single-value scalars (`primary_genre`, `decade`, `region`) — the demo never
  relies on `ARRAY<...>` multi-label grouping.
- Hybrid Search is **not** combined with Search Aggregation here; that would need a fresh
  capability check before extending.

## Fixed synthetic dataset contract

The tracked production dataset is
[`data/movie-catalog-r1`](data/movie-catalog-r1). It contains **90 original, fictional movies**
authored by the milvus3-demos project — 15 each across six genres (`Adventure`, `Animation`,
`Comedy`, `Drama`, `Sci-Fi`, `Thriller`), spanning five decades and ten regions — plus five preset
queries. Every title, synopsis, year, and region is synthetic; posters are generated placeholder
gradients (the `poster_path` references a `posters/*.jpg` that the UI renders as a color palette,
not a real image), so the page never implies these films are real or endorsed by any studio.

`movies.json` holds the movies, `presets.json` holds the five preset queries, and `manifest.json`
records the counts, embedding contract, and provenance. All metadata and text are dedicated to the
public domain under CC0 1.0 (`licenses/LICENSE.txt`); see `NOTICE.md` and `LICENSE-CONFLICT.md`.

Validate the tracked bytes and dataset contract offline:

```bash
make movie-discovery-data-check
```

## API

| Endpoint | Method | Purpose |
| --- | --- | --- |
| `/healthz/live` | GET | Liveness. |
| `/healthz/ready` | GET | Readiness: prepared collection on Milvus 3.0.0. |
| `/api/v1/status` | GET | Dataset, presets, model, and Milvus audit. |
| `/api/v1/prepare` | POST | Embed the catalog, create and load the collection from scratch. |
| `/api/v1/search` | POST | Faceted search. Body: `query`, `view` (`grouped` \| `list`), `facet` (`genre` \| `decade` \| `region`), `limit`, `max_runtime`. |
| `/api/v1/cleanup` | POST | Drop only the exact demo collection. |

`/api/v1/search` returns `grouped.buckets` (with `key`, `count`, and `candidates`) on the grouped
path, or `list.results` on the list path. Scores are **COSINE similarity, not a probability**.

## Score semantics and honest limits

- Scores are COSINE similarity between BGE-M3 query and synopsis vectors, never probabilities or
  model confidence.
- Group `count` and candidate lists cover only retained retrieval candidates — not the full
  catalog. The response `semantics` field states this explicitly.
- `poster_placeholders` is always `true`: posters are generated gradient placeholders, not real
  artwork, and the page labels them as such.

## Safe local validation

These checks do not require Milvus or a container:

```bash
make movie-discovery-data-check
UV_OFFLINE=1 uv run --offline --project demos/movie-discovery/backend --group dev \
  pytest demos/movie-discovery/backend/tests
npm test --workspace @milvus3-demos/movie-discovery
npm run typecheck
```

## Real lifecycle

The following command builds an isolated image and proves the backend becomes search-ready
against the project GA endpoint at `127.0.0.1:49530`, then cleans up exactly:

```bash
make movie-discovery-image-check
```
