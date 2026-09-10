# EmbeddingList MAX_SIM NASA handbook retrieval

This demo retrieves a fixed 40-page selection from the **NASA Systems Engineering Handbook,
NASA/SP-2016-6105 Rev 2** with real ColSmol multi-vector embeddings and Milvus 3.0
`EmbeddingList` search. Page-patch vectors and query-token vectors are 128-dimensional. Milvus
executes `MAX_SIM_COSINE`; the ColPali local scorer independently checks the complete 40-page order
and every score within an absolute tolerance of `0.005`. Any disagreement fails closed. For the
Milvus Top 3, the app also computes a local spatial explanation from the same real query and page
multi-vectors; this explanation is not returned by Milvus.

```text
Browser /demos/embedding-list-max-sim
  -> Portal proxy
  -> isolated FastAPI backend on 127.0.0.1:48040
  -> CPU FP32 vidore/colSmol-256M
  -> Milvus 3.0.0 GA on 127.0.0.1:49530
  -> EmbeddingList + HNSW/MAX_SIM_COSINE
  -> page-level entities, order, and MAX_SIM scores
  -> application-side ColSmol query-token x spatial-patch explanation
```

## Fixed NTRS source contract

The tracked production dataset is
[`data/nasa-systems-engineering-handbook-rev2`](data/nasa-systems-engineering-handbook-rev2):

- Document: *NASA Systems Engineering Handbook*
- Identifier and revision: `NASA/SP-2016-6105 Rev 2`
- NTRS record: [`20170001761`](https://ntrs.nasa.gov/citations/20170001761)
- Distribution field: `PUBLIC`
- Rights determination field: `PUBLIC_USE_PERMITTED`
- Third-party-material field: `false`
- Authors: Steven R. Hirshorn; Linda D. Voss; Linda K. Bromley
- Source PDF SHA-256: `3153ae2e53e29452d5997efafe280a5f05cd21b43a047e988a17e1dd5207a38e`
- Rendering: PyMuPDF 1.28.2, RGB, 144 DPI, no annotations or page-content changes

The distribution and rights-determination values above are reproduced as structured NTRS metadata;
the demo does not infer a broader license category. Page images are unmodified renders from the
saved official PDF. NASA does not endorse Milvus or this demo.

`dataset-manifest.json` records exact PDF index, printed page, section, page title, visual type,
dimensions, renderer, source metadata, author affiliations, and SHA-256 for every selected page.
`SHA256SUMS` closes the complete dataset file set. `queries.json` contains the eight approved natural
questions plus internal target and same-PDF hard-negative mappings. Only query IDs and natural text
are exposed by the API; answer mappings are not returned to the Portal.

Validate the tracked bytes and reproduce all 40 page renders into a new, non-overwriting directory:

```bash
make embedding-list-max-sim-data-check
```

The builder refuses an existing destination and verifies the fixed PDF byte count, PDF SHA-256,
356-page source length, selected page order, page metadata, page dimensions, and complete file
hashes before accepting either the tracked or rebuilt tree.

## Canonical natural questions

1. How should systems engineers identify stakeholders, elicit their needs, validate expectations
   against operational scenarios, and baseline the result before defining technical requirements?
2. How are system-level requirements flowed down into allocated and derived subsystem requirements
   while maintaining traceability to their parent requirements?
3. What evidence should engineers collect to show both that a built product conforms to its
   specified requirements and that typical users can succeed with it in a realistic operational
   setting?
4. How should a team structure a technical risk so that initiating events, uncertain likelihood,
   and uncertain consequences can be analyzed together?
5. After someone proposes a baseline change, what sequence of review, board decision, release,
   implementation, and status accounting should control it?
6. When competing engineering alternatives rank closely, how can qualitative and quantitative
   analysis be combined to decide whether further uncertainty reduction is worthwhile?
7. Which diagram shows a spacecraft traveling from Earth orbit down to a lunar landing site along
   a drawn flight trajectory?
8. Which chart plots a curve that rises as cost increases against effectiveness, with labeled axes?

## CPU-only model contract

- Model: `vidore/colSmol-256M`
- Adapter revision: `ff628001884d6bfb066658e86a32b5f233906bf5`
- Base model: `vidore/ColSmolVLM-Instruct-256M-base`
- Base revision: `99ca96f1f6b95b3a69e6abef74a2416cb738fed0`
- Device: exactly `cpu`; `auto`, `cuda`, and GPU indices are rejected
- Inference dtype: exactly `torch.float32`
- Page inference: sequential, one rendered page per model call
- Dependency source: the explicit PyTorch CPU wheel index in `backend/pyproject.toml`

Both model snapshots are mounted read-only and loaded with `local_files_only=True`,
`HF_HUB_OFFLINE=1`, and `TRANSFORMERS_OFFLINE=1`. The lock contains CPU builds of PyTorch and
TorchVision and no GPU runtime packages. The demo never installs or configures a host GPU runtime.

## Cache, metadata, and timings

Only page embeddings are cached. Query embeddings and search results are always computed online.
The atomic safetensors cache key binds the dataset hashes, page identities, render contract, exact
model revisions, package versions, CPU device, FP32 inference dtype, and vector dimension.
Incomplete, modified, or mismatched cache state fails closed.

## Local ColSmol explanation contract

Milvus ranks page entities with `MAX_SIM_COSINE`; it does not return token matches, patch traces, or
heatmaps. After the complete Milvus/local ranking comparison passes, the FastAPI application uses
the already computed query and page vectors to explain each of the Milvus Top 3:

- visible non-stopword query concepts map to their exact ColSmol query-vector rows;
- the ten query-augmentation special tokens are excluded from the visible explanation;
- ColPali Engine's local-image mask excludes the non-spatial global image patch;
- Idefics3 subpatch token order is rearranged into a deterministic 32 x 32 normalized page grid;
- subword similarities are combined by maximum, then visible concepts are combined by maximum;
- patch intensity is normalized deterministically between the page median and page p98.

Every search response carries the concept labels and token indices, all 1,024 spatial patch records,
model sequence indices, normalized page coordinates, raw cosine similarity, intensity, and the full
aggregation description. Search results also carry the source PNG width and height. The UI preserves
that per-page aspect ratio and overlays the normalized 32 x 32 explanation on the exact same image
canvas, including landscape pages. Editing or switching the query immediately removes the old
ranking, selected page, and heatmap.

The dedicated Collection is `milvus3_demos_nasa_seh_colsmol_pages`; its index is
`milvus3_demos_nasa_seh_patch_embeddings`. Each row stores the PDF index, printed page, title,
section, document identifier, NTRS ID and record URL, distribution, rights determination,
third-party-material flag, source/page hashes, and `patches[patch_embedding]`.

Prepare responses expose dataset verification, cache, page inference, and Milvus timings. Search
responses expose query inference, Milvus search, local scoring, local explanation, and total
milliseconds. Cold/warm timings are observations from the current machine, not static performance
claims. MAX_SIM is a similarity score, not a probability.

## Safe local validation

These checks do not require the model, a container, or Milvus:

```bash
make embedding-list-max-sim-data-check
UV_OFFLINE=1 uv run --offline --project demos/embedding-list-max-sim/backend --group dev \
  pytest demos/embedding-list-max-sim/backend/tests -m "not integration"
npm test --workspace @milvus3-demos/embedding-list-max-sim
npm test --workspace @milvus3-demos/portal
npm run typecheck
```

## Deferred real lifecycle

The following lifecycle commands intentionally build an isolated CPU image, load the real model,
use only the project GA endpoint at `127.0.0.1:49530`, and then prove exact cleanup. They are not
part of the safe validation above:

```bash
make embedding-list-max-sim-image-check
make embedding-list-max-sim-e2e
```

The real lifecycle covers cold and warm preparation, all eight natural queries, browser screenshots
and traces at the required viewports, CPU and memory evidence, latency reports, image inspection,
raw SDK audits, and exact cleanup. It also submits a 513-character query through the real Portal and
FastAPI path, requires the API's 422 rejection and cleared stale evidence, then retries a canonical
query and requires real Milvus ranking plus the local explanation to recover. The four pre-existing
Milvus deployments are captured before and after and must remain unchanged; only the isolated NASA
demo Collection on the project GA endpoint is in scope.

Lifecycle evidence defaults below `artifacts/validation/S-20260826-002/`. Evidence directories are
unique and non-overwriting and receive a semantic SHA-256 manifest.
