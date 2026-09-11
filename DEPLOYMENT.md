# Deployment Guide

This repository ships three independent, self-contained Milvus 3.0 demos. Each demo is a single
image that serves both its compiled front-end and its `/api/v1/` API from one port, so every demo
is built, published, and run on its own — there is no shared package, portal, or gateway between
them.

| Demo | Backend port | Image name (local) | Notable runtime dependency |
| --- | --- | --- | --- |
| `function-chain-rerank` | `48020` | `milvus3-demos/function-chain-rerank` | Milvus 3.0.0 + MinIO (XGBoost UBJ FileResource) |
| `embedding-list-max-sim` | `48040` | `milvus3-demos/embedding-list-max-sim` | Milvus 3.0.0 + local ColSmol model cache |
| `structarray-search` | `48030` | `milvus3-demos/structarray-search` | Milvus 3.0.0 + gated CoVLA data + BGE-M3 ONNX cache |

All three bind to loopback only. Demos 2 and 3 use `network_mode: host`; Demo 1 uses a bridge
network plus the external `milvus3-demos-ga` network.

## Prerequisites

- Docker 27+ and Docker Compose v2
- `uv` 0.11+ and Node.js 22 (development only — not needed to build images)
- A running Milvus 3.0.0 standalone (see below)

## 1. Start Milvus 3.0.0

The demos depend on a project-local Milvus 3.0.0 GA stack (etcd + MinIO + standalone). It is a
separate Compose project and is not started or stopped by the demo commands.

```bash
export MILVUS_RUNTIME_DIR=/path/to/writable/runtime/dir
docker compose -f infra/milvus/docker-compose.yml up -d
```

This exposes:

```text
Milvus:  http://127.0.0.1:49530
WebUI:   http://127.0.0.1:49091/webui/
```

`MILVUS_RUNTIME_DIR` is a required writable directory where etcd/MinIO/Milvus persist their data.

## 2. Build an image

Build from the repository root. Each Dockerfile copies only its own demo's `web/`, `backend/`, and
(where applicable) `data/`, so the images are fully independent.

```bash
# Demo 1 — Function Chain Rerank
docker build -f demos/function-chain-rerank/Dockerfile -t milvus3-demos/function-chain-rerank:local .

# Demo 2 — EmbeddingList MAX_SIM
docker build -f demos/embedding-list-max-sim/Dockerfile -t milvus3-demos/embedding-list-max-sim:local .

# Demo 3 — StructArray Search
docker build -f demos/structarray-search/Dockerfile -t milvus3-demos/structarray-search:local .
```

Each build runs a Node stage (`npm ci` + `vite build`) and a Python stage (`uv sync --frozen
--no-dev`), then copies the compiled `dist/` into the Python stage. `STATIC_DIR` tells the FastAPI
app to mount the front-end at `/` while `/api/v1/*` and `/healthz/*` take priority.

## 3. Run a demo

### Demo 1 — Function Chain Rerank

Requires a local Hugging Face hub cache (for the BGE-M3 ONNX query encoder). On startup the app
trains the XGBoost model in-container, uploads it to MinIO as a FileResource, and creates the
collection — so Milvus and MinIO must both be up.

```bash
export HF_HUB_CACHE_HOST=/path/to/local/huggingface/hub
docker compose -f infra/function-chain-rerank/docker-compose.yml up -d --build
curl -fsS http://127.0.0.1:48020/healthz/ready
```

Open `http://127.0.0.1:48020/` for the UI.

### Demo 2 — EmbeddingList MAX_SIM

Requires a local ColSmol model cache and uses `network_mode: host`.

```bash
export HF_HUB_CACHE_HOST=/path/to/local/huggingface/hub
docker compose -f infra/embedding-list-max-sim/docker-compose.yml up -d --build
curl -fsS http://127.0.0.1:48040/healthz/live
```

Open `http://127.0.0.1:48040/` for the UI.

### Demo 3 — StructArray Search

Requires the gated CoVLA dataset and a local BGE-M3 ONNX cache; uses `network_mode: host`. The
CoVLA source data is never committed — supply it out-of-tree.

```bash
export COVLA_DATA_DIR=/path/to/covla-dataset
export HF_HUB_CACHE_HOST=/path/to/local/huggingface/hub
docker compose -f infra/structarray-search/docker-compose.yml up -d --build
curl -fsS http://127.0.0.1:48030/healthz/live
```

Open `http://127.0.0.1:48030/` for the UI.

## Required environment variables

Compose files fail fast on missing required paths (no private host defaults are baked in).

| Variable | Used by | Meaning |
| --- | --- | --- |
| `MILVUS_RUNTIME_DIR` | Milvus stack | Writable dir for etcd/MinIO/Milvus data |
| `HF_HUB_CACHE_HOST` | Demos 1, 2, 3 | Host Hugging Face hub cache (mounted read-only) |
| `COVLA_DATA_DIR` | Demo 3 | Approved CoVLA dataset root (mounted read-only) |

Optional overrides: `FUNCTION_CHAIN_PORT` (default `48020`), `STRUCTARRAY_PORT` (default `48030`),
`EMBEDDING_LIST_PORT` (default `48040`). See `infra/*/docker-compose.yml` for the full set.

## Verification targets

Each demo has a `make` target that builds the image, starts it, hits its health endpoint, and
cleans up:

```bash
make function-chain-rerank-image-check    # needs HF_HUB_CACHE_HOST
make embedding-list-max-sim-image-check   # needs HF_HUB_CACHE_HOST
make structarray-search-image-check       # needs COVLA_DATA_DIR + HF_HUB_CACHE_HOST
```

Fast validation without containers:

```bash
make validate
```

## Daily iteration

1. Edit code under `demos/<demo-id>/`.
2. Run that demo's checks: `make <demo-id>-data-check` and `make <demo-id>-image-check`.
3. Commit and push: `git commit ... && git push origin main`.
4. Rebuild and republish the affected image only — the other two are untouched.

Because each demo owns its own backend `uv.lock`, web `package-lock.json`, and Dockerfile, a change
to one demo cannot change the other two images.
