FROM node:22.14.0-alpine AS web

# Deployable under a sub-path: pass --build-arg VITE_BASE_PATH=/embedding-list
ARG VITE_BASE_PATH=/
ENV VITE_BASE_PATH=${VITE_BASE_PATH}

WORKDIR /workspace/web
COPY demos/embedding-list-max-sim/web/package.json \
    demos/embedding-list-max-sim/web/package-lock.json ./
RUN npm ci
COPY demos/embedding-list-max-sim/web/index.html \
    demos/embedding-list-max-sim/web/vite.config.ts \
    demos/embedding-list-max-sim/web/tsconfig.json ./
COPY demos/embedding-list-max-sim/web/src src
RUN npm run build

FROM ghcr.io/astral-sh/uv:0.11.26 AS uv

FROM python:3.12-slim

COPY --from=uv /uv /uvx /bin/
WORKDIR /workspace

COPY demos/embedding-list-max-sim/backend/pyproject.toml \
    demos/embedding-list-max-sim/backend/uv.lock \
    demos/embedding-list-max-sim/backend/
COPY demos/embedding-list-max-sim/data/nasa-systems-engineering-handbook-rev2 \
    demos/embedding-list-max-sim/data/nasa-systems-engineering-handbook-rev2

WORKDIR /workspace/demos/embedding-list-max-sim/backend
COPY demos/embedding-list-max-sim/backend/src src
COPY --from=web /workspace/web/dist ../web/dist
RUN uv sync --frozen --no-dev

# Bake the pinned ColSmol adapter and base snapshots into the image so no
# external model mount is required at runtime.
ENV HF_HOME=/models
RUN .venv/bin/hf download vidore/colSmol-256M \
        --revision ff628001884d6bfb066658e86a32b5f233906bf5 \
    && .venv/bin/hf download vidore/ColSmolVLM-Instruct-256M-base \
        --revision 99ca96f1f6b95b3a69e6abef74a2416cb738fed0 \
    && mkdir -p /models/hub/models--vidore--colSmol-256M/refs \
    && printf 'ff628001884d6bfb066658e86a32b5f233906bf5' > /models/hub/models--vidore--colSmol-256M/refs/main \
    && mkdir -p /models/hub/models--vidore--ColSmolVLM-Instruct-256M-base/refs \
    && printf '99ca96f1f6b95b3a69e6abef74a2416cb738fed0' > /models/hub/models--vidore--ColSmolVLM-Instruct-256M-base/refs/main \
    && rm -rf /models/hub/.cache \
    && chmod -R a+rX /models

ENV PATH="/workspace/demos/embedding-list-max-sim/backend/.venv/bin:$PATH" \
    STATIC_DIR="/workspace/demos/embedding-list-max-sim/web/dist" \
    COLSMOL_DEVICE="cpu" \
    EMBEDDING_LIST_PORT="48040" \
    EMBEDDING_LIST_RUNTIME_DIR="/workspace/artifacts/runtime/embedding-list-max-sim/volume/run" \
    EMBEDDING_LIST_DATASET_DIR="/workspace/demos/embedding-list-max-sim/data/nasa-systems-engineering-handbook-rev2" \
    HF_HUB_CACHE="/models/hub" \
    HF_HUB_OFFLINE="1" \
    TRANSFORMERS_OFFLINE="1" \
    PYTHONDONTWRITEBYTECODE="1" \
    PYTHONUNBUFFERED="1" \
    UV_LINK_MODE="copy"

EXPOSE 48040
HEALTHCHECK --interval=10s --timeout=5s --start-period=360s --retries=3 \
    CMD ["python", "-c", "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"EMBEDDING_LIST_PORT\"]}/healthz/live', timeout=5)"]
CMD ["sh", "-c", "exec uvicorn embedding_list_demo.main:app --host 127.0.0.1 --port \"${EMBEDDING_LIST_PORT}\" --root-path \"${APP_ROOT_PATH:-}\""]
