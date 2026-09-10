FROM ghcr.io/astral-sh/uv:0.11.26 AS uv

FROM python:3.12-slim

COPY --from=uv /uv /uvx /bin/
WORKDIR /workspace

COPY packages/python/milvus-demo-common packages/python/milvus-demo-common
COPY demos/embedding-list-max-sim/backend/pyproject.toml \
    demos/embedding-list-max-sim/backend/uv.lock \
    demos/embedding-list-max-sim/backend/
COPY demos/embedding-list-max-sim/data/nasa-systems-engineering-handbook-rev2 \
    demos/embedding-list-max-sim/data/nasa-systems-engineering-handbook-rev2

WORKDIR /workspace/demos/embedding-list-max-sim/backend
COPY demos/embedding-list-max-sim/backend/src src
RUN uv sync --frozen --no-dev

ENV PATH="/workspace/demos/embedding-list-max-sim/backend/.venv/bin:$PATH" \
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
HEALTHCHECK --interval=3s --timeout=4s --start-period=5s --retries=20 \
    CMD ["python", "-c", "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"EMBEDDING_LIST_PORT\"]}/healthz/live', timeout=4)"]
CMD ["sh", "-c", "exec uvicorn embedding_list_demo.main:app --host 127.0.0.1 --port \"${EMBEDDING_LIST_PORT}\""]
