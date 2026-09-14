FROM node:22.14.0-alpine AS web

WORKDIR /workspace/web
COPY demos/structarray-search/web/package.json \
    demos/structarray-search/web/package-lock.json ./
RUN npm ci
COPY demos/structarray-search/web/index.html \
    demos/structarray-search/web/vite.config.ts \
    demos/structarray-search/web/tsconfig.json ./
COPY demos/structarray-search/web/src src
RUN npm run build

FROM ghcr.io/astral-sh/uv:0.11.26 AS uv

FROM python:3.12-slim

COPY --from=uv /uv /uvx /bin/
WORKDIR /workspace

COPY demos/structarray-search/backend/pyproject.toml \
    demos/structarray-search/backend/uv.lock \
    demos/structarray-search/backend/

WORKDIR /workspace/demos/structarray-search/backend
COPY demos/structarray-search/backend/src src
COPY --from=web /workspace/web/dist ../web/dist
RUN uv sync --frozen --no-dev

ENV PATH="/workspace/demos/structarray-search/backend/.venv/bin:$PATH" \
    STATIC_DIR="/workspace/demos/structarray-search/web/dist" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_CACHE=/model-cache/huggingface/hub \
    STRUCTARRAY_PORT=48030
EXPOSE 48030
HEALTHCHECK --interval=10s --timeout=5s --start-period=120s --retries=3 \
    CMD ["python", "-c", "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"STRUCTARRAY_PORT\"]}/healthz/live', timeout=5)"]
CMD ["sh", "-c", "exec uvicorn structarray_hybrid_demo.main:app --host 127.0.0.1 --port \"${STRUCTARRAY_PORT}\""]
