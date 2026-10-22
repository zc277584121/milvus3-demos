FROM node:22.14.0-alpine AS web

# Deployable under a sub-path: pass --build-arg VITE_BASE_PATH=/minhash-near-duplicate
ARG VITE_BASE_PATH=/
ENV VITE_BASE_PATH=${VITE_BASE_PATH}

WORKDIR /workspace/web
COPY demos/minhash-near-duplicate/web/package.json \
    demos/minhash-near-duplicate/web/package-lock.json ./
RUN npm ci
COPY demos/minhash-near-duplicate/web/index.html \
    demos/minhash-near-duplicate/web/vite.config.ts \
    demos/minhash-near-duplicate/web/tsconfig.json ./
COPY demos/minhash-near-duplicate/web/src src
RUN npm run build

FROM ghcr.io/astral-sh/uv:0.11.26 AS uv

FROM python:3.12-slim

RUN groupadd --gid 10001 app && useradd --uid 10001 --gid app --no-create-home app

COPY --from=uv /uv /uvx /bin/
WORKDIR /workspace

COPY demos/minhash-near-duplicate/backend/pyproject.toml \
    demos/minhash-near-duplicate/backend/uv.lock \
    demos/minhash-near-duplicate/backend/

WORKDIR /workspace/demos/minhash-near-duplicate/backend
COPY demos/minhash-near-duplicate/backend/src src
COPY demos/minhash-near-duplicate/data ../data
RUN uv sync --frozen --no-dev

COPY --from=web /workspace/web/dist ../web/dist

ENV PATH="/workspace/demos/minhash-near-duplicate/backend/.venv/bin:$PATH" \
    STATIC_DIR="/workspace/demos/minhash-near-duplicate/web/dist" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MINHASH_PORT=48050
USER 10001:10001
EXPOSE 48050
HEALTHCHECK --interval=10s --timeout=5s --start-period=60s --retries=3 \
    CMD ["python", "-c", "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"MINHASH_PORT\"]}/healthz/live', timeout=5)"]
CMD ["sh", "-c", "exec uvicorn minhash_near_dup_demo.main:app --host 0.0.0.0 --port \"${MINHASH_PORT}\" --root-path \"${APP_ROOT_PATH:-}\""]
