FROM node:22.14.0-alpine AS web

# Deployable under a sub-path: pass --build-arg VITE_BASE_PATH=/function-chain-rerank
ARG VITE_BASE_PATH=/
ENV VITE_BASE_PATH=${VITE_BASE_PATH}

WORKDIR /workspace/web
COPY demos/function-chain-rerank/web/package.json \
    demos/function-chain-rerank/web/package-lock.json ./
RUN npm ci
COPY demos/function-chain-rerank/web/index.html \
    demos/function-chain-rerank/web/vite.config.ts \
    demos/function-chain-rerank/web/tsconfig.json ./
COPY demos/function-chain-rerank/web/src src
RUN npm run build

FROM ghcr.io/astral-sh/uv:0.11.26 AS uv

FROM python:3.12-slim

COPY --from=uv /uv /uvx /bin/
WORKDIR /workspace

RUN groupadd --gid 10001 app && useradd --uid 10001 --gid app --no-create-home app

COPY demos/function-chain-rerank/backend/pyproject.toml \
    demos/function-chain-rerank/backend/uv.lock \
    demos/function-chain-rerank/backend/

WORKDIR /workspace/demos/function-chain-rerank/backend
COPY demos/function-chain-rerank/backend/src src
COPY demos/function-chain-rerank/data ../data
COPY --from=web /workspace/web/dist ../web/dist
RUN uv sync --frozen --no-dev

ENV PATH="/workspace/demos/function-chain-rerank/backend/.venv/bin:$PATH" \
    STATIC_DIR="/workspace/demos/function-chain-rerank/web/dist" \
    HF_HUB_CACHE="/models/hub" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
USER 10001:10001
EXPOSE 8000
HEALTHCHECK --interval=3s --timeout=4s --start-period=5s --retries=20 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz/ready', timeout=4)"]
CMD ["sh", "-c", "exec uvicorn function_chain_demo.main:app --host 0.0.0.0 --port 8000 --root-path \"${APP_ROOT_PATH:-}\""]
