FROM node:22.14.0-alpine AS web

# Deployable under a sub-path: pass --build-arg VITE_BASE_PATH=/structarray-search
ARG VITE_BASE_PATH=/
ENV VITE_BASE_PATH=${VITE_BASE_PATH}

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

RUN groupadd --gid 10001 app && useradd --uid 10001 --gid app --no-create-home app

COPY --from=uv /uv /uvx /bin/
WORKDIR /workspace

COPY demos/structarray-search/backend/pyproject.toml \
    demos/structarray-search/backend/uv.lock \
    demos/structarray-search/backend/

WORKDIR /workspace/demos/structarray-search/backend
COPY demos/structarray-search/backend/src src
COPY demos/structarray-search/data ../data
COPY --from=web /workspace/web/dist ../web/dist
RUN uv sync --frozen --no-dev

# Bake the pinned BGE-M3 ONNX snapshot into the image so no external model
# mount is required at runtime.
ENV HF_HOME=/models
RUN .venv/bin/hf download gpahal/bge-m3-onnx-int8 \
        --revision 2b34e84df040034d4b9eabb62383a87c18955822 \
    && rm -rf /models/hub/.cache \
    && chmod -R a+rX /models

ENV PATH="/workspace/demos/structarray-search/backend/.venv/bin:$PATH" \
    STATIC_DIR="/workspace/demos/structarray-search/web/dist" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_CACHE=/models/hub \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    STRUCTARRAY_PORT=48030
USER 10001:10001
EXPOSE 48030
HEALTHCHECK --interval=10s --timeout=5s --start-period=120s --retries=3 \
    CMD ["python", "-c", "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"STRUCTARRAY_PORT\"]}/healthz/live', timeout=5)"]
CMD ["sh", "-c", "exec uvicorn structarray_hybrid_demo.main:app --host 0.0.0.0 --port \"${STRUCTARRAY_PORT}\" --root-path \"${APP_ROOT_PATH:-}\""]
