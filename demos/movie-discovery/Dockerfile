FROM node:22.14.0-alpine AS web

# Deployable under a sub-path: pass --build-arg VITE_BASE_PATH=/movie-discovery
ARG VITE_BASE_PATH=/
ENV VITE_BASE_PATH=${VITE_BASE_PATH}

WORKDIR /workspace/web
COPY demos/movie-discovery/web/package.json \
    demos/movie-discovery/web/package-lock.json ./
RUN npm ci
COPY demos/movie-discovery/web/index.html \
    demos/movie-discovery/web/vite.config.ts \
    demos/movie-discovery/web/tsconfig.json ./
COPY demos/movie-discovery/web/src src
RUN npm run build

FROM ghcr.io/astral-sh/uv:0.11.26 AS uv

FROM python:3.12-slim

RUN groupadd --gid 10001 app && useradd --uid 10001 --gid app --no-create-home app

COPY --from=uv /uv /uvx /bin/
WORKDIR /workspace

# Optional: pass --build-arg PYPI_INDEX=<mirror> to resolve Python deps from a
# mirror when the default registry is unreachable. Defaults to the official
# PyPI so production builds keep the same behavior as the other demos.
ARG PYPI_INDEX=
ENV UV_DEFAULT_INDEX=${PYPI_INDEX}

COPY demos/movie-discovery/backend/pyproject.toml \
    demos/movie-discovery/backend/uv.lock \
    demos/movie-discovery/backend/

WORKDIR /workspace/demos/movie-discovery/backend
COPY demos/movie-discovery/backend/src src
COPY demos/movie-discovery/data ../data
RUN uv sync --frozen --no-dev

# Bake the pinned BGE-M3 ONNX snapshot into the image so no external model
# mount is required at runtime. Kept BEFORE the frontend dist COPY so
# frontend-only changes reuse this cached layer instead of re-downloading the
# model.
ENV HF_HOME=/models
RUN .venv/bin/hf download gpahal/bge-m3-onnx-int8 \
        --revision 2b34e84df040034d4b9eabb62383a87c18955822 \
    && rm -rf /models/hub/.cache \
    && chmod -R a+rX /models

COPY --from=web /workspace/web/dist ../web/dist

ENV PATH="/workspace/demos/movie-discovery/backend/.venv/bin:$PATH" \
    STATIC_DIR="/workspace/demos/movie-discovery/web/dist" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_CACHE=/models/hub \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    MOVIE_PORT=48060
USER 10001:10001
EXPOSE 48060
HEALTHCHECK --interval=10s --timeout=5s --start-period=180s --retries=3 \
    CMD ["python", "-c", "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"MOVIE_PORT\"]}/healthz/live', timeout=5)"]
CMD ["sh", "-c", "exec uvicorn movie_discovery_demo.main:app --host 0.0.0.0 --port \"${MOVIE_PORT}\" --root-path \"${APP_ROOT_PATH:-}\""]
