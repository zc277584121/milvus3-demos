FROM ghcr.io/astral-sh/uv:0.11.26 AS uv

FROM python:3.12-slim

COPY --from=uv /uv /uvx /bin/
WORKDIR /workspace

COPY packages/python/milvus-demo-common packages/python/milvus-demo-common
COPY demos/structarray-search/backend/pyproject.toml \
    demos/structarray-search/backend/uv.lock \
    demos/structarray-search/backend/

WORKDIR /workspace/demos/structarray-search/backend
COPY demos/structarray-search/backend/src src
RUN uv sync --frozen --no-dev

ENV PATH="/workspace/demos/structarray-search/backend/.venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_CACHE=/model-cache/huggingface/hub \
    STRUCTARRAY_PORT=48030
EXPOSE 48030
HEALTHCHECK --interval=3s --timeout=4s --start-period=5s --retries=20 \
    CMD ["python", "-c", "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"STRUCTARRAY_PORT\"]}/healthz/live', timeout=4)"]
CMD ["sh", "-c", "exec uvicorn structarray_hybrid_demo.main:app --host 127.0.0.1 --port \"${STRUCTARRAY_PORT}\""]
