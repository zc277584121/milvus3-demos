FROM ghcr.io/astral-sh/uv:0.11.26 AS uv

FROM python:3.12-slim

COPY --from=uv /uv /uvx /bin/
WORKDIR /workspace

RUN groupadd --gid 10001 app && useradd --uid 10001 --gid app --no-create-home app

COPY packages/python/milvus-demo-common packages/python/milvus-demo-common
COPY demos/function-chain-rerank/backend/pyproject.toml \
    demos/function-chain-rerank/backend/uv.lock \
    demos/function-chain-rerank/backend/

WORKDIR /workspace/demos/function-chain-rerank/backend
COPY demos/function-chain-rerank/backend/src src
COPY demos/function-chain-rerank/data ../data
RUN uv sync --frozen --no-dev

ENV PATH="/workspace/demos/function-chain-rerank/backend/.venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
USER 10001:10001
EXPOSE 8000
HEALTHCHECK --interval=3s --timeout=4s --start-period=5s --retries=20 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz/ready', timeout=4)"]
CMD ["uvicorn", "function_chain_demo.main:app", "--host", "0.0.0.0", "--port", "8000"]
