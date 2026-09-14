"""FastAPI boundary for real ColSmol EmbeddingList retrieval."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Response, status
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from embedding_list_demo.api_models import (
    CleanupResponse,
    LiveHealthResponse,
    ManualResponse,
    PrepareResponse,
    ReadyHealthResponse,
    SearchRequest,
    SearchResponse,
    StatusResponse,
)
from embedding_list_demo.cache import CacheContractError
from embedding_list_demo.config import ConfigurationError
from embedding_list_demo.manual import ManualContractError
from embedding_list_demo.model import ModelContractError
from embedding_list_demo.repository import RepositoryContractError
from embedding_list_demo.service import (
    EmbeddingListService,
    ScoreComparisonError,
    ServiceContractError,
)

CONTRACT_ERRORS = (
    CacheContractError,
    ConfigurationError,
    ManualContractError,
    ModelContractError,
    RepositoryContractError,
    ServiceContractError,
)


def create_app(*, service: EmbeddingListService | None = None) -> FastAPI:
    """Create an injectable FastAPI application.

    The real (uninjected) service prepares itself on startup so the packaged
    image is search-ready immediately; injected test doubles keep the deferred
    prepare() contract.
    """

    resolved = service or EmbeddingListService()
    # Auto-prepare only the real, uninjected service so the packaged image is
    # search-ready on startup. It reuses the disk embedding cache and adopts a
    # pre-existing Collection, so restarts stay fast. Test doubles skip this
    # (they enter lifespan only under an explicit context manager).
    auto_prepare = service is None

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        if auto_prepare:
            resolved.prepare()
        yield

    application = FastAPI(
        title="ColSmol EmbeddingList MAX_SIM Demo",
        version="0.2.0",
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
        lifespan=lifespan,
    )

    @application.get("/healthz/live", response_model=LiveHealthResponse)
    def live_health() -> LiveHealthResponse:
        return LiveHealthResponse()

    @application.get("/healthz/ready", response_model=ReadyHealthResponse)
    def ready_health(response: Response) -> ReadyHealthResponse:
        try:
            current = resolved.status()
            milvus = current["milvus"]
            model = current["model"]
            ready = (
                current["status"] == "ready"
                and milvus["server_version"] == "3.0.0"
                and milvus["target_exists"] is True
                and model["cpu_only"] is True
                and model["device_type"] == "cpu"
                and model["inference_dtype"] == "float32"
            )
            if not ready:
                response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
            return ReadyHealthResponse(
                status="healthy" if ready else "unhealthy",
                model_cache_available=True,
                device_type=str(model["device_type"]),
                inference_dtype=str(model["inference_dtype"]),
                cpu_only=bool(model["cpu_only"]),
                collection_exists=bool(milvus["target_exists"]),
                server_version=str(milvus["server_version"]),
            )
        except Exception:
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
            return ReadyHealthResponse(
                status="unhealthy",
                model_cache_available=False,
                device_type="cpu",
                inference_dtype="float32",
                cpu_only=True,
                collection_exists=False,
                server_version=None,
            )

    @application.get("/api/v1/status", response_model=StatusResponse)
    def demo_status() -> StatusResponse:
        try:
            return StatusResponse.model_validate(resolved.status())
        except CONTRACT_ERRORS as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @application.get("/api/v1/manual", response_model=ManualResponse)
    def manual_manifest() -> ManualResponse:
        try:
            return ManualResponse.model_validate(resolved.manual())
        except CONTRACT_ERRORS as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @application.post("/api/v1/prepare", response_model=PrepareResponse)
    def prepare() -> PrepareResponse:
        try:
            return PrepareResponse.model_validate(resolved.prepare())
        except CONTRACT_ERRORS as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @application.post("/api/v1/search", response_model=SearchResponse)
    def search(request: SearchRequest) -> SearchResponse:
        try:
            return SearchResponse.model_validate(resolved.search(request.query))
        except ScoreComparisonError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except ServiceContractError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except CONTRACT_ERRORS as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @application.get("/api/v1/pages/{page_id}", response_class=FileResponse)
    def page_image(page_id: str) -> FileResponse:
        try:
            path = resolved.page_image(page_id)
        except (ManualContractError, ServiceContractError, OSError) as exc:
            raise HTTPException(status_code=404, detail="NASA handbook page not found") from exc
        return FileResponse(path, media_type="image/png", filename=path.name)

    @application.post("/api/v1/cleanup", response_model=CleanupResponse)
    def cleanup() -> CleanupResponse:
        try:
            return CleanupResponse.model_validate(resolved.cleanup())
        except CONTRACT_ERRORS as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    static_dir = os.environ.get("STATIC_DIR")
    if static_dir and Path(static_dir).is_dir():
        application.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

    return application


app = create_app()
