"""FastAPI boundary for the MinHash + LSH near-duplicate demo."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Response, status
from fastapi.staticfiles import StaticFiles

from minhash_near_dup_demo.api_models import (
    CleanupResponse,
    LiveHealthResponse,
    PrepareResponse,
    ReadyHealthResponse,
    SearchRequest,
    SearchResponse,
    StatusResponse,
)
from minhash_near_dup_demo.config import ConfigurationError
from minhash_near_dup_demo.data import DataContractError
from minhash_near_dup_demo.repository import RepositoryContractError, SearchContractError
from minhash_near_dup_demo.service import MinHashService, ServiceContractError

CONTRACT_ERRORS = (
    ConfigurationError,
    DataContractError,
    RepositoryContractError,
    SearchContractError,
    ServiceContractError,
)


def create_app(*, service: MinHashService | None = None) -> FastAPI:
    """Create an injectable FastAPI application.

    The real (uninjected) service adopts a pre-existing exact Collection on
    startup or prepares it from scratch, so the packaged image is search-ready
    immediately. Test doubles keep the deferred lifecycle contract.
    """

    resolved = service or MinHashService()
    auto_prepare = service is None
    root_path = os.environ.get("APP_ROOT_PATH", "").rstrip("/")

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        if auto_prepare:
            if not resolved.adopt():
                resolved.prepare()
        yield

    application = FastAPI(
        title="MinHash + LSH Near-Duplicate Detection Demo",
        version="3.0.0",
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
        root_path=root_path,
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
            ready = (
                current["status"] == "ready"
                and milvus["server_version"] == "3.0.0"
                and milvus["target_exists"] is True
            )
            if not ready:
                response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
            return ReadyHealthResponse(
                status="healthy" if ready else "unhealthy",
                collection_exists=bool(milvus["target_exists"]),
                server_version=str(milvus["server_version"]),
            )
        except Exception:
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
            return ReadyHealthResponse(
                status="unhealthy",
                collection_exists=False,
                server_version=None,
            )

    @application.get("/api/v1/status", response_model=StatusResponse)
    def demo_status() -> StatusResponse:
        try:
            return StatusResponse.model_validate(resolved.status())
        except CONTRACT_ERRORS as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @application.post("/api/v1/prepare", response_model=PrepareResponse)
    def prepare() -> PrepareResponse:
        try:
            return PrepareResponse.model_validate(resolved.prepare())
        except RepositoryContractError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except CONTRACT_ERRORS as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @application.post("/api/v1/search", response_model=SearchResponse)
    def search(request: SearchRequest) -> SearchResponse:
        try:
            return SearchResponse.model_validate(
                resolved.search(
                    query=request.query,
                    threshold=request.threshold,
                    limit=request.limit,
                    selected_doc_id=request.selected_doc_id,
                ).public_dict()
            )
        except SearchContractError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except CONTRACT_ERRORS as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

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
