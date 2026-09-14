"""FastAPI boundary for the StructArray parent + child hybrid demo."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Response, status
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from structarray_hybrid_demo.api_models import (
    CleanupResponse,
    LiveHealthResponse,
    PrepareResponse,
    ReadyHealthResponse,
    SearchRequest,
    SearchResponse,
    StatusResponse,
)
from structarray_hybrid_demo.config import VECTOR_DIMENSION, ConfigurationError
from structarray_hybrid_demo.data import DataContractError
from structarray_hybrid_demo.embedding import ModelContractError
from structarray_hybrid_demo.evidence import EvidencePathError, EvidenceResolver
from structarray_hybrid_demo.repository import RepositoryContractError, SearchContractError
from structarray_hybrid_demo.service import ServiceContractError, StructArrayHybridService

CONTRACT_ERRORS = (
    ConfigurationError,
    DataContractError,
    ModelContractError,
    RepositoryContractError,
    SearchContractError,
    ServiceContractError,
)


def create_app(*, service: StructArrayHybridService | None = None) -> FastAPI:
    resolved = service or StructArrayHybridService()
    # Auto-prepare only the real, uninjected service (startup-ready image).
    # Test doubles skip this because they only enter lifespan when wrapped in
    # a context manager, and the double has no prepare() to run.
    auto_prepare = service is None

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        if auto_prepare:
            # Rebuild from scratch so a stale collection left by a previous
            # container never blocks a fresh, search-ready start.
            resolved.cleanup()
            resolved.prepare()
        yield

    application = FastAPI(
        title="StructArray Parent + Child Semantic Hybrid Demo",
        version="3.0.0",
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
                and milvus["collection_exists"] is True
                and model["device"] == "cpu"
                and model["dense_output_name"] == "dense_vecs"
            )
            if not ready:
                response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
            return ReadyHealthResponse(
                status="healthy" if ready else "unhealthy",
                model_cache_available=bool(model["cache_available"]),
                device_type="cpu",
                inference_dtype="float32",
                cpu_only=True,
                collection_exists=bool(milvus["collection_exists"]),
                server_version=str(milvus["server_version"]),
                vector_dimension=int(model["vector_dimension"]),
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
                vector_dimension=VECTOR_DIMENSION,
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
                    limit=request.limit,
                    parent_weight=request.parent_weight,
                    collapse_strategy=request.collapse_strategy,
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

    @application.get("/api/v1/evidence/{kind}/{file_name}", response_class=FileResponse)
    def evidence_frame(kind: str, file_name: str) -> FileResponse:
        resolver = EvidenceResolver(
            config=resolved.config,
            allowed_names=resolved.evidence_frame_names(),
        )
        try:
            path = resolver.resolve(kind=kind, file_name=file_name)
        except (EvidencePathError, OSError) as exc:
            raise HTTPException(status_code=404, detail="Evidence frame not found") from exc
        return FileResponse(path, media_type="image/jpeg", filename=path.name)

    static_dir = os.environ.get("STATIC_DIR")
    if static_dir and Path(static_dir).is_dir():
        application.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

    return application


app = create_app()
