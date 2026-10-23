"""FastAPI boundary for the Movie Discovery demo."""

from __future__ import annotations

import os
import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Response, status
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from movie_discovery_demo.api_models import (
    CleanupResponse,
    LiveHealthResponse,
    PrepareResponse,
    ReadyHealthResponse,
    SearchRequest,
    SearchResponse,
    StatusResponse,
)
from movie_discovery_demo.config import VECTOR_DIMENSION, ConfigurationError
from movie_discovery_demo.data import DataContractError
from movie_discovery_demo.embedding import ModelContractError
from movie_discovery_demo.repository import RepositoryContractError, SearchContractError
from movie_discovery_demo.service import MovieDiscoveryService, ServiceContractError

CONTRACT_ERRORS = (
    ConfigurationError,
    DataContractError,
    ModelContractError,
    RepositoryContractError,
    SearchContractError,
    ServiceContractError,
)


def create_app(*, service: MovieDiscoveryService | None = None) -> FastAPI:
    resolved = service or MovieDiscoveryService()
    auto_prepare = service is None
    root_path = os.environ.get("APP_ROOT_PATH", "").rstrip("/")

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        if auto_prepare:
            resolved.cleanup()
            resolved.prepare()
        yield

    application = FastAPI(
        title="Movie Discovery with Search Aggregation Demo",
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
                    view=request.view,
                    facet=request.facet,
                    limit=request.limit,
                    max_runtime=request.max_runtime,
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

    def _poster_response(path: Path, filename: str) -> FileResponse:
        if not path.is_file():
            raise HTTPException(status_code=404, detail="Poster not found")
        return FileResponse(
            path,
            media_type="image/jpeg",
            filename=filename,
            headers={"Cache-Control": "public, max-age=86400, immutable"},
        )

    @application.get("/api/v1/poster/{movie_id}", response_class=FileResponse)
    def poster(movie_id: str) -> FileResponse:
        if not re.fullmatch(r"m\d{4}", movie_id):
            raise HTTPException(status_code=404, detail="Poster not found")
        path = resolved.config.data_root / "posters" / f"{movie_id}.jpg"
        return _poster_response(path, path.name)

    @application.get("/api/v1/poster/{movie_id}/thumb", response_class=FileResponse)
    def poster_thumb(movie_id: str) -> FileResponse:
        if not re.fullmatch(r"m\d{4}", movie_id):
            raise HTTPException(status_code=404, detail="Poster not found")
        path = resolved.config.data_root / "posters" / "thumbs" / f"{movie_id}.jpg"
        return _poster_response(path, path.name)

    static_dir = os.environ.get("STATIC_DIR")
    if static_dir and Path(static_dir).is_dir():
        application.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

    return application


app = create_app()
