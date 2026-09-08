"""FastAPI service for the real Milvus Function Chain rerank demo."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Response, status
from fastapi.responses import FileResponse

from function_chain_demo.api_models import (
    DemoStatusResponse,
    LiveHealthResponse,
    QueryOptionResponse,
    ReadyHealthResponse,
    SearchRequest,
    SearchResponse,
)
from function_chain_demo.catalog import DATASET, DATASET_IDENTITY, FEATURE_NAMES, QUERIES
from function_chain_demo.config import DemoSettings
from function_chain_demo.modeling import MODEL_VERSION
from function_chain_demo.service import DemoProvisioner, RuntimeManifest, SearchService

DEMO_ID = "function-chain-rerank"
CAPABILITY = "XGBoost + Function Chain rerank"


def create_app(
    *,
    settings: DemoSettings | None = None,
    provisioner: DemoProvisioner | None = None,
    search_service: SearchService | None = None,
) -> FastAPI:
    resolved_settings = settings or DemoSettings()
    resolved_provisioner = provisioner or DemoProvisioner(resolved_settings)
    resolved_search = search_service or SearchService(resolved_settings)

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        manifest = resolved_provisioner.prepare()
        application.state.manifest = manifest
        try:
            yield
        finally:
            if resolved_settings.cleanup_on_shutdown:
                resolved_provisioner.cleanup()
                application.state.manifest = None

    application = FastAPI(
        title="Function Chain Rerank Demo",
        version="0.2.0",
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
        lifespan=lifespan,
    )
    application.state.manifest = None
    application.state.search_service = resolved_search

    @application.get("/healthz/live", response_model=LiveHealthResponse)
    def live_health() -> LiveHealthResponse:
        return LiveHealthResponse()

    @application.get("/healthz/ready", response_model=ReadyHealthResponse)
    def ready_health(response: Response) -> ReadyHealthResponse:
        manifest: RuntimeManifest | None = application.state.manifest
        if manifest is None:
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
            return ReadyHealthResponse(
                status="unhealthy",
                milvus_version=None,
                collection_name=resolved_settings.collection_name,
                model_sha256=None,
            )
        return ReadyHealthResponse(
            status="healthy",
            milvus_version=manifest.milvus_version,
            collection_name=manifest.collection_name,
            model_sha256=str(manifest.model["sha256"]),
        )

    @application.get("/api/v1/status", response_model=DemoStatusResponse)
    def demo_status() -> DemoStatusResponse:
        return DemoStatusResponse(
            demo_id=DEMO_ID,
            capability=CAPABILITY,
            implementation_status="backend_ready",
            expected_milvus_version=resolved_settings.milvus_expected_version,
            execution_path="milvus_l0_xgboost_function_chain",
            model_version=MODEL_VERSION,
            feature_order=list(FEATURE_NAMES),
            dataset=DATASET_IDENTITY,
        )

    @application.get("/api/v1/queries", response_model=list[QueryOptionResponse])
    def query_options() -> list[QueryOptionResponse]:
        return [
            QueryOptionResponse(
                id=query.id,
                query_text=query.query_text,
                story=query.story,
                split=query.split,
                dataset=DATASET_IDENTITY,
            )
            for query in QUERIES
        ]

    @application.get("/api/v1/assets/{asset_path:path}", response_class=FileResponse)
    def product_asset(asset_path: str) -> FileResponse:
        try:
            path = DATASET.product_asset(asset_path)
        except (KeyError, OSError, ValueError) as exc:
            raise HTTPException(status_code=404, detail="Unknown product asset") from exc
        return FileResponse(
            path,
            media_type="image/jpeg",
            headers={
                "Cache-Control": "public, max-age=31536000, immutable",
                "X-Content-Type-Options": "nosniff",
                "X-Dataset-Revision": str(DATASET_IDENTITY["revision"]),
                "X-Content-SHA256": next(
                    str(entry["sha256"])
                    for entry in DATASET.manifest["files"]
                    if entry["path"] == asset_path
                ),
            },
        )

    @application.post("/api/v1/search", response_model=SearchResponse)
    def compare_search(request: SearchRequest) -> SearchResponse:
        try:
            comparison = application.state.search_service.compare(
                request.query_text, query_id=request.query_id
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Unknown demo query") from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return SearchResponse.model_validate(comparison.public_dict())

    return application


app = create_app()
