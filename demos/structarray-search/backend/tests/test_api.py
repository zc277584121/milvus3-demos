from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from structarray_hybrid_demo.main import create_app
from structarray_hybrid_demo.service import StructArrayHybridService


class FakeEmbedder:
    loaded = True
    cache_available = True

    @property
    def identity(self):
        return type(
            "Identity",
            (),
            {
                "public_dict": lambda self: {
                    "model_id": "gpahal/bge-m3-onnx-int8",
                    "revision": "2b34e84df040034d4b9eabb62383a87c18955822",
                    "vector_dimension": 1024,
                    "dense_output_name": "dense_vecs",
                    "device": "cpu",
                }
            },
        )()

    def load(self) -> None: ...

    def unload(self) -> None: ...


class FakeRepository:
    def status(self):
        return {
            "milvus_uri": "http://127.0.0.1:49530",
            "server_version": "3.0.0",
            "collection_name": "milvus3_demos_structarray_hybrid_synthetic",
            "collection_exists": True,
            "raw_collection_names": ["milvus3_demos_structarray_hybrid_synthetic"],
            "row_count": 30,
            "index_names": ["a", "b"],
        }

    def cleanup(self):
        return {
            "status": "clean",
            "server_version": "3.0.0",
            "collection_name": "milvus3_demos_structarray_hybrid_synthetic",
            "dropped": True,
            "indexes_deleted_with_collection": ["a", "b"],
            "collection_absent_after": True,
        }


class FakeService(StructArrayHybridService):
    def __init__(self) -> None:
        self.embedder = FakeEmbedder()
        self.repository = FakeRepository()

    def status(self):
        return {
            "status": "ready",
            "sample_size": 30,
            "model": {
                **self.embedder.identity.public_dict(),
                "loaded": True,
                "cache_available": True,
            },
            "dataset": {
                "dataset_id": "synthetic-driving-scenes-r1",
                "dataset_version": 1,
                "video_count": 30,
                "observation_count": 540,
                "evidence_frame_count": 30,
                "manifest_sha256": "m",
            },
            "milvus": self.repository.status(),
            "query_presets": [
                {
                    "id": "black-suv-residential",
                    "text": "a black suv in a residential neighborhood",
                    "scene_terms": ["local_residential"],
                    "object_terms": ["suv"],
                    "color_terms": ["black"],
                }
            ],
        }


def test_health_and_status(client: TestClient) -> None:
    live = client.get("/healthz/live")
    assert live.status_code == 200
    assert live.json()["demo_id"] == "structarray-search"

    status = client.get("/api/v1/status")
    assert status.status_code == 200
    body = status.json()
    assert body["status"] == "ready"
    assert body["model"]["vector_dimension"] == 1024
    assert body["milvus"]["collection_exists"] is True


@pytest.fixture
def client() -> TestClient:
    app = create_app(service=FakeService())
    return TestClient(app)


def test_ready_health_reflects_prepared_collection(client: TestClient) -> None:
    ready = client.get("/healthz/ready")
    assert ready.status_code == 200
    assert ready.json()["status"] == "healthy"


def test_search_requires_query_text(client: TestClient) -> None:
    response = client.post("/api/v1/search", json={"query": "", "limit": 8})
    assert response.status_code == 422


def test_cleanup_contract(client: TestClient) -> None:
    response = client.post("/api/v1/cleanup")
    assert response.status_code == 200
    assert response.json()["status"] == "clean"
