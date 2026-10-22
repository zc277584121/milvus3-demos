from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from movie_discovery_demo.main import create_app
from movie_discovery_demo.service import MovieDiscoveryService


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
            "collection_name": "milvus3_demos_movie_catalog",
            "collection_exists": True,
            "raw_collection_names": ["milvus3_demos_movie_catalog"],
            "row_count": 90,
            "index_names": ["milvus3_demos_movie_synopsis_vector"],
        }

    def cleanup(self):
        return {
            "status": "clean",
            "server_version": "3.0.0",
            "collection_name": "milvus3_demos_movie_catalog",
            "dropped": True,
            "indexes_deleted_with_collection": ["milvus3_demos_movie_synopsis_vector"],
            "collection_absent_after": True,
        }


class FakeService(MovieDiscoveryService):
    def __init__(self) -> None:
        self.embedder = FakeEmbedder()
        self.repository = FakeRepository()

    def status(self):
        return {
            "status": "ready",
            "movie_count": 90,
            "model": {
                **self.embedder.identity.public_dict(),
                "loaded": True,
                "cache_available": True,
            },
            "dataset": {
                "dataset_id": "movie-catalog-r1",
                "dataset_version": 1,
                "movie_count": 90,
                "movies_sha256": "m",
                "presets_sha256": "p",
            },
            "milvus": self.repository.status(),
            "facets": [
                {"id": "genre", "field": "primary_genre", "label": "Genre"},
                {"id": "decade", "field": "decade", "label": "Decade"},
                {"id": "region", "field": "region", "label": "Region"},
            ],
            "query_presets": [
                {
                    "id": "lighthearted-adventure",
                    "label": "Lighthearted adventure",
                    "text": "A lighthearted adventure with ordinary people",
                    "description": "Warm, low-stakes journeys.",
                }
            ],
            "poster_placeholders": True,
        }


@pytest.fixture
def client() -> TestClient:
    app = create_app(service=FakeService())
    return TestClient(app)


def test_health_and_status(client: TestClient) -> None:
    live = client.get("/healthz/live")
    assert live.status_code == 200
    assert live.json()["demo_id"] == "movie-discovery"

    status = client.get("/api/v1/status")
    assert status.status_code == 200
    body = status.json()
    assert body["status"] == "ready"
    assert body["model"]["vector_dimension"] == 1024
    assert body["milvus"]["collection_exists"] is True
    assert body["poster_placeholders"] is True


def test_ready_health_reflects_prepared_collection(client: TestClient) -> None:
    ready = client.get("/healthz/ready")
    assert ready.status_code == 200
    assert ready.json()["status"] == "healthy"


def test_search_requires_query_text(client: TestClient) -> None:
    response = client.post("/api/v1/search", json={"query": "", "view": "grouped"})
    assert response.status_code == 422


def test_cleanup_contract(client: TestClient) -> None:
    response = client.post("/api/v1/cleanup")
    assert response.status_code == 200
    assert response.json()["status"] == "clean"
