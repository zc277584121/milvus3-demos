from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from minhash_near_dup_demo.main import create_app
from minhash_near_dup_demo.service import MinHashService


class FakeRepository:
    def __init__(self) -> None:
        self.owns_collection = True

    def raw_audit(self):
        return {
            "milvus_uri": "http://127.0.0.1:49530",
            "server_version": "3.0.0",
            "expected_server_version": "3.0.0",
            "raw_collection_names": ["milvus3_demos_minhash_near_duplicate"],
            "target_collection": "milvus3_demos_minhash_near_duplicate",
            "target_exists": True,
            "index_names": ["milvus3_demos_minhash_signature"],
            "indexes": [],
            "schema": {},
            "collection_stats": {"row_count": 36},
        }

    def cleanup(self):
        class Report:
            def public_dict(self):
                return {
                    "status": "clean",
                    "collection_name": "milvus3_demos_minhash_near_duplicate",
                    "dropped": True,
                    "target_absent_after": True,
                    "unknown_collections_unchanged": True,
                }

        return Report()


class FakeService(MinHashService):
    def __init__(self) -> None:
        self.repository = FakeRepository()

    def status(self):
        return {
            "status": "ready",
            "implementation_status": "implemented",
            "execution_path": "milvus_minhash_lsh_near_duplicate",
            "capability": "MinHash Function + MINHASH_LSH",
            "collection_name": "milvus3_demos_minhash_near_duplicate",
            "index_name": "milvus3_demos_minhash_signature",
            "text_field": "content",
            "signature_field": "minhash_signature",
            "metric_type": "MHJACCARD",
            "dataset": {
                "dataset_id": "original-short-stories-r1",
                "dataset_version": 1,
                "document_count": 36,
                "documents_sha256": "d",
                "presets_sha256": "p",
            },
            "query_presets": [
                {
                    "id": "river-weekend",
                    "label": "A weekend by the river",
                    "title": "A Weekend by the River",
                    "text": "On Sunday, we followed the river path to the old bridge.",
                    "description": "A lightly edited copy.",
                }
            ],
            "milvus": self.repository.raw_audit(),
            "last_prepare": None,
        }


@pytest.fixture
def client() -> TestClient:
    app = create_app(service=FakeService())
    return TestClient(app)


def test_live_health(client: TestClient) -> None:
    live = client.get("/healthz/live")
    assert live.status_code == 200
    assert live.json()["demo_id"] == "minhash-near-duplicate"


def test_status_reflects_prepared_collection(client: TestClient) -> None:
    status = client.get("/api/v1/status")
    assert status.status_code == 200
    body = status.json()
    assert body["status"] == "ready"
    assert body["milvus"]["target_exists"] is True


def test_ready_health_reflects_prepared_collection(client: TestClient) -> None:
    ready = client.get("/healthz/ready")
    assert ready.status_code == 200
    assert ready.json()["status"] == "healthy"


def test_search_rejects_empty_query(client: TestClient) -> None:
    response = client.post("/api/v1/search", json={"query": "", "threshold": 0.85, "limit": 5})
    assert response.status_code == 422


def test_search_rejects_out_of_range_threshold(client: TestClient) -> None:
    response = client.post("/api/v1/search", json={"query": "hello", "threshold": 1.5, "limit": 5})
    assert response.status_code == 422


def test_cleanup_contract(client: TestClient) -> None:
    response = client.post("/api/v1/cleanup")
    assert response.status_code == 200
    assert response.json()["status"] == "clean"
