from __future__ import annotations

from fastapi.testclient import TestClient

from embedding_list_demo.config import default_dataset_root
from embedding_list_demo.main import create_app
from embedding_list_demo.manual import load_manual, load_queries, resolve_page_image


class FakeService:
    def __init__(self) -> None:
        self.root = default_dataset_root()
        self.manifest = load_manual(self.root)
        self.queries = load_queries(self.root, self.manifest)

    def dataset(self) -> dict[str, object]:
        return {
            "dataset_id": self.manifest.dataset_id,
            "title": self.manifest.title,
            "revision": self.manifest.revision,
            "document_identifier": self.manifest.document_identifier,
            "ntrs_id": self.manifest.ntrs_id,
            "ntrs_record_url": self.manifest.ntrs_record_url,
            "official_pdf_url": self.manifest.official_pdf_url,
            "distribution": self.manifest.distribution,
            "rights_determination": self.manifest.rights_determination,
            "contains_third_party_material": self.manifest.contains_third_party_material,
            "attribution": self.manifest.attribution,
            "authors": [author.public_dict() for author in self.manifest.authors],
            "language": self.manifest.language,
            "render_statement": self.manifest.render_statement,
            "endorsement_statement": self.manifest.endorsement_statement,
            "renderer": self.manifest.renderer,
            "renderer_version": self.manifest.renderer_version,
            "render_dpi": self.manifest.render_dpi,
            "pdf_page_index_base": self.manifest.pdf_page_index_base,
            "source_pdf_page_count": self.manifest.source_pdf_page_count,
            "source_pdf_bytes": self.manifest.source_pdf_bytes,
            "page_count": self.manifest.page_count,
            "manifest_sha256": self.manifest.manifest_sha256,
            "pdf_sha256": self.manifest.pdf_sha256,
            "queries_sha256": self.manifest.queries_sha256,
        }

    @staticmethod
    def model() -> dict[str, object]:
        return {
            "model_id": "vidore/colSmol-256M",
            "inference_dtype": "float32",
            "configured_device": "cpu",
            "device_type": "cpu",
            "cpu_only": True,
        }

    def status(self):
        return {
            "status": "not_prepared",
            "implementation_status": "implemented",
            "runtime_ready": False,
            "execution_path": "milvus_embedding_list_max_sim_cosine",
            "collection_name": "collection",
            "index_name": "index",
            "anns_field": "patches[patch_embedding]",
            "metric_type": "MAX_SIM_COSINE",
            "dataset_exists": True,
            "embedding_cache_exists": False,
            "dataset": self.dataset(),
            "query_presets": [query.public_dict() for query in self.queries],
            "last_prepare": None,
            "prepare_timings": {"cold": None, "warm": None},
            "model": self.model(),
            "milvus": {"server_version": "3.0.0", "target_exists": False},
        }

    def manual(self):
        return {
            "generated": False,
            **self.manifest.public_dict(),
            "query_presets": [query.public_dict() for query in self.queries],
        }

    def prepare(self):
        timings = {
            "dataset_verify_ms": 0.1,
            "model_identity_ms": 0.1,
            "cache_read_ms": 0.1,
            "page_inference_ms": 1.0,
            "cache_write_ms": 0.1,
            "milvus_prepare_ms": 0.5,
            "total_ms": 1.9,
        }
        return {
            "status": "prepared",
            "implementation_status": "implemented",
            "runtime_ready": True,
            "manual_generated": False,
            "manual": self.manifest.public_dict(),
            "query_presets": [query.public_dict() for query in self.queries],
            "cache": {"hit": False},
            "model": self.model(),
            "page_inference": {"vector_counts": [2] * 40},
            "milvus": {"status": "prepared"},
            "timings": timings,
            "prepare_timings": {"cold": timings, "warm": None},
            "latency_ms": 1.9,
        }

    def search(self, query: str):
        timings = {
            "query_inference_ms": 0.4,
            "milvus_search_ms": 0.2,
            "local_score_ms": 0.1,
            "total_ms": 0.7,
        }
        return {
            "status": "passed",
            "execution_path": "milvus_embedding_list_max_sim_cosine",
            "query": query,
            "collection_name": "collection",
            "index_name": "index",
            "anns_field": "patches[patch_embedding]",
            "metric_type": "MAX_SIM_COSINE",
            "dataset": self.dataset(),
            "query_embedding": {"vector_counts": [3], "vector_dimension": 128},
            "page_vector_counts": [2] * 40,
            "milvus_page_order": ["nasa-seh-printed-053"],
            "local_page_order": ["nasa-seh-printed-053"],
            "ranking_equal": True,
            "score_absolute_tolerance": 0.005,
            "max_score_delta": 0.0,
            "score_within_tolerance": True,
            "results": [],
            "local_explanations": [],
            "model": self.model(),
            "timings": timings,
            "latency_ms": 0.7,
        }

    def page_image(self, page_id: str):
        return resolve_page_image(self.root, self.manifest, page_id)

    @staticmethod
    def cleanup():
        return {
            "status": "clean",
            "milvus": {"target_absent_after": True},
            "embedding_cache_removed": True,
            "dataset_source_preserved": True,
            "runtime_root_exists": False,
            "model_loaded": False,
        }


def test_api_contracts_cpu_readiness_metadata_timings_and_page_evidence() -> None:
    service = FakeService()
    client = TestClient(create_app(service=service))  # type: ignore[arg-type]

    assert client.get("/healthz/live").json()["status"] == "healthy"
    ready = client.get("/healthz/ready")
    assert ready.status_code == 503
    assert ready.json()["device_type"] == "cpu"
    status = client.get("/api/v1/status").json()
    assert status["implementation_status"] == "implemented"
    assert status["runtime_ready"] is False
    assert status["dataset"]["distribution"] == "PUBLIC"
    assert status["dataset"]["rights_determination"] == "PUBLIC_USE_PERMITTED"
    assert len(status["dataset"]["authors"]) == 3
    assert len(status["query_presets"]) == 8
    assert all(set(item) == {"query_id", "text"} for item in status["query_presets"])
    manual = client.get("/api/v1/manual").json()
    assert manual["page_count"] == 40
    assert len(manual["manifest_sha256"]) == 64
    assert "target_page_ids" not in str(manual["query_presets"])
    prepare = client.post("/api/v1/prepare").json()
    assert prepare["status"] == "prepared"
    assert prepare["timings"]["page_inference_ms"] == 1.0
    search = client.post("/api/v1/search", json={"query": service.queries[0].text})
    assert search.status_code == 200
    assert search.json()["ranking_equal"] is True
    assert "ground_truth" not in search.json()
    page = client.get("/api/v1/pages/nasa-seh-printed-053")
    assert page.status_code == 200
    assert page.headers["content-type"] == "image/png"
    cleanup = client.post("/api/v1/cleanup").json()
    assert cleanup["dataset_source_preserved"] is True


def test_api_rejects_queries_outside_the_length_contract() -> None:
    client = TestClient(create_app(service=FakeService()))  # type: ignore[arg-type]

    assert client.post("/api/v1/search", json={"query": ""}).status_code == 422
    response = client.post("/api/v1/search", json={"query": "x" * 513})

    assert response.status_code == 422
    assert response.json()["detail"][0]["msg"] == ("String should have at most 512 characters")
