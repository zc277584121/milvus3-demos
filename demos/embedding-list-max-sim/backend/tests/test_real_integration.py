from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from embedding_list_demo.config import COLLECTION_NAME, RuntimeConfig, default_hf_hub_cache
from embedding_list_demo.main import create_app
from embedding_list_demo.manual import load_manual, load_queries
from embedding_list_demo.repository import EmbeddingListRepository
from embedding_list_demo.service import EmbeddingListService


@pytest.mark.skipif(
    os.getenv("RUN_EMBEDDING_LIST_INTEGRATION") != "1",
    reason="Real CPU ColSmol and project-GA Milvus integration is opt-in",
)
def test_real_cpu_colsmol_embedding_list_max_sim_and_cleanup(tmp_path: Path) -> None:
    config = RuntimeConfig(
        runtime_root=tmp_path / "runtime",
        hf_hub_cache=default_hf_hub_cache(),
        device="cpu",
    )
    repository = EmbeddingListRepository()
    service = EmbeddingListService(config=config, repository=repository)
    client = TestClient(create_app(service=service))
    before = repository.raw_audit()
    assert before.server_version == "3.0.0"
    assert COLLECTION_NAME not in before.raw_collection_names
    report: dict[str, object] = {"before": before.public_dict()}
    cleanup_done = False
    cleanup: dict[str, object] | None = None

    try:
        assert client.get("/api/v1/status").json()["status"] == "not_prepared"
        prepare_response = client.post("/api/v1/prepare")
        assert prepare_response.status_code == 200, prepare_response.text
        prepare = prepare_response.json()
        cached_response = client.post("/api/v1/prepare")
        assert cached_response.status_code == 200, cached_response.text
        cached_prepare = cached_response.json()
        assert client.get("/healthz/ready").status_code == 200
        manual = client.get("/api/v1/manual").json()
        assert manual["page_count"] == 40
        assert manual["distribution"] == "PUBLIC"
        assert manual["rights_determination"] == "PUBLIC_USE_PERMITTED"
        assert len(manual["authors"]) == 3
        assert len(manual["query_presets"]) == 8
        assert all(set(item) == {"query_id", "text"} for item in manual["query_presets"])
        page_response = client.get("/api/v1/pages/nasa-seh-printed-053")
        assert page_response.status_code == 200
        assert page_response.headers["content-type"] == "image/png"
        searches = []
        for query_preset in manual["query_presets"]:
            search_response = client.post("/api/v1/search", json={"query": query_preset["text"]})
            assert search_response.status_code == 200, search_response.text
            searches.append(search_response.json())
        search = searches[0]
        report.update(
            {
                "prepare": prepare,
                "cached_prepare": cached_prepare,
                "search": search,
                "repeated_searches": searches,
            }
        )

        assert prepare["cache"]["hit"] is False
        assert cached_prepare["cache"]["hit"] is True
        assert prepare["manual"]["page_count"] == 40
        assert prepare["model"]["configured_device"] == "cpu"
        assert prepare["model"]["inference_dtype"] == "float32"
        assert prepare["page_inference"]["vector_dimension"] == 128
        assert len(prepare["page_inference"]["vector_counts"]) == 40
        assert search["query_embedding"]["vector_dimension"] == 128
        assert search["query_embedding"]["vector_counts"][0] > 1
        internal_manual = load_manual(config.dataset_root)
        internal_queries = load_queries(config.dataset_root, internal_manual)
        for query, item in zip(internal_queries, searches, strict=True):
            ranked_ids = [result["page_id"] for result in item["results"]]
            assert ranked_ids[0] == query.target_page_ids[0]
            assert all(ranked_ids.index(negative.page_id) > 0 for negative in query.hard_negatives)
            assert len(ranked_ids) == 40
            assert "ground_truth" not in item
            assert all("is_expected_page" not in result for result in item["results"])
            explanations = item["local_explanations"]
            assert [value["page_id"] for value in explanations] == ranked_ids[:3]
            assert all(
                value["source"] == "local_colsmol_query_page_multi_vector"
                and value["rank_source"] == "milvus_page_level_max_sim_cosine"
                and value["grid"]
                == {
                    "columns": 32,
                    "rows": 32,
                    "patch_count": 1024,
                    "coordinate_space": "normalized_unmodified_page",
                }
                and len(value["patches"]) == 1024
                and min(patch["intensity"] for patch in value["patches"]) == 0
                and max(patch["intensity"] for patch in value["patches"]) == 1
                and len({patch["intensity"] for patch in value["patches"]}) > 2
                for value in explanations
            )
        assert all(item["ranking_equal"] is True for item in searches)
        assert all(item["score_within_tolerance"] is True for item in searches)
        assert all(item["status"] == "passed" for item in searches)
        assert all(item["max_score_delta"] <= 0.005 for item in searches)
        cleanup_response = client.post("/api/v1/cleanup")
        assert cleanup_response.status_code == 200, cleanup_response.text
        cleanup = cleanup_response.json()
        cleanup_done = True
    finally:
        if not cleanup_done:
            cleanup = service.cleanup()
        after = repository.raw_audit()
        report["cleanup"] = cleanup
        report["after"] = after.public_dict()
        output = os.getenv("EMBEDDING_LIST_INTEGRATION_REPORT")
        if output:
            destination = Path(output)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

    assert cleanup is not None
    assert cleanup["milvus"]["target_absent_after"] is True
    assert cleanup["milvus"]["unknown_collections_unchanged"] is True
    assert cleanup["milvus"]["file_resources_unchanged"] is True
    assert cleanup["dataset_source_preserved"] is True
    assert config.runtime_root.exists() is False
    assert after.raw_collection_names == before.raw_collection_names
    assert after.raw_file_resource_names == before.raw_file_resource_names
    assert COLLECTION_NAME not in after.raw_collection_names
