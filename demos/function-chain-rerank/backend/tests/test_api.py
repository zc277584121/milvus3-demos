import json

from fastapi.testclient import TestClient

from function_chain_demo.catalog import DATASET_IDENTITY, PRODUCTS, QUERIES
from function_chain_demo.config import DemoSettings
from function_chain_demo.main import create_app
from function_chain_demo.service import RankedProduct, SearchComparison, build_chain_steps


def ranked_product(product_id: int) -> RankedProduct:
    return RankedProduct.from_product(PRODUCTS[product_id - 1], rank=1, score=0.99)


class FakeSearchService:
    def compare(self, query_text: str, query_id: str | None = None) -> SearchComparison:
        query = QUERIES[0]
        if query_id not in {None, query.id}:
            raise KeyError(query_id)
        if query_text.strip() != query.query_text:
            raise ValueError("query_id and query_text do not identify the same fixed query")
        vector = (ranked_product(1), ranked_product(2))
        business = (ranked_product(2), ranked_product(1))
        return SearchComparison(
            query_id=query.id,
            query_text=query.query_text,
            execution_path="milvus_l0_xgboost_function_chain",
            model_resource_name="milvus3_demos_function_chain_rerank_model",
            function_chain={
                "stage": "L0_RERANK",
                "operation": "xgboost",
                "feature_order": [
                    "semantic_score",
                    "rating",
                    "popularity",
                    "price_affinity",
                    "freshness",
                ],
                "parallel_inputs": False,
                "intermediate_feature_orders": False,
                "vector_order_score": "semantic_score",
                "business_order_score": "business_score",
                "chain_steps": build_chain_steps("milvus3_demos_function_chain_rerank_model"),
            },
            dataset=DATASET_IDENTITY,
            data_model={
                "fields": [
                    {
                        "name": "id",
                        "type": "INT64",
                        "role": "primary_key",
                        "note": "stable product id",
                    },
                ],
                "chain_trace": {"item_id": "SYN-DESK-001", "title": "x", "steps": []},
            },
            vector_order=vector,
            business_order=business,
        )


def make_client() -> TestClient:
    settings = DemoSettings(cleanup_on_shutdown=False)
    app = create_app(settings=settings, search_service=FakeSearchService())
    return TestClient(app)


def test_status_describes_real_synthetic_and_simulated_operations_contract() -> None:
    response = make_client().get("/api/v1/status")

    assert response.status_code == 200
    payload = response.json()
    assert payload["demo_id"] == "function-chain-rerank"
    assert payload["execution_path"] == "milvus_l0_xgboost_function_chain"
    assert payload["model_version"] == "synthetic-commerce-catalog-r1-reranker-v1"
    assert payload["dataset"] == DATASET_IDENTITY
    assert payload["dataset"]["real_product_photos"] is False
    assert payload["dataset"]["real_transaction_data"] is False
    assert payload["dataset"]["contains_simulated_operational_signals"] is True


def test_query_options_expose_only_natural_query_contract() -> None:
    response = make_client().get("/api/v1/queries")

    assert response.status_code == 200
    queries = response.json()
    assert len(queries) == 6
    forbidden = {"business_target_id", "baseline_reference_id", "hard_negative_ids"}
    assert all(not (set(query) & forbidden) for query in queries)
    assert all(query["query_text"] and query["story"] for query in queries)
    serialized = json.dumps(queries)
    assert all(key not in serialized for key in forbidden)
    assert all(query["dataset"] == DATASET_IDENTITY for query in queries)


def test_search_response_preserves_vector_and_business_order_without_api_sorting() -> None:
    query = QUERIES[0]
    response = make_client().post(
        "/api/v1/search",
        json={"query_id": query.id, "query_text": query.query_text},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["query_text"] == query.query_text
    assert payload["execution_path"] == "milvus_l0_xgboost_function_chain"
    assert payload["function_chain"] == {
        "stage": "L0_RERANK",
        "operation": "xgboost",
        "feature_order": [
            "semantic_score",
            "rating",
            "popularity",
            "price_affinity",
            "freshness",
        ],
        "parallel_inputs": False,
        "intermediate_feature_orders": False,
        "vector_order_score": "semantic_score",
        "business_order_score": "business_score",
        "chain_steps": build_chain_steps("milvus3_demos_function_chain_rerank_model"),
    }
    assert payload["data_model"]["fields"][0]["name"] == "id"
    assert payload["data_model"]["chain_trace"]["item_id"] == "SYN-DESK-001"
    assert [product["id"] for product in payload["vector_order"]] == [1, 2]
    assert [product["id"] for product in payload["business_order"]] == [2, 1]
    product = payload["vector_order"][0]
    assert product["item_id"] == PRODUCTS[0].item_id
    assert product["selected_image_id"] == PRODUCTS[0].selected_image_id
    assert product["image_role"] == PRODUCTS[0].image_role
    assert product["image_mime"] == "image/jpeg"
    assert product["operational_signal_provenance"] == (
        "deterministic_simulated_operational_signal"
    )
    assert product["field_provenance"]["rating_value"] == (
        "deterministic_simulated_operational_signal"
    )


def test_unknown_or_mismatched_query_is_rejected_without_fallback() -> None:
    query = QUERIES[0]
    unknown = make_client().post(
        "/api/v1/search",
        json={"query_id": "unknown", "query_text": query.query_text},
    )
    mismatch = make_client().post(
        "/api/v1/search",
        json={"query_id": query.id, "query_text": "a different natural product search"},
    )

    assert unknown.status_code == 404
    assert mismatch.status_code == 422


def test_only_manifest_listed_jpegs_are_served_and_traversal_is_denied() -> None:
    client = make_client()
    product = PRODUCTS[0]
    response = client.get(f"/api/v1/assets/{product.image_path}")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/jpeg")
    assert response.headers["x-dataset-revision"] == "synthetic-commerce-catalog-r1"
    assert response.headers["x-content-sha256"] == product.image_sha256
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.content.startswith(b"\xff\xd8")
    assert response.content.endswith(b"\xff\xd9")

    for unsafe_path in (
        "products.json",
        "images/unknown.jpg",
        "%2e%2e%2fmanifest.json",
        "images%2f%2e%2e%2fmanifest.json",
        f"{product.image_path}%00.svg",
    ):
        denied = client.get(f"/api/v1/assets/{unsafe_path}")
        assert denied.status_code == 404
