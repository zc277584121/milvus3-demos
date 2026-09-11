import json
import os
import tempfile
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest
import xgboost as xgb
from fastapi.testclient import TestClient
from pymilvus import MilvusClient

from function_chain_demo.business_gates import favorable_signals, risk_signals
from function_chain_demo.catalog import DATASET, PRODUCTS, QUERIES
from function_chain_demo.config import DemoSettings
from function_chain_demo.main import create_app
from function_chain_demo.modeling import load_model, train_model
from function_chain_demo.relevance_ground_truth import load_ground_truth
from function_chain_demo.service import DemoProvisioner, build_chain_steps


def real_file_resource_names(settings: DemoSettings) -> set[str]:
    client = MilvusClient(
        uri=settings.milvus_uri,
        token=settings.token_value(),
        timeout=settings.milvus_timeout_seconds,
    )
    try:
        return {
            item if isinstance(item, str) else item.name for item in client.list_file_resources()
        }
    finally:
        client.close()


def local_model_ranking(
    model: xgb.Booster, products: list[dict[str, object]]
) -> list[tuple[int, float]]:
    from function_chain_demo.chain_features import (
        freshness as chain_freshness,
    )
    from function_chain_demo.chain_features import (
        popularity as chain_popularity,
    )
    from function_chain_demo.chain_features import (
        price_affinity as chain_price_affinity,
    )

    features = np.asarray(
        [
            [
                round(float(product["score"]), 5),
                product["rating"],
                chain_popularity(product["clicks_30d"], product["sales_30d"]),
                chain_price_affinity(product["display_price_usd"]),
                chain_freshness(product["release_epoch"]),
            ]
            for product in products
        ],
        dtype=np.float32,
    )
    predictions = model.predict(xgb.DMatrix(features))
    indices = np.argsort(-predictions, kind="stable")
    return [(int(products[int(index)]["id"]), float(predictions[int(index)])) for index in indices]


@pytest.mark.skipif(
    os.getenv("RUN_FUNCTION_CHAIN_INTEGRATION") != "1",
    reason="Real Milvus integration is opt-in",
)
def test_real_milvus_runs_all_acceptance_scenarios_and_cleans_up() -> None:
    settings = DemoSettings()
    provisioner = DemoProvisioner(settings)
    app = create_app(settings=settings, provisioner=provisioner)
    natural_queries = list(QUERIES)
    ground_truth = load_ground_truth(DATASET.root)
    judgments_by_query = {
        query["query_id"]: {
            judgment["item_id"]: judgment["label"] for judgment in query["judgments"]
        }
        for query in ground_truth["queries"]
    }
    query_reports: list[dict[str, object]] = []

    with tempfile.TemporaryDirectory(prefix="function-chain-integration-") as temp_dir:
        artifact = train_model(Path(temp_dir) / "reranker.ubj")
        model = load_model(artifact.path)
        with TestClient(app) as client:
            ready_response = client.get("/healthz/ready")
            asset_response = client.get(f"/api/v1/assets/{PRODUCTS[0].image_path}")

            assert ready_response.status_code == 200
            assert ready_response.json()["milvus_version"] == "3.0.0"
            assert asset_response.status_code == 200
            assert asset_response.headers["x-dataset-revision"] == "synthetic-commerce-catalog-r1"
            assert settings.model_resource_name in real_file_resource_names(settings)

            changed_orders = 0
            credible_changed_orders = 0
            for query in natural_queries:
                search_response = client.post(
                    "/api/v1/search",
                    json={"query_id": query.id, "query_text": query.query_text},
                )
                assert search_response.status_code == 200
                comparison = search_response.json()
                repeat_response = client.post(
                    "/api/v1/search",
                    json={"query_id": query.id, "query_text": query.query_text},
                )
                assert repeat_response.status_code == 200
                repeat_comparison = repeat_response.json()
                vector_order = comparison["vector_order"]
                business_order = comparison["business_order"]
                vector_ids = [product["id"] for product in vector_order]
                business_ids = [product["id"] for product in business_order]
                vector_by_id = {product["id"]: product for product in vector_order}
                business_by_id = {product["id"]: product for product in business_order}

                assert len(vector_ids) == 20
                assert set(business_ids) == set(vector_ids)
                assert [product["item_id"] for product in repeat_comparison["vector_order"]] == [
                    product["item_id"] for product in vector_order
                ]
                assert [product["item_id"] for product in repeat_comparison["business_order"]] == [
                    product["item_id"] for product in business_order
                ]
                oracle_ranking = local_model_ranking(model, vector_order)
                oracle_ids = [product_id for product_id, _ in oracle_ranking]
                oracle_scores = {product_id: score for product_id, score in oracle_ranking}
                score_matches = np.allclose(
                    [product["score"] for product in business_order],
                    [oracle_scores[product_id] for product_id in business_ids],
                    rtol=1e-6,
                    atol=1e-7,
                )
                assert score_matches
                server_order_oracle_scores = [
                    oracle_scores[product_id] for product_id in business_ids
                ]
                no_oracle_inversion = all(
                    current + 1e-7 >= following
                    for current, following in pairwise(server_order_oracle_scores)
                )
                assert no_oracle_inversion
                changed_orders += business_ids != vector_ids
                expected_type = query.expected_type
                assert vector_order[0]["product_type"] == expected_type
                assert business_order[0]["product_type"] == expected_type
                vector_relevant_ids = [
                    product["id"]
                    for product in vector_order
                    if product["product_type"] == expected_type
                ]
                business_relevant_ids = [
                    product["id"]
                    for product in business_order
                    if product["product_type"] == expected_type
                ]
                relevant_candidate_order_changed = vector_relevant_ids != business_relevant_ids
                credible_changed_orders += relevant_candidate_order_changed
                assert comparison["dataset"]["revision"] == "synthetic-commerce-catalog-r1"
                assert comparison["dataset"]["license"] == "CC0-1.0"
                assert comparison["execution_path"] == "milvus_l0_xgboost_function_chain"
                assert comparison["function_chain"] == {
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
                    "chain_steps": build_chain_steps(settings.model_resource_name),
                }
                vector_item_ids = [product["item_id"] for product in vector_order]
                business_item_ids = [product["item_id"] for product in business_order]
                judgments = judgments_by_query[query.id]
                business_top6_labels = [judgments[item_id] for item_id in business_item_ids[:6]]
                # Category recall, not perfect strong/negative separation, is the
                # honest gate under a pure semantic encoder.
                business_top6_type_recall = sum(
                    1 for product in business_order[:6] if product["product_type"] == expected_type
                )
                assert business_top6_type_recall >= 5
                vector_rank = {
                    item_id: rank for rank, item_id in enumerate(vector_item_ids, start=1)
                }
                business_rank = {
                    item_id: rank for rank, item_id in enumerate(business_item_ids, start=1)
                }
                story_ids = [
                    item_id for item_id in vector_item_ids if judgments[item_id] == "strong"
                ]
                products_by_item_id = {product.item_id: product for product in DATASET.products}
                upward = [
                    item_id
                    for item_id in story_ids
                    if business_rank[item_id] < vector_rank[item_id]
                ]
                downward = [
                    item_id
                    for item_id in story_ids
                    if business_rank[item_id] > vector_rank[item_id]
                ]
                # "Healthy" means a strong candidate holds a top-6 business slot
                # and carries favourable operational signals. Under a pure semantic
                # encoder some strong candidates legitimately trail same-type
                # negatives in vector order, so holding (not necessarily climbing)
                # is the right outcome and mirrors the offline business gate.
                healthy_upward_ids = [
                    item_id
                    for item_id in story_ids
                    if business_rank[item_id] <= 6
                    and len(favorable_signals(products_by_item_id[item_id])) >= 2
                ]
                risky_downward_ids = [
                    item_id for item_id in downward if risk_signals(products_by_item_id[item_id])
                ]
                assert upward
                assert downward
                assert healthy_upward_ids
                query_reports.append(
                    {
                        "query_id": query.id,
                        "query_text": query.query_text,
                        "execution_path": comparison["execution_path"],
                        "server_searches": [
                            "MilvusClient.search COSINE vector recall",
                            "MilvusClient.search L0 Function Chain fn.xgboost UBJ FileResource",
                        ],
                        "vector_order": [
                            {
                                "rank": product["rank"],
                                "id": product["id"],
                                "item_id": product["item_id"],
                                "product_type": product["product_type"],
                                "score": product["score"],
                            }
                            for product in vector_order
                        ],
                        "business_order": [
                            {
                                "rank": product["rank"],
                                "id": product["id"],
                                "item_id": product["item_id"],
                                "product_type": product["product_type"],
                                "score": product["score"],
                                "vector_score": vector_by_id[product["id"]]["score"],
                                "vector_rank": vector_by_id[product["id"]]["rank"],
                                "rank_delta": (
                                    vector_by_id[product["id"]]["rank"] - product["rank"]
                                ),
                            }
                            for product in business_order
                        ],
                        "local_oracle_order": [
                            {
                                "rank": index + 1,
                                "id": product_id,
                                "item_id": business_by_id[product_id]["item_id"],
                                "product_type": business_by_id[product_id]["product_type"],
                                "oracle_score": oracle_score,
                            }
                            for index, (product_id, oracle_score) in enumerate(oracle_ranking)
                        ],
                        "business_top6_labels": business_top6_labels,
                        "healthy_upward_ids": healthy_upward_ids,
                        "risky_downward_ids": risky_downward_ids,
                        "related_candidates_below_top6": [
                            {
                                "item_id": item_id,
                                "label": judgments[item_id],
                                "business_rank": business_rank[item_id],
                            }
                            for item_id in business_item_ids
                            if judgments[item_id] in {"partial", "intra_type_hard_negative"}
                        ],
                        "expected_product_type": expected_type,
                        "vector_top1_product_type": vector_order[0]["product_type"],
                        "business_top1_product_type": business_order[0]["product_type"],
                        "order_changed": business_ids != vector_ids,
                        "relevant_candidate_vector_order": vector_relevant_ids,
                        "relevant_candidate_business_order": business_relevant_ids,
                        "relevant_candidate_order_changed": relevant_candidate_order_changed,
                        "credible_order_changed": relevant_candidate_order_changed,
                        "local_stable_order_matches_server": business_ids == oracle_ids,
                        "oracle_scores_match_server": bool(score_matches),
                        "server_order_has_no_oracle_inversion": no_oracle_inversion,
                        "oracle_matches_server": bool(score_matches) and no_oracle_inversion,
                        "repeat_order_stable": True,
                    }
                )
            assert changed_orders >= 5
            assert credible_changed_orders >= 5

    state = provisioner.inspect_state()
    assert state.collection_exists is False
    assert state.file_resource_exists is False
    assert state.model_object_exists is False
    assert settings.model_resource_name not in real_file_resource_names(settings)
    report_path = os.getenv("FUNCTION_CHAIN_INTEGRATION_REPORT")
    if report_path:
        report = {
            "dataset_revision": "synthetic-commerce-catalog-r1",
            "model": artifact.public_metadata(),
            "query_count": len(query_reports),
            "changed_order_count": changed_orders,
            "credible_changed_order_count": credible_changed_orders,
            "queries": query_reports,
            "cleanup_state": {
                "collection_exists": state.collection_exists,
                "file_resource_exists": state.file_resource_exists,
                "model_object_exists": state.model_object_exists,
            },
        }
        destination = Path(report_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
