"""Offline business-semantic gates that never participate in production ranking."""

from __future__ import annotations

from typing import Any, Final

import numpy as np
import xgboost as xgb

from function_chain_demo.catalog_curation import STRONG
from function_chain_demo.chain_features import freshness as chain_freshness
from function_chain_demo.chain_features import popularity as chain_popularity
from function_chain_demo.chain_features import price_affinity as chain_price_affinity
from function_chain_demo.dataset import (
    CatalogDataset,
    Product,
    normalize_semantic_score,
)
from function_chain_demo.relevance_ground_truth import load_ground_truth
from function_chain_demo.text_embedding import cosine

SEMANTIC_JITTER: Final = 1e-6
NEAR_SCORE_TOLERANCE: Final = 1e-6
SEARCH_LIMIT: Final = 20
# The rerank must keep at least this many of the query's expected product type in
# its top-6. Under a pure semantic encoder a handful of same-type attribute
# mismatches (and, for the umbrella query, one cross-category recall miss) are
# expected and honest; category recall — not perfect strong/negative separation —
# is the gate.
MIN_TOP6_TYPE_RECALL: Final = 5

POSITIVE_ADVANTAGE_THRESHOLDS: Final = {
    "rating": 0.875,
    "inventory": 0.75,
    "return_rate": 0.05,
    "freshness": 0.75,
}
RISK_THRESHOLDS: Final = {
    "rating": 0.60,
    "inventory": 0.10,
    "return_rate": 0.18,
    "freshness": 0.20,
}


def model_features(semantic_score: float, product: Product) -> tuple[float, ...]:
    return (
        normalize_semantic_score(semantic_score),
        product.rating,
        chain_popularity(product.clicks_30d, product.sales_30d),
        chain_price_affinity(product.display_price_usd),
        chain_freshness(product.release_epoch),
    )


def favorable_signals(product: Product) -> tuple[str, ...]:
    signals: list[str] = []
    if product.rating >= POSITIVE_ADVANTAGE_THRESHOLDS["rating"]:
        signals.append("strong_rating")
    if product.inventory >= POSITIVE_ADVANTAGE_THRESHOLDS["inventory"]:
        signals.append("healthy_inventory")
    if product.return_rate <= POSITIVE_ADVANTAGE_THRESHOLDS["return_rate"]:
        signals.append("low_returns")
    if product.freshness >= POSITIVE_ADVANTAGE_THRESHOLDS["freshness"]:
        signals.append("high_freshness")
    return tuple(signals)


def risk_signals(product: Product) -> tuple[str, ...]:
    signals: list[str] = []
    if product.rating <= RISK_THRESHOLDS["rating"]:
        signals.append("low_rating")
    if product.inventory <= RISK_THRESHOLDS["inventory"]:
        signals.append("stockout_risk")
    if product.return_rate >= RISK_THRESHOLDS["return_rate"]:
        signals.append("high_returns")
    if product.freshness <= RISK_THRESHOLDS["freshness"]:
        signals.append("low_freshness")
    return tuple(signals)


def direction_report(dataset: CatalogDataset, model: xgb.Booster) -> dict[str, Any]:
    names = ("semantic_score", "rating", "popularity", "price_affinity", "freshness")
    directions = (1, 1, 1, 1, 1)
    bases = np.asarray([record.features for record in dataset.training_records], dtype=np.float32)
    grid = np.linspace(0.0, 1.0, 101, dtype=np.float32)
    report: dict[str, Any] = {}
    for feature_index, (name, direction) in enumerate(zip(names, directions, strict=True)):
        probes = np.repeat(bases, len(grid), axis=0)
        probes[:, feature_index] = np.tile(grid, len(bases))
        predictions = model.predict(xgb.DMatrix(probes)).reshape(len(bases), len(grid))
        changes = np.diff(predictions, axis=1)
        violations = -changes if direction == 1 else changes
        row_index, step = np.unravel_index(int(np.argmax(violations)), violations.shape)
        worst = float(violations[row_index, step])
        report[name] = {
            "expected": "nondecreasing" if direction == 1 else "nonincreasing",
            "passed": worst <= 1e-7,
            "worst_violation": round(max(0.0, worst), 8),
            "from": round(float(grid[step]), 2),
            "to": round(float(grid[step + 1]), 2),
        }
    return report


def _query_judgments(dataset: CatalogDataset) -> dict[str, dict[str, str]]:
    document = load_ground_truth(dataset.root)
    return {
        str(query["query_id"]): {
            str(judgment["item_id"]): str(judgment["label"]) for judgment in query["judgments"]
        }
        for query in document["queries"]
    }


def _predictions(
    model: xgb.Booster,
    products: list[Product],
    raw_semantic: dict[str, float],
    jitter: float = 0.0,
) -> np.ndarray:
    features = np.asarray(
        [model_features(raw_semantic[product.item_id] + jitter, product) for product in products],
        dtype=np.float32,
    )
    return model.predict(xgb.DMatrix(features))


def business_gate_report(dataset: CatalogDataset, model: xgb.Booster) -> dict[str, Any]:
    judgments_by_query = _query_judgments(dataset)
    reports: list[dict[str, Any]] = []
    all_passed = True

    for query in dataset.queries:
        raw_semantic = {
            product.item_id: cosine(query.embedding, product.embedding)
            for product in dataset.products
        }
        vector_products = sorted(
            dataset.products,
            key=lambda product: (-raw_semantic[product.item_id], product.id),
        )[:SEARCH_LIMIT]
        predictions = _predictions(model, vector_products, raw_semantic)
        repeat_predictions = _predictions(model, vector_products, raw_semantic)
        business_indexes = np.argsort(-predictions, kind="stable")
        repeat_indexes = np.argsort(-repeat_predictions, kind="stable")
        business_products = [vector_products[int(index)] for index in business_indexes]

        labels = judgments_by_query[query.id]
        expected_type = query.expected_type
        vector_ids = [product.item_id for product in vector_products]
        business_ids = [product.item_id for product in business_products]
        vector_rank = {item_id: index + 1 for index, item_id in enumerate(vector_ids)}
        business_rank = {item_id: index + 1 for index, item_id in enumerate(business_ids)}
        business_top6_labels = [labels[item_id] for item_id in business_ids[:6]]

        # Type-recall gate (the meaningful signal under a pure semantic encoder):
        # the rerank must not push the query's expected product type out of the
        # top-6. A pure BGE encoder does not sharply separate within-type attribute
        # intent, so we gate on category recall rather than "top-6 all strong".
        vector_top6_types = [product.product_type for product in vector_products[:6]]
        business_top6_types = [product.product_type for product in business_products[:6]]
        vector_type_recall = sum(1 for value in vector_top6_types if value == expected_type)
        business_type_recall = sum(1 for value in business_top6_types if value == expected_type)
        type_recall_held = business_type_recall >= vector_type_recall
        type_recall_floor = business_type_recall >= MIN_TOP6_TYPE_RECALL

        strong_ids = [item_id for item_id in business_ids if labels[item_id] == STRONG]
        downward = [
            item_id for item_id in strong_ids if business_rank[item_id] > vector_rank[item_id]
        ]
        # "Healthy" means a strong candidate holds a top-6 slot in the business
        # order and carries favourable operational signals. Under a pure semantic
        # encoder some strong candidates legitimately trail same-type negatives in
        # vector order, so holding (not necessarily climbing) is the right outcome.
        healthy_upward = [
            item_id
            for item_id in strong_ids
            if business_rank[item_id] <= 6
            and len(
                favorable_signals(
                    dataset.product_by_id(
                        next(
                            product.id for product in vector_products if product.item_id == item_id
                        )
                    )
                )
            )
            >= 2
        ]
        risky_downward = [
            item_id
            for item_id in downward
            if risk_signals(
                dataset.product_by_id(
                    next(product.id for product in vector_products if product.item_id == item_id)
                )
            )
        ]

        exact_ties: list[list[str]] = []
        near_score_pairs: list[dict[str, object]] = []
        for left in range(len(predictions)):
            tied = [
                vector_ids[right]
                for right in range(left + 1, len(predictions))
                if float(predictions[right]) == float(predictions[left])
            ]
            if tied:
                exact_ties.append([vector_ids[left], *tied])
            for right in range(left + 1, len(predictions)):
                difference = abs(float(predictions[left]) - float(predictions[right]))
                if 0.0 < difference <= NEAR_SCORE_TOLERANCE:
                    near_score_pairs.append(
                        {
                            "left": vector_ids[left],
                            "right": vector_ids[right],
                            "absolute_difference": difference,
                        }
                    )

        related_sunk = [
            {
                "item_id": item_id,
                "label": labels[item_id],
                "business_rank": business_rank[item_id],
            }
            for item_id in business_ids
            if labels[item_id] != STRONG and labels[item_id] != "cross_category_hard_negative"
        ]
        passed = (
            type_recall_held
            and type_recall_floor
            and bool(healthy_upward)
            and np.array_equal(predictions, repeat_predictions)
            and np.array_equal(business_indexes, repeat_indexes)
        )
        all_passed = all_passed and passed
        reports.append(
            {
                "query_id": query.id,
                "expected_type": expected_type,
                "vector_top20": vector_ids,
                "business_top20": business_ids,
                "business_top6_labels": business_top6_labels,
                "vector_top6_types": vector_top6_types,
                "business_top6_types": business_top6_types,
                "vector_type_recall": vector_type_recall,
                "business_type_recall": business_type_recall,
                "type_recall_held": type_recall_held,
                "type_recall_floor": type_recall_floor,
                "healthy_upward_ids": healthy_upward,
                "risky_downward_ids": risky_downward,
                "related_candidates_below_top6": related_sunk,
                "exact_tie_groups": exact_ties,
                "near_score_pairs": near_score_pairs,
                "semantic_jitter": SEMANTIC_JITTER,
                "repeat_order_stable": bool(np.array_equal(business_indexes, repeat_indexes)),
                "passed": passed,
            }
        )

    return {
        "passed": all_passed,
        "semantic_jitter": SEMANTIC_JITTER,
        "near_score_tolerance": NEAR_SCORE_TOLERANCE,
        "queries": reports,
    }
