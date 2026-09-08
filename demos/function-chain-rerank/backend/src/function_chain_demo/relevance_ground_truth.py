"""Evaluation-only natural-intent ground truth for the six canonical queries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final

from function_chain_demo.catalog_curation import (
    BACKGROUND,
    INTRA_TYPE_HARD_NEGATIVE,
    PARTIAL,
    STRONG,
)

CROSS_CATEGORY_HARD_NEGATIVE: Final = "cross_category_hard_negative"
GROUND_TRUTH_FILE: Final = "relevance-ground-truth.json"

INTENT_SPECS: Final[dict[str, dict[str, str]]] = {
    "compact-dark-wood-desk": {
        "product_type": "DESK",
        "intent": "compact desk with a visibly dark wood or wood-look finish for a small office",
        "strong_rule": (
            "desk is compact and visibly dark brown, espresso, walnut, chestnut, or black wood-look"
        ),
        "partial_rule": "desk matches compact size but has a clear finish mismatch",
        "intra_type_hard_rule": (
            "desk subtype, size, or finish conflicts with compact dark-wood use"
        ),
    },
    "soft-neutral-living-room-rug": {
        "product_type": "RUG",
        "intent": "soft-looking living-room rug in a neutral color family",
        "strong_rule": (
            "rug is beige, ivory, cream, natural, taupe, grey, charcoal, or another clear neutral"
        ),
        "partial_rule": "rug has a neutral base plus a visible non-neutral color",
        "intra_type_hard_rule": "rug is prominently red, pink, blue, or otherwise non-neutral",
    },
    "red-weekend-backpack": {
        "product_type": "BACKPACK",
        "intent": "red backpack suitable for gym, casual travel, or weekend carrying",
        "strong_rule": (
            "backpack is visibly red-family and suitable for travel, sport, or everyday carrying"
        ),
        "partial_rule": "backpack use fits but its visible color does not",
        "intra_type_hard_rule": (
            "specialized child/tool use or a strong color mismatch conflicts with the intent"
        ),
    },
    "comfortable-on-ear-headphones": {
        "product_type": "HEADPHONES",
        "intent": "comfortable on-ear headphones for ordinary music listening",
        "strong_rule": "catalog and image show an on-ear everyday-listening form factor",
        "partial_rule": "headphones are everyday-capable but explicitly over-ear",
        "intra_type_hard_rule": (
            "in-ear, earbud, PlayStation chat, or single-ear products are not on-ear headphones"
        ),
    },
    "automatic-black-commuter-umbrella": {
        "product_type": "UMBRELLA",
        "intent": "black automatic folding umbrella suitable for a rain commute",
        "strong_rule": "umbrella is black, automatic, portable, and intended for rain/travel use",
        "partial_rule": "automatic portable umbrella has an explicit non-black color",
        "intra_type_hard_rule": "patio, market, beach, or golf umbrella is not a commuter umbrella",
    },
    "warm-bronze-modern-bed": {
        "product_type": "BED",
        "intent": "modern bed frame with a warm bronze-family metal finish",
        "strong_rule": (
            "bed frame is bronze, dark bronze, oiled bronze, or burnished bronze; a synthetic "
            "parent variant may use shared authored-image evidence from an explicit bronze-family child"
        ),
        "partial_rule": "bed frame uses gold or another clearly non-bronze finish",
        "intra_type_hard_rule": (
            "bunk, loft, child, dresser, or other coarse-type mismatch is not "
            "the requested bed frame"
        ),
    },
}


def build_ground_truth(
    queries: list[dict[str, object]], source_items: list[dict[str, object]]
) -> dict[str, object]:
    query_documents: list[dict[str, object]] = []
    for query in queries:
        query_id = str(query["id"])
        spec = INTENT_SPECS[query_id]
        judgments: list[dict[str, object]] = []
        for item in source_items:
            product_type = str(item["official_metadata"]["product_type"])
            if product_type == spec["product_type"]:
                label = str(item["selection_class"])
                reason = str(item["selection_reason"])
            else:
                label = CROSS_CATEGORY_HARD_NEGATIVE
                reason = (
                    f"{product_type} is a different product category from {spec['product_type']}"
                )
            judgments.append(
                {
                    "product_id": int(item["sequence"]),
                    "item_id": str(item["item_id"]),
                    "product_type": product_type,
                    "label": label,
                    "reason": reason,
                }
            )
        query_documents.append(
            {
                "query_id": query_id,
                "query_text": str(query["query_text"]),
                **spec,
                "visible_top6_accepted_labels": [STRONG],
                "judgments": judgments,
            }
        )
    return {
        "schema_version": 1,
        "evaluation_only": True,
        "production_ranking_consumers": [],
        "labels": {
            STRONG: "fully satisfies the natural shopping intent",
            PARTIAL: "same broad product with a material modifier mismatch",
            INTRA_TYPE_HARD_NEGATIVE: (
                "coarse same-type record with a disqualifying use, form, or color"
            ),
            BACKGROUND: (
                "same-type catalog filler added by the expansion pass; not an intent slice"
            ),
            CROSS_CATEGORY_HARD_NEGATIVE: "different product category",
        },
        "queries": query_documents,
    }


def load_ground_truth(root: Path) -> dict[str, object]:
    document = json.loads((root / GROUND_TRUTH_FILE).read_text(encoding="utf-8"))
    if (
        document.get("schema_version") != 1
        or document.get("evaluation_only") is not True
        or document.get("production_ranking_consumers") != []
    ):
        raise ValueError("Relevance ground truth boundary is invalid")
    queries = document.get("queries")
    if not isinstance(queries, list) or {query.get("query_id") for query in queries} != set(
        INTENT_SPECS
    ):
        raise ValueError("Relevance ground truth query coverage is invalid")
    for query in queries:
        judgments = query.get("judgments")
        if not isinstance(judgments, list) or len(judgments) != 240:
            raise ValueError("Relevance ground truth must judge every product")
        counts = {
            label: sum(judgment.get("label") == label for judgment in judgments)
            for label in (STRONG, PARTIAL, INTRA_TYPE_HARD_NEGATIVE)
        }
        if counts[STRONG] != 6 or counts[PARTIAL] < 1 or counts[INTRA_TYPE_HARD_NEGATIVE] < 1:
            raise ValueError("Relevance ground truth class coverage is invalid")
    return document
