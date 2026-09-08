"""Deterministic offline catalog curation for the six audience-facing intents."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

STRONG: Final = "strong"
PARTIAL: Final = "partial"
INTRA_TYPE_HARD_NEGATIVE: Final = "intra_type_hard_negative"
BACKGROUND: Final = "catalog_background"

TARGET_INTENT_TYPES: Final = (
    "RUG",
    "BACKPACK",
    "HEADPHONES",
    "BED",
    "DESK",
    "UMBRELLA",
)

SELECTION_LABEL_COUNTS: Final[dict[str, dict[str, int]]] = {
    product_type: {STRONG: 6, PARTIAL: 1, INTRA_TYPE_HARD_NEGATIVE: 1}
    for product_type in TARGET_INTENT_TYPES
}
SELECTION_LABEL_COUNTS["HEADPHONES"] = {
    STRONG: 6,
    PARTIAL: 1,
    INTRA_TYPE_HARD_NEGATIVE: 2,
}

# Five products are retained for every background type except PLANTER, then a second
# expansion pass adds six neutral background products to every one of the 20 types.
# Together with the intent slices above, these quotas preserve exactly 240 unique
# synthetic products.
TYPE_ITEM_COUNTS: Final[dict[str, int]] = {
    "SOFA": 11,
    "CHAIR": 11,
    "TABLE": 11,
    "LAMP": 11,
    "RUG": 14,
    "HANDBAG": 11,
    "BACKPACK": 14,
    "SHOES": 11,
    "BOOT": 11,
    "SANDAL": 11,
    "HAT": 11,
    "SUITCASE": 11,
    "HEADPHONES": 15,
    "DRINKING_CUP": 11,
    "PILLOW": 11,
    "BED": 14,
    "SHELF": 11,
    "DESK": 14,
    "PLANTER": 12,
    "UMBRELLA": 14,
}

DERIVED_ENGLISH_TITLE_PROVENANCE: Final = (
    "deterministic_english_summary_from_synthetic_authored_metadata"
)


@dataclass(frozen=True)
class CurationJudgment:
    label: str
    reason: str


def normalized_source_text(values: list[str]) -> str:
    return " ".join(values).casefold()


def _has_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)


def _compact_desk(title: str) -> bool:
    if _has_any(title, ("compact", "space-saving", "space saving", "foldable")):
        return True
    imperial = re.search(r"(\d+(?:\.\d+)?)\s*(?:\"|inch|inches|w\b)", title)
    if imperial and float(imperial.group(1)) <= 45.0:
        return True
    metric = re.search(r"(\d+(?:\.\d+)?)\s*x\s*\d+(?:\.\d+)?\s*x", title)
    return bool(metric and float(metric.group(1)) <= 115.0)


def classify_intent_candidate(
    product_type: str,
    title: str,
    source_text: str,
    *,
    bronze_family_image_evidence: bool = False,
) -> CurationJudgment | None:
    """Classify one same-type source record without using query IDs or item IDs."""
    title_text = title.casefold()
    text = f"{title_text} {source_text}"

    if product_type == "DESK":
        dark = _has_any(
            text,
            (
                "dark espresso",
                "antique espresso",
                "espresso",
                "walnut",
                "chestnut",
                "dark brown",
                "black",
            ),
        )
        wood = _has_any(
            text,
            ("wood", "pine", "mdf", "veneer", "walnut", "chestnut", "espresso", "particle board"),
        )
        incompatible = _has_any(text, ("gaming", "corner desk", "l-shaped", "l shaped"))
        compact = _compact_desk(title_text)
        if dark and wood and compact and not incompatible and "white" not in text:
            return CurationJudgment(
                STRONG, "compact dimensions and a dark wood or wood-look finish"
            )
        if compact and not incompatible:
            return CurationJudgment(PARTIAL, "compact desk with an explicit finish mismatch")
        if incompatible or "white" in text:
            return CurationJudgment(
                INTRA_TYPE_HARD_NEGATIVE,
                "desk subtype or size/finish conflicts with the compact dark-wood intent",
            )
        return None

    if product_type == "RUG":
        neutral = _has_any(
            text,
            (
                "neutral",
                "beige",
                "ivory",
                "cream",
                "natural",
                "taupe",
                "sand",
                "grey",
                "gray",
                "charcoal",
                "off white",
            ),
        )
        conflict = _has_any(text, ("red", "pink", "orange", "aqua", "green", "yellow", "vibrant"))
        blue = "blue" in text
        if neutral and not conflict and not blue:
            return CurationJudgment(STRONG, "rug has an explicit neutral color family")
        if neutral and blue and not conflict:
            return CurationJudgment(PARTIAL, "neutral base includes a visible blue color conflict")
        if conflict or blue:
            return CurationJudgment(
                INTRA_TYPE_HARD_NEGATIVE, "rug has a prominent non-neutral color conflict"
            )
        return None

    if product_type == "BACKPACK":
        red = bool(re.search(r"\b(?:red|maroon|burgundy|crimson)\b", text))
        title_color_conflict = _has_any(
            title_text,
            ("grey", "gray", "yellow", "blue", "purple", "salmon", "orange", "pink", "black"),
        )
        child_or_tool = _has_any(text, ("kids", "children", "tool bag", "small backpack, yellow"))
        trip_use = _has_any(
            text,
            ("travel", "weekend", "overnight", "sports", "hiking", "casual", "laptop", "classic"),
        )
        if red and trip_use and not child_or_tool and not title_color_conflict:
            return CurationJudgment(
                STRONG, "red backpack suitable for travel, sport, or weekend carrying"
            )
        if trip_use and not child_or_tool:
            return CurationJudgment(
                PARTIAL, "trip-suitable backpack with an explicit color mismatch"
            )
        if child_or_tool or _has_any(text, ("yellow", "orange", "pink")):
            return CurationJudgment(
                INTRA_TYPE_HARD_NEGATIVE,
                "backpack color or specialized use conflicts with the red weekend intent",
            )
        return None

    if product_type == "HEADPHONES":
        hard = _has_any(
            text, ("in-ear", "in ear", "earbud", "playstation", "chat headset", "single ear")
        )
        kids = _has_any(text, ("for kids", "children", "volume limited"))
        on_ear = _has_any(text, ("on-ear", "on ear", "supra-aural"))
        over_ear = _has_any(text, ("over-ear", "over ear"))
        if on_ear and not hard and not kids and "usb headset" not in text:
            return CurationJudgment(STRONG, "explicit on-ear form factor for everyday listening")
        if over_ear and not hard and not kids:
            return CurationJudgment(
                PARTIAL, "everyday headphones with an over-ear form-factor mismatch"
            )
        if hard:
            return CurationJudgment(
                INTRA_TYPE_HARD_NEGATIVE,
                "in-ear or chat-only form factor conflicts with everyday on-ear listening",
            )
        return None

    if product_type == "UMBRELLA":
        hard_use = _has_any(text, ("patio", "market", "cantilever", "beach", "golf"))
        automatic = _has_any(
            text,
            (
                "automatic",
                "automático",
                "automática",
                "automatico",
                "automatische",
                "自動",
                "ワンタッチ",
            ),
        )
        black = _has_any(text, ("black", "nero", "negro", "schwarz", "ブラック", "黒色", " blk"))
        portable = _has_any(
            text,
            (
                "travel",
                "compact",
                "folding",
                "foldable",
                "plegable",
                "pieghevole",
                "portatile",
                "portátil",
                "compatto",
                "折りたたみ",
            ),
        )
        if automatic and black and portable and not hard_use:
            return CurationJudgment(
                STRONG, "black automatic folding umbrella suitable for commuting"
            )
        if automatic and portable and not hard_use:
            return CurationJudgment(
                PARTIAL, "automatic travel umbrella with an explicit color mismatch"
            )
        if hard_use:
            return CurationJudgment(
                INTRA_TYPE_HARD_NEGATIVE,
                "patio, beach, or golf use conflicts with a commuter umbrella",
            )
        return None

    if product_type == "BED":
        is_bed = _has_any(title_text, (" bed", "bed ", "bed frame")) and "dresser" not in title_text
        hard = _has_any(text, ("bunk bed", "loft bed", "kids bed", "dresser"))
        bronze = _has_any(text, ("bronze", "burnished bronze", "oiled bronze", "dark bronze"))
        gold = bool(re.search(r"\bgold\b", text))
        warm_modern = _has_any(
            text, ("modern", "contemporary", "walnut", "mahogany", "espresso", "warm", "brown")
        )
        if is_bed and (bronze or bronze_family_image_evidence) and not gold and "blue" not in text:
            reason = "bronze bed frame"
            if bronze_family_image_evidence and not bronze:
                reason = (
                    "bed variant shares its authored main image with an explicit bronze-family "
                    "catalog bed listing"
                )
            return CurationJudgment(STRONG, reason)
        if is_bed and gold and not hard:
            return CurationJudgment(
                PARTIAL, "gold metal finish conflicts with the requested bronze-family finish"
            )
        if is_bed and warm_modern and not hard:
            return CurationJudgment(PARTIAL, "modern warm bed with a non-bronze finish")
        if hard:
            return CurationJudgment(
                INTRA_TYPE_HARD_NEGATIVE,
                "bunk, loft, child, or non-bed subtype conflicts with the modern bed-frame intent",
            )
        return None

    return None


def derived_umbrella_title(brand: str | None, source_text: str) -> str:
    """Build a transparent English attribute summary from synthetic authored fields."""
    ascii_brand_parts = re.findall(r"[A-Za-z][A-Za-z0-9&.-]{2,}", brand or "")
    parts = [ascii_brand_parts[-1] if ascii_brand_parts else "Synthetic"]
    if "12" in source_text and _has_any(source_text, ("rib", "ribs", "伞骨")):
        parts.append("12-Rib")
    if _has_any(source_text, ("vent", "通風孔", "通风孔", "ventilatore")):
        parts.append("Wind-Vented")
    parts.extend(("Automatic", "Black", "Folding Travel Umbrella"))
    return " ".join(parts)


def curation_contract() -> dict[str, object]:
    return {
        "execution": "offline catalog preparation only",
        "production_ranking_consumers": [],
        "item_id_rules": False,
        "query_id_rules": False,
        "target_intent_types": list(TARGET_INTENT_TYPES),
        "type_item_counts": dict(TYPE_ITEM_COUNTS),
        "selection_label_counts": {
            product_type: dict(counts) for product_type, counts in SELECTION_LABEL_COUNTS.items()
        },
        "labels": [STRONG, PARTIAL, INTRA_TYPE_HARD_NEGATIVE, BACKGROUND],
        "derived_english_title_provenance": DERIVED_ENGLISH_TITLE_PROVENANCE,
    }
