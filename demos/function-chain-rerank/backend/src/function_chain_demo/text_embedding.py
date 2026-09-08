"""Deterministic relevance-guarded hashed TF-IDF embeddings for synthetic catalog text."""

from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Final

EMBEDDING_DIMENSION: Final = 512
EMBEDDING_SEED: Final = "synthetic-commerce-catalog-r1-hashed-tfidf-v4"
EMBEDDING_VERSION: Final = "type-and-attribute-anchor-hashed-tfidf-sha256-word-1-2-v4"
TOKEN_PATTERN: Final = r"[a-z0-9]+"
_TOKEN_RE = re.compile(TOKEN_PATTERN)

SEMANTIC_TYPE_ALIASES: Final[dict[str, tuple[str, ...]]] = {
    "SOFA": ("sofa", "couch", "loveseat"),
    "CHAIR": ("chair", "armchair"),
    "TABLE": ("table", "dining table", "coffee table"),
    "LAMP": ("lamp", "lighting"),
    "RUG": ("rug", "carpet"),
    "HANDBAG": ("handbag", "purse", "shoulder bag", "tote bag"),
    "BACKPACK": ("backpack", "rucksack"),
    "SHOES": ("shoes", "shoe", "sneakers", "pumps"),
    "BOOT": ("boots", "boot"),
    "SANDAL": ("sandals", "sandal"),
    "HAT": ("hat", "cap"),
    "SUITCASE": ("suitcase", "luggage"),
    "HEADPHONES": ("headphones", "headphone", "headset", "earphones", "earphone"),
    "DRINKING_CUP": ("drinking cup", "cup", "mug", "tumbler"),
    "PILLOW": ("pillow", "cushion"),
    "BED": ("bed frame", "bed"),
    "SHELF": ("shelf", "shelving", "bookcase"),
    "DESK": ("desk", "workstation"),
    "PLANTER": ("planter", "plant pot", "flower pot"),
    "UMBRELLA": ("umbrella", "parasol"),
}
SEMANTIC_ATTRIBUTE_ALIASES: Final[dict[str, tuple[str, ...]]] = {
    "DARK_WOOD_FINISH": (
        "dark wood",
        "dark espresso",
        "antique espresso",
        "espresso",
        "walnut",
        "chestnut",
        "dark brown",
        "warm brown wood",
    ),
    "NEUTRAL_COLOR": (
        "neutral",
        "beige",
        "ivory",
        "cream",
        "natural",
        "taupe",
        "grey",
        "gray",
        "charcoal",
        "off white",
        "chestnut",
    ),
    "RED_COLOR": ("red", "maroon", "burgundy", "crimson"),
    "PINK_COLOR": ("pink", "blush"),
    "BLUE_COLOR": ("blue", "navy", "aqua", "turquoise"),
    "WHITE_COLOR": ("white", "high gloss white"),
    "YELLOW_COLOR": ("yellow", "mustard"),
    "GREEN_COLOR": ("green", "olive", "seafoam"),
    "ON_EAR_FORM": ("on ear", "on-ear", "supra aural", "supra-aural"),
    "BLACK_AUTOMATIC_TRAVEL_UMBRELLA": (
        "automatic black umbrella",
        "automatic black folding travel umbrella",
        "automatic travel umbrella black",
    ),
    "WARM_METAL_BED_FINISH": (
        "bronze",
        "burnished bronze",
        "oiled bronze",
        "dark bronze",
    ),
}
SEMANTIC_ANCHOR_WEIGHT: Final = 64.0
SEMANTIC_ATTRIBUTE_WEIGHT: Final = 48.0
HASH_DIMENSION_OFFSET: Final = len(SEMANTIC_TYPE_ALIASES) + len(SEMANTIC_ATTRIBUTE_ALIASES)
HASHED_DIMENSIONS: Final = EMBEDDING_DIMENSION - HASH_DIMENSION_OFFSET

FIELD_WEIGHTS: Final[dict[str, float]] = {
    "title": 3.0,
    "product_type": 2.5,
    "brand": 1.0,
    "color": 2.0,
    "material": 1.5,
    "style": 1.25,
    "node_name": 1.0,
    "description": 0.75,
    "bullet_points": 1.0,
}
COLOR_ATTRIBUTES: Final = {
    "DARK_WOOD_FINISH",
    "NEUTRAL_COLOR",
    "RED_COLOR",
    "PINK_COLOR",
    "BLUE_COLOR",
    "WHITE_COLOR",
    "YELLOW_COLOR",
    "GREEN_COLOR",
    "WARM_METAL_BED_FINISH",
    "BLACK_AUTOMATIC_TRAVEL_UMBRELLA",
}
NON_NEUTRAL_COLOR_ATTRIBUTES: Final = {
    "RED_COLOR",
    "PINK_COLOR",
    "BLUE_COLOR",
    "YELLOW_COLOR",
    "GREEN_COLOR",
    "BLACK_AUTOMATIC_TRAVEL_UMBRELLA",
}


def _words(text: str) -> tuple[str, ...]:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return tuple(_TOKEN_RE.findall(normalized))


def _tokens(text: str) -> tuple[str, ...]:
    words = _words(text)
    unigrams = [f"u:{word}" for word in words]
    bigrams = [f"b:{left}_{right}" for left, right in zip(words, words[1:])]
    return tuple(unigrams + bigrams)


def _bucket(token: str) -> tuple[int, float]:
    digest = hashlib.sha256(f"{EMBEDDING_SEED}:{token}".encode()).digest()
    index = HASH_DIMENSION_OFFSET + int.from_bytes(digest[:8], "big") % HASHED_DIMENSIONS
    sign = 1.0 if digest[8] & 1 else -1.0
    return index, sign


def _weighted_buckets(fields: Mapping[str, object]) -> dict[int, float]:
    values: dict[int, float] = {}
    product_type = fields.get("product_type")
    if isinstance(product_type, str) and product_type in SEMANTIC_TYPE_ALIASES:
        anchor_index = tuple(SEMANTIC_TYPE_ALIASES).index(product_type)
        values[anchor_index] = SEMANTIC_ANCHOR_WEIGHT
    semantic_texts: list[str] = []
    title_texts: list[str] = []
    for field_name, weight in FIELD_WEIGHTS.items():
        raw_value = fields.get(field_name)
        if raw_value is None:
            continue
        texts = raw_value if isinstance(raw_value, (list, tuple)) else (raw_value,)
        for raw_text in texts:
            if not isinstance(raw_text, str) or not raw_text.strip():
                continue
            semantic_texts.append(raw_text)
            if field_name == "title":
                title_texts.append(raw_text)
            for token in _tokens(raw_text):
                index, sign = _bucket(token)
                values[index] = values.get(index, 0.0) + sign * weight
    semantic_text = " ".join(semantic_texts)
    attributes = set(semantic_attribute_types(semantic_text))
    title_color_attributes = set(semantic_attribute_types(" ".join(title_texts))) & COLOR_ATTRIBUTES
    if title_color_attributes:
        attributes -= COLOR_ATTRIBUTES
        attributes |= title_color_attributes
    if attributes & NON_NEUTRAL_COLOR_ATTRIBUTES:
        attributes.discard("NEUTRAL_COLOR")
    for attribute in attributes:
        anchor_index = len(SEMANTIC_TYPE_ALIASES) + tuple(SEMANTIC_ATTRIBUTE_ALIASES).index(
            attribute
        )
        values[anchor_index] = SEMANTIC_ATTRIBUTE_WEIGHT
    return values


def _query_buckets(query_text: str) -> dict[int, float]:
    values: dict[int, float] = {}
    for product_type in semantic_anchor_types(query_text):
        anchor_index = tuple(SEMANTIC_TYPE_ALIASES).index(product_type)
        values[anchor_index] = SEMANTIC_ANCHOR_WEIGHT
    for attribute in semantic_attribute_types(query_text):
        anchor_index = len(SEMANTIC_TYPE_ALIASES) + tuple(SEMANTIC_ATTRIBUTE_ALIASES).index(
            attribute
        )
        values[anchor_index] = SEMANTIC_ATTRIBUTE_WEIGHT
    for token in _tokens(query_text):
        index, sign = _bucket(token)
        values[index] = values.get(index, 0.0) + sign
    return values


def semantic_anchor_types(text: str) -> tuple[str, ...]:
    """Return product types matched by fixed token-boundary aliases, never query IDs."""
    words = _words(text)
    matches: list[str] = []
    for product_type, aliases in SEMANTIC_TYPE_ALIASES.items():
        for alias in aliases:
            alias_words = _words(alias)
            if any(
                words[index : index + len(alias_words)] == alias_words
                for index in range(len(words) - len(alias_words) + 1)
            ):
                matches.append(product_type)
                break
    return tuple(matches)


def semantic_attribute_types(text: str) -> tuple[str, ...]:
    """Return generic shopping attributes matched by fixed token-boundary aliases."""
    words = _words(text)
    matches: list[str] = []
    for attribute, aliases in SEMANTIC_ATTRIBUTE_ALIASES.items():
        for alias in aliases:
            alias_words = _words(alias)
            if any(
                words[index : index + len(alias_words)] == alias_words
                for index in range(len(words) - len(alias_words) + 1)
            ):
                matches.append(attribute)
                break
    return tuple(matches)


@dataclass(frozen=True)
class EmbeddingContract:
    dimension: int
    version: str
    seed: str
    token_pattern: str
    ngrams: tuple[int, ...]
    field_weights: dict[str, float]
    hash_dimension_offset: int
    hashed_dimensions: int
    semantic_anchor_weight: float
    semantic_type_aliases: dict[str, tuple[str, ...]]
    semantic_attribute_weight: float
    semantic_attribute_aliases: dict[str, tuple[str, ...]]
    idf: tuple[float, ...]

    def public_metadata(self) -> dict[str, object]:
        return {
            "dimension": self.dimension,
            "version": self.version,
            "seed": self.seed,
            "token_pattern": self.token_pattern,
            "ngrams": list(self.ngrams),
            "field_weights": dict(self.field_weights),
            "hash_dimension_offset": self.hash_dimension_offset,
            "hashed_dimensions": self.hashed_dimensions,
            "semantic_anchor_weight": self.semantic_anchor_weight,
            "semantic_type_aliases": {
                name: list(aliases) for name, aliases in self.semantic_type_aliases.items()
            },
            "semantic_attribute_weight": self.semantic_attribute_weight,
            "semantic_attribute_aliases": {
                name: list(aliases) for name, aliases in self.semantic_attribute_aliases.items()
            },
        }

    def as_json(self) -> dict[str, object]:
        return {**self.public_metadata(), "idf": list(self.idf)}


def build_contract(documents: Iterable[Mapping[str, object]]) -> EmbeddingContract:
    document_buckets = [set(_weighted_buckets(document)) for document in documents]
    if not document_buckets:
        raise ValueError("At least one catalog document is required")
    document_count = len(document_buckets)
    frequencies = [0] * EMBEDDING_DIMENSION
    for indexes in document_buckets:
        for index in indexes:
            frequencies[index] += 1
    idf = tuple(
        round(math.log((1.0 + document_count) / (1.0 + frequency)) + 1.0, 10) if frequency else 0.0
        for frequency in frequencies
    )
    return EmbeddingContract(
        dimension=EMBEDDING_DIMENSION,
        version=EMBEDDING_VERSION,
        seed=EMBEDDING_SEED,
        token_pattern=TOKEN_PATTERN,
        ngrams=(1, 2),
        field_weights=dict(FIELD_WEIGHTS),
        hash_dimension_offset=HASH_DIMENSION_OFFSET,
        hashed_dimensions=HASHED_DIMENSIONS,
        semantic_anchor_weight=SEMANTIC_ANCHOR_WEIGHT,
        semantic_type_aliases=dict(SEMANTIC_TYPE_ALIASES),
        semantic_attribute_weight=SEMANTIC_ATTRIBUTE_WEIGHT,
        semantic_attribute_aliases=dict(SEMANTIC_ATTRIBUTE_ALIASES),
        idf=idf,
    )


def load_contract(value: Mapping[str, object]) -> EmbeddingContract:
    expected_keys = {
        "dimension",
        "version",
        "seed",
        "token_pattern",
        "ngrams",
        "field_weights",
        "hash_dimension_offset",
        "hashed_dimensions",
        "semantic_anchor_weight",
        "semantic_type_aliases",
        "semantic_attribute_weight",
        "semantic_attribute_aliases",
        "idf",
    }
    if set(value) != expected_keys:
        raise ValueError("Embedding contract keys are invalid")
    contract = EmbeddingContract(
        dimension=int(value["dimension"]),
        version=str(value["version"]),
        seed=str(value["seed"]),
        token_pattern=str(value["token_pattern"]),
        ngrams=tuple(int(item) for item in value["ngrams"]),  # type: ignore[arg-type]
        field_weights={
            str(name): float(weight)
            for name, weight in dict(value["field_weights"]).items()  # type: ignore[arg-type]
        },
        hash_dimension_offset=int(value["hash_dimension_offset"]),
        hashed_dimensions=int(value["hashed_dimensions"]),
        semantic_anchor_weight=float(value["semantic_anchor_weight"]),
        semantic_type_aliases={
            str(name): tuple(str(alias) for alias in aliases)
            for name, aliases in dict(value["semantic_type_aliases"]).items()  # type: ignore[arg-type]
        },
        semantic_attribute_weight=float(value["semantic_attribute_weight"]),
        semantic_attribute_aliases={
            str(name): tuple(str(alias) for alias in aliases)
            for name, aliases in dict(value["semantic_attribute_aliases"]).items()  # type: ignore[arg-type]
        },
        idf=tuple(float(item) for item in value["idf"]),  # type: ignore[arg-type]
    )
    if (
        contract.dimension != EMBEDDING_DIMENSION
        or contract.version != EMBEDDING_VERSION
        or contract.seed != EMBEDDING_SEED
        or contract.token_pattern != TOKEN_PATTERN
        or contract.ngrams != (1, 2)
        or contract.field_weights != FIELD_WEIGHTS
        or contract.hash_dimension_offset != HASH_DIMENSION_OFFSET
        or contract.hashed_dimensions != HASHED_DIMENSIONS
        or contract.semantic_anchor_weight != SEMANTIC_ANCHOR_WEIGHT
        or contract.semantic_type_aliases != SEMANTIC_TYPE_ALIASES
        or contract.semantic_attribute_weight != SEMANTIC_ATTRIBUTE_WEIGHT
        or contract.semantic_attribute_aliases != SEMANTIC_ATTRIBUTE_ALIASES
        or len(contract.idf) != EMBEDDING_DIMENSION
        or any(not math.isfinite(value) or value < 0.0 for value in contract.idf)
    ):
        raise ValueError("Embedding contract does not match the fixed implementation")
    return contract


def _normalize(values: dict[int, float], contract: EmbeddingContract) -> tuple[float, ...]:
    vector = [0.0] * contract.dimension
    for index, value in values.items():
        if value == 0.0 or contract.idf[index] == 0.0:
            continue
        magnitude = 1.0 + math.log(abs(value))
        vector[index] = math.copysign(magnitude * contract.idf[index], value)
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        raise ValueError("Text has no terms represented by the catalog contract")
    return tuple(round(value / norm, 8) for value in vector)


def encode_document(fields: Mapping[str, object], contract: EmbeddingContract) -> tuple[float, ...]:
    return _normalize(_weighted_buckets(fields), contract)


def encode_query(query_text: str, contract: EmbeddingContract) -> tuple[float, ...]:
    if not query_text.strip():
        raise ValueError("Query text cannot be empty")
    return _normalize(_query_buckets(query_text), contract)


def cosine(left: Iterable[float], right: Iterable[float]) -> float:
    left_values = tuple(left)
    right_values = tuple(right)
    if len(left_values) != len(right_values):
        raise ValueError("Cosine vectors must have the same dimension")
    return round(sum(a * b for a, b in zip(left_values, right_values, strict=True)), 8)
