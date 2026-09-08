"""Generate and strictly validate the fixed synthetic commerce catalog revision."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Final

from function_chain_demo.synthetic_source import (
    SELECTION_VERSION,
    STATIC_OBJECTS,
    SYNTHETIC_IDENTITY_FIELD,
    SYNTHETIC_IMAGE_FIELD,
    SYNTHETIC_METADATA_FIELD,
    canonical_json,
    jpeg_dimensions,
    selection_rule,
    sha256_file,
)
from function_chain_demo.synthetic_catalog import build_items
from function_chain_demo.business_profiles import (
    ITEM_PROFILE_NAMES,
    PROFILE_TEMPLATES,
    PROFILE_VERSION,
    operational_profile,
)
from function_chain_demo.catalog_curation import (
    DERIVED_ENGLISH_TITLE_PROVENANCE,
    TYPE_ITEM_COUNTS,
)
from function_chain_demo.chain_features import freshness as chain_freshness
from function_chain_demo.chain_features import popularity as chain_popularity
from function_chain_demo.chain_features import price_affinity as chain_price_affinity
from function_chain_demo.embedding_onnx import EMBEDDING_DIMENSION as EMBEDDING_DIMENSION
from function_chain_demo.embedding_onnx import (
    EmbeddingContract,
    build_contract,
    cosine,
    encode_document,
    encode_query,
    load_contract,
)
from function_chain_demo.relevance_ground_truth import (
    GROUND_TRUTH_FILE,
    build_ground_truth,
    load_ground_truth,
)

DATASET_ID: Final = "synthetic-commerce-catalog"
DATASET_REVISION: Final = "synthetic-commerce-catalog-r1"
DATASET_SEED: Final = 20260820
DATASET_AS_OF_DATE: Final = date(2026, 8, 20)
GENERATOR_VERSION: Final = "3.2.0"
IMAGE_MIME: Final = "image/jpeg"
SEMANTIC_SCORE_DECIMALS: Final = 5
FEATURE_NAMES: Final[tuple[str, ...]] = (
    "semantic_score",
    "rating",
    "popularity",
    "price_affinity",
    "freshness",
)
DEFAULT_DATASET_ROOT: Final = Path(__file__).resolve().parents[3] / "data" / DATASET_REVISION

SOURCE_FIELD = SYNTHETIC_METADATA_FIELD
SOURCE_IDENTITY_FIELD = SYNTHETIC_IDENTITY_FIELD
SOURCE_IMAGE_FIELD = SYNTHETIC_IMAGE_FIELD
SIMULATED_FIELD = "deterministic_simulated_operational_signal"
DERIVED_SIMULATED_FIELD = "derived_from_deterministic_simulated_operational_signal"
DERIVED_TEXT_FIELD = "derived_from_synthetic_authored_metadata"
DERIVED_LOCALIZED_TITLE_FIELD = DERIVED_ENGLISH_TITLE_PROVENANCE

OFFICIAL_FIELDS: Final = (
    "title",
    "product_type",
    "brand",
    "color",
    "material",
    "style",
    "node_name",
    "description",
    "bullet_points",
)
SIMULATED_FIELDS: Final = (
    "display_price_usd",
    "rating_value",
    "clicks_30d",
    "sales_30d",
    "inventory_units",
    "inventory_capacity",
    "return_rate",
    "release_date",
)
DERIVED_SIMULATED_FIELDS: Final = ("rating", "inventory", "freshness", "release_epoch")
LABEL_WEIGHTS: Final[tuple[float, ...]] = (0.80, 0.02, 0.08, 0.06, 0.04)

QUERY_SPECS: Final = (
    {
        "id": "compact-dark-wood-desk",
        "query_text": "a compact dark wood desk for a small home office",
        "story": "Find a space-conscious desk using only natural shopping language.",
        "split": "train",
        "expected_type": "DESK",
    },
    {
        "id": "soft-neutral-living-room-rug",
        "query_text": "a soft neutral rug for a cozy living room",
        "story": "Find a visually neutral living-room rug using catalog text.",
        "split": "train",
        "expected_type": "RUG",
    },
    {
        "id": "red-weekend-backpack",
        "query_text": "a red backpack for the gym and weekend trips",
        "story": "Find a red multipurpose backpack without exposing evaluation keys.",
        "split": "train",
        "expected_type": "BACKPACK",
    },
    {
        "id": "comfortable-on-ear-headphones",
        "query_text": "comfortable on-ear headphones for everyday music",
        "story": "Find comfortable everyday headphones from natural text.",
        "split": "validation",
        "expected_type": "HEADPHONES",
    },
    {
        "id": "automatic-black-commuter-umbrella",
        "query_text": "a sturdy automatic black umbrella for commuting in heavy rain",
        "story": "Find a sturdy commuter umbrella using natural shopping language.",
        "split": "train",
        "expected_type": "UMBRELLA",
    },
    {
        "id": "warm-bronze-modern-bed",
        "query_text": "a warm bronze bed frame for a modern bedroom",
        "story": "Find a modern warm-toned bed frame from catalog text.",
        "split": "validation",
        "expected_type": "BED",
    },
)

PRODUCT_KEYS: Final = {
    "id",
    "item_id",
    *OFFICIAL_FIELDS,
    "main_image_id",
    "selected_image_id",
    "image_role",
    "source_object_path",
    "source_url",
    "image_path",
    "image_mime",
    "image_width",
    "image_height",
    "image_bytes",
    "image_etag",
    "image_md5",
    "image_sha256",
    *SIMULATED_FIELDS,
    *DERIVED_SIMULATED_FIELDS,
    "embedding",
    "field_provenance",
}
QUERY_KEYS: Final = {"id", "query_text", "story", "split", "expected_type", "embedding"}
TRAINING_KEYS: Final = {
    "query_id",
    "product_id",
    "split",
    "features",
    "label",
    "label_provenance",
}


class DatasetValidationError(ValueError):
    """Raised when the checked-in dataset violates its strict contract."""


@dataclass(frozen=True)
class Product:
    id: int
    item_id: str
    title: str | None
    product_type: str
    brand: str | None
    color: str | None
    material: str | None
    style: str | None
    node_name: str | None
    description: str | None
    bullet_points: tuple[str, ...]
    main_image_id: str
    selected_image_id: str
    image_role: str
    source_object_path: str
    source_url: str
    image_path: str
    image_mime: str
    image_width: int
    image_height: int
    image_bytes: int
    image_etag: str
    image_md5: str
    image_sha256: str
    display_price_usd: float
    rating_value: float
    clicks_30d: int
    sales_30d: int
    inventory_units: int
    inventory_capacity: int
    return_rate: float
    release_date: str
    release_epoch: int
    rating: float
    inventory: float
    freshness: float
    embedding: tuple[float, ...]
    field_provenance: dict[str, str]

    @property
    def description_text(self) -> str:
        parts = [self.description or "", *self.bullet_points]
        return " ".join(part for part in parts if part)

    def as_milvus_row(self) -> dict[str, object]:
        return {
            "id": self.id,
            "item_id": self.item_id,
            "title": self.title or "",
            "product_type": self.product_type,
            "brand": self.brand or "",
            "color": self.color or "",
            "material": self.material or "",
            "style": self.style or "",
            "node_name": self.node_name or "",
            "description": self.description_text,
            "main_image_id": self.main_image_id,
            "display_price_usd": self.display_price_usd,
            "rating_value": self.rating_value,
            "clicks_30d": self.clicks_30d,
            "sales_30d": self.sales_30d,
            "inventory_units": self.inventory_units,
            "inventory_capacity": self.inventory_capacity,
            "return_rate": self.return_rate,
            "release_date": self.release_date,
            "release_epoch": self.release_epoch,
            "rating": self.rating,
            "inventory": self.inventory,
            "freshness": self.freshness,
            "image_path": self.image_path,
            "image_mime": self.image_mime,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "image_sha256": self.image_sha256,
            "embedding": list(self.embedding),
        }


@dataclass(frozen=True)
class DemoQuery:
    id: str
    query_text: str
    story: str
    split: str
    expected_type: str
    embedding: tuple[float, ...]


@dataclass(frozen=True)
class TrainingRecord:
    query_id: str
    product_id: int
    split: str
    features: tuple[float, ...]
    label: float
    label_provenance: str


@dataclass(frozen=True)
class CatalogDataset:
    root: Path
    manifest: dict[str, Any]
    manifest_sha256: str
    products: tuple[Product, ...]
    queries: tuple[DemoQuery, ...]
    training_records: tuple[TrainingRecord, ...]
    embedding_contract: EmbeddingContract
    source_objects: dict[str, Any]

    def product_asset(self, relative_path: str) -> Path:
        allowed = {product.image_path for product in self.products}
        if relative_path not in allowed:
            raise KeyError(relative_path)
        candidate = (self.root / relative_path).resolve(strict=True)
        if not candidate.is_relative_to(self.root.resolve()) or not candidate.is_file():
            raise DatasetValidationError("Product asset escaped the dataset root")
        return candidate

    def product_by_id(self, product_id: int) -> Product:
        for product in self.products:
            if product.id == product_id:
                return product
        raise KeyError(product_id)

    def public_identity(self) -> dict[str, object]:
        return {
            "dataset_id": DATASET_ID,
            "revision": DATASET_REVISION,
            "manifest_sha256": self.manifest_sha256,
            "dataset_name": "Synthetic Commerce Catalog",
            "publisher_and_data_credit": "milvus3-demos project (no external publisher)",
            "license": "CC0-1.0",
            "license_url": "https://creativecommons.org/publicdomain/zero/1.0/",
            "license_conflict_record": "none",
            "publication_review_required": False,
            "source_type": "synthetic_catalog_with_simulated_operations",
            "synthetic": True,
            "real_product_metadata": False,
            "real_product_photos": False,
            "real_transaction_data": False,
            "contains_simulated_operational_signals": True,
            "simulated_signal_label": SIMULATED_FIELD,
            "as_of_date": DATASET_AS_OF_DATE.isoformat(),
            "product_count": len(self.products),
            "product_type_count": len({product.product_type for product in self.products}),
            "image_count": len({product.image_path for product in self.products}),
            "embedding": self.embedding_contract.public_metadata(),
            "semantic_score_preprocessing": {
                "operation": "round_decimal",
                "decimal": SEMANTIC_SCORE_DECIMALS,
                "training_and_milvus_consistent": True,
            },
            "relevance_ground_truth": {
                "path": GROUND_TRUTH_FILE,
                "evaluation_only": True,
                "production_ranking_consumers": [],
            },
            "field_provenance": self.manifest["field_provenance"],
        }


def _stable_unit(item_id: str, field: str) -> float:
    payload = f"{DATASET_SEED}:{DATASET_REVISION}:{item_id}:{field}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") / (2**64 - 1)


def _stable_int(item_id: str, field: str, minimum: int, maximum: int) -> int:
    return minimum + int(_stable_unit(item_id, field) * (maximum - minimum + 1))


def simulated_signals(item_id: str) -> dict[str, object]:
    """Create fictional operational values; no output is a merchant or user fact."""
    display_price = round(18.0 + _stable_unit(item_id, "display_price_usd") * 882.0, 2)
    profile = operational_profile(item_id)
    if profile is None:
        rating_value = round(3.8 + _stable_unit(item_id, "rating_value") * 0.5, 1)
        inventory_ratio = 0.35 + _stable_unit(item_id, "inventory_ratio") * 0.30
        return_rate = round(0.07 + _stable_unit(item_id, "return_rate") * 0.06, 4)
        freshness_target = 0.35 + _stable_unit(item_id, "freshness") * 0.30
        age_days = round((1.0 - freshness_target) * 730)
    else:
        rating_value = profile.rating_value
        inventory_ratio = profile.inventory_ratio
        return_rate = profile.return_rate
        age_days = profile.age_days
    clicks = _stable_int(item_id, "clicks_30d", 320, 12_000)
    sales = min(clicks, _stable_int(item_id, "sales_30d", 8, 850))
    capacity = 200 if profile is not None else _stable_int(item_id, "inventory_capacity", 100, 240)
    units = round(inventory_ratio * capacity)
    release_date = DATASET_AS_OF_DATE - timedelta(days=age_days)
    release_epoch = (release_date - date(1970, 1, 1)).days * 86400
    return {
        "display_price_usd": display_price,
        "rating_value": rating_value,
        "clicks_30d": clicks,
        "sales_30d": sales,
        "inventory_units": units,
        "inventory_capacity": capacity,
        "return_rate": return_rate,
        "release_date": release_date.isoformat(),
        "release_epoch": release_epoch,
        "rating": round((rating_value - 1.0) / 4.0, 6),
        "inventory": round(units / capacity, 6),
        "freshness": round(max(0.0, 1.0 - age_days / 730.0), 6),
    }


def product_provenance() -> dict[str, str]:
    provenance = {
        "id": "project_dataset_sequence",
        "item_id": SOURCE_IDENTITY_FIELD,
        **{field: SOURCE_FIELD for field in OFFICIAL_FIELDS},
        "title": "per_product_declared_title_provenance",
        "main_image_id": SOURCE_IDENTITY_FIELD,
        "selected_image_id": SOURCE_IMAGE_FIELD,
        "image_role": SOURCE_IMAGE_FIELD,
        "source_object_path": SOURCE_IMAGE_FIELD,
        "source_url": SOURCE_IMAGE_FIELD,
        "image_path": "project_copy_of_hash_pinned_synthetic_placeholder_jpeg",
        "image_mime": SOURCE_IMAGE_FIELD,
        "image_width": SOURCE_IMAGE_FIELD,
        "image_height": SOURCE_IMAGE_FIELD,
        "image_bytes": "verified_source_object_metadata",
        "image_etag": "verified_source_object_metadata",
        "image_md5": "verified_source_object_digest",
        "image_sha256": "verified_source_object_digest",
        **{field: SIMULATED_FIELD for field in SIMULATED_FIELDS},
        **{field: DERIVED_SIMULATED_FIELD for field in DERIVED_SIMULATED_FIELDS},
        "embedding": DERIVED_TEXT_FIELD,
        "field_provenance": "project_data_contract",
    }
    if set(provenance) != PRODUCT_KEYS:
        raise AssertionError("Product provenance does not cover every field")
    return provenance


def item_product_provenance(source_item: dict[str, object]) -> dict[str, str]:
    provenance = product_provenance()
    metadata_provenance = source_item.get("metadata_provenance")
    if not isinstance(metadata_provenance, dict) or metadata_provenance.get("title") not in {
        SOURCE_FIELD,
        DERIVED_LOCALIZED_TITLE_FIELD,
    }:
        raise DatasetValidationError("Source title provenance is invalid")
    provenance["title"] = str(metadata_provenance["title"])
    return provenance


def _product_document(metadata: dict[str, object]) -> dict[str, object]:
    return {field: metadata[field] for field in OFFICIAL_FIELDS}


def _embedding_document(item: dict[str, object]) -> dict[str, object]:
    """Build the embedding input from the authored metadata fields."""
    return _product_document(dict(item["official_metadata"]))


def _training_features(
    query_embedding: tuple[float, ...], product: dict[str, object]
) -> tuple[float, ...]:
    return (
        normalize_semantic_score(
            cosine(query_embedding, product["embedding"])  # type: ignore[arg-type]
        ),
        float(product["rating"]),
        chain_popularity(float(product["clicks_30d"]), float(product["sales_30d"])),
        chain_price_affinity(float(product["display_price_usd"])),
        chain_freshness(float(product["release_epoch"])),
    )


def normalize_semantic_score(value: float) -> float:
    """Match Milvus Function Chain round_decimal preprocessing exactly."""
    return round(float(value), SEMANTIC_SCORE_DECIMALS)


def _training_label(features: tuple[float, ...]) -> float:
    value = sum(weight * feature for weight, feature in zip(LABEL_WEIGHTS, features, strict=True))
    return round(max(0.0, min(1.0, value)), 8)


def _notice() -> str:
    return """# Synthetic Commerce Catalog notice

This revision is fully synthetic. Product identifiers, brand names, titles,
descriptions, bullet points, colors, and all other metadata are authored by the
milvus3-demos project for the Function Chain reranking demonstration. They are
not derived from any real retailer, merchant, or public dataset.

Every product image is a deterministic solid-color JPEG placeholder
(640 x 480, baseline JPEG) and is not a product photograph. Placeholder images
are labeled `synthetic-placeholder` in the source manifest; this revision must
not be published as if it contained real product photography.

Operational values (display price, rating, inventory, return rate, release age,
clicks, and sales) are deterministic simulated signals and are labeled as such
on every field. They are not merchant, transaction, or user facts.

The authored catalog content is dedicated to the public domain under CC0 1.0
(https://creativecommons.org/publicdomain/zero/1.0/); see `licenses/LICENSE.txt`.
No third-party attribution applies because no third-party content is included.
"""


def _conflict_notice() -> str:
    return """# License conflict record

No license conflict applies to this revision. The catalog metadata and the
solid-color placeholder images are authored by the milvus3-demos project and
dedicated to the public domain under CC0 1.0 (`licenses/LICENSE.txt`).

No third-party metadata, photography, or license texts are included, so there
is no controlling third-party grant to re-check. A public release must still
disclose that the images are synthetic placeholders and not product photographs.
"""


def _file_entry(path: Path, root: Path, media_type: str) -> dict[str, object]:
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "media_type": media_type,
    }


def generate_dataset(
    output: Path, source_manifest_path: Path, source_cache: Path, force: bool = False
) -> Path:
    source = json.loads(source_manifest_path.read_text())
    if (
        source.get("revision") != DATASET_REVISION
        or source.get("selection_rule") != selection_rule()
    ):
        raise DatasetValidationError("Source manifest identity or selection rule is invalid")
    items = source.get("items")
    if not isinstance(items, list) or len(items) != 240:
        raise DatasetValidationError("Source manifest must contain exactly 240 items")
    if output.exists():
        if not force:
            raise FileExistsError(f"Dataset output exists: {output}")
        if output.resolve() == Path("/") or output.name != DATASET_REVISION:
            raise DatasetValidationError("Refusing to replace an unexpected output path")
        shutil.rmtree(output)
    output.mkdir(parents=True)
    (output / "images").mkdir()
    (output / "licenses").mkdir()

    contract = build_contract(_product_document(item["official_metadata"]) for item in items)
    products: list[dict[str, object]] = []
    for item in items:
        source_image = source_cache / item["object_path"]
        if not source_image.is_file() or sha256_file(source_image) != item["sha256"]:
            raise DatasetValidationError(f"Missing or changed cache object: {item['object_path']}")
        image_path = f"images/{int(item['sequence']):03d}-{item['item_id']}.jpg"
        destination = output / image_path
        shutil.copyfile(source_image, destination)
        metadata = dict(item["official_metadata"])
        product = {
            "id": int(item["sequence"]),
            "item_id": item["item_id"],
            **metadata,
            "main_image_id": item["main_image_id"],
            "selected_image_id": item["selected_image_id"],
            "image_role": item["image_role"],
            "source_object_path": item["object_path"],
            "source_url": item["url"],
            "image_path": image_path,
            "image_mime": IMAGE_MIME,
            "image_width": item["image_width"],
            "image_height": item["image_height"],
            "image_bytes": item["bytes"],
            "image_etag": item["http_head"]["etag"],
            "image_md5": item["md5"],
            "image_sha256": item["sha256"],
            **simulated_signals(str(item["item_id"])),
            "embedding": list(encode_document(_embedding_document(item), contract)),
            "field_provenance": item_product_provenance(item),
        }
        if set(product) != PRODUCT_KEYS:
            raise AssertionError("Generated product keys are incomplete")
        products.append(product)

    queries = [
        {**spec, "embedding": list(encode_query(str(spec["query_text"]), contract))}
        for spec in QUERY_SPECS
    ]
    relevance_ground_truth = build_ground_truth(queries, items)
    training_records: list[dict[str, object]] = []
    for query in queries:
        query_embedding = tuple(query["embedding"])
        for product in products:
            features = _training_features(query_embedding, product)
            training_records.append(
                {
                    "query_id": query["id"],
                    "product_id": product["id"],
                    "split": query["split"],
                    "features": list(features),
                    "label": _training_label(features),
                    "label_provenance": SIMULATED_FIELD,
                }
            )

    documents = {
        "products.json": {
            "dataset_id": DATASET_ID,
            "revision": DATASET_REVISION,
            "products": products,
        },
        "queries.json": {
            "dataset_id": DATASET_ID,
            "revision": DATASET_REVISION,
            "queries": queries,
        },
        "training-records.json": {
            "dataset_id": DATASET_ID,
            "revision": DATASET_REVISION,
            "feature_order": list(FEATURE_NAMES),
            "label_provenance": SIMULATED_FIELD,
            "training_records": training_records,
        },
        "embedding-contract.json": contract.as_json(),
        GROUND_TRUTH_FILE: relevance_ground_truth,
        "source-objects.json": source,
    }
    for name, document in documents.items():
        (output / name).write_bytes(canonical_json(document))

    static_by_kind = {item["kind"]: item for item in source["static_objects"]}
    for kind, name in (
        ("dataset_readme", "SYNTHETIC-README.md"),
        ("controlling_license", "LICENSE.txt"),
    ):
        source_path = source_cache / static_by_kind[kind]["object_path"]
        if sha256_file(source_path) != static_by_kind[kind]["sha256"]:
            raise DatasetValidationError(f"Source evidence hash mismatch: {kind}")
        shutil.copyfile(source_path, output / "licenses" / name)
    (output / "NOTICE.md").write_text(_notice(), encoding="utf-8")
    (output / "LICENSE-CONFLICT.md").write_text(_conflict_notice(), encoding="utf-8")

    hash_paths = sorted(
        path for path in output.rglob("*") if path.is_file() and path.name != "hash-manifest.sha256"
    )
    hash_lines = [
        f"{sha256_file(path)}  {path.relative_to(output).as_posix()}" for path in hash_paths
    ]
    (output / "hash-manifest.sha256").write_text("\n".join(hash_lines) + "\n", encoding="utf-8")

    files: list[dict[str, object]] = []
    for path in sorted(path for path in output.rglob("*") if path.is_file()):
        suffix = path.suffix.lower()
        media_type = {
            ".json": "application/json",
            ".jpg": IMAGE_MIME,
            ".md": "text/markdown",
            ".txt": "text/plain",
            ".sha256": "text/plain",
        }[suffix]
        files.append(_file_entry(path, output, media_type))
    manifest = {
        "schema_version": 2,
        "dataset_id": DATASET_ID,
        "revision": DATASET_REVISION,
        "seed": DATASET_SEED,
        "as_of_date": DATASET_AS_OF_DATE.isoformat(),
        "generator_version": GENERATOR_VERSION,
        "source": {
            "dataset": "Synthetic Commerce Catalog",
            "publisher_and_data_credit": "milvus3-demos project (no external publisher)",
            "type": "synthetic_catalog_with_simulated_operations",
            "selection_version": SELECTION_VERSION,
            "source_objects": "source-objects.json",
        },
        "license": {
            "development_controlling_spdx": "CC0-1.0",
            "url": "https://creativecommons.org/publicdomain/zero/1.0/",
            "attribution": "NOTICE.md",
            "conflict_record": "none",
            "publication_review_required": False,
            "endorsement": False,
        },
        "simulation_boundary": {
            "real_product_metadata": False,
            "real_product_photos": False,
            "real_transaction_data": False,
            "placeholder_images": True,
            "placeholder_image_provenance": SOURCE_IMAGE_FIELD,
            "derived_localized_title_summary_is_simulated": False,
            "derived_localized_title_summary_provenance": DERIVED_LOCALIZED_TITLE_FIELD,
            "label": SIMULATED_FIELD,
            "fields": list(SIMULATED_FIELDS + DERIVED_SIMULATED_FIELDS),
            "training_labels": SIMULATED_FIELD,
            "operational_profile_contract": {
                "version": PROFILE_VERSION,
                "key": "item_id",
                "profiled_item_count": len(ITEM_PROFILE_NAMES),
                "templates": {
                    name: {
                        "rating_value": profile.rating_value,
                        "inventory_ratio": profile.inventory_ratio,
                        "return_rate": profile.return_rate,
                        "age_days": profile.age_days,
                    }
                    for name, profile in PROFILE_TEMPLATES.items()
                },
                "query_identity_used": False,
                "vector_rank_used": False,
                "frozen_target_used": False,
            },
        },
        "field_provenance": product_provenance(),
        "embedding": contract.public_metadata(),
        "feature_order": list(FEATURE_NAMES),
        "training": {
            "split_unit": "query_group",
            "train_query_ids": [query["id"] for query in queries if query["split"] == "train"],
            "validation_query_ids": [
                query["id"] for query in queries if query["split"] == "validation"
            ],
            "label_formula": (
                "clamp(0,1,0.80*round(semantic_score,5)+0.02*rating+0.08*popularity"
                "+0.06*price_affinity+0.04*freshness)"
            ),
            "semantic_score_preprocessing": {
                "operation": "round_decimal",
                "decimal": SEMANTIC_SCORE_DECIMALS,
                "applied_before_label_and_model_training": True,
                "milvus_function_chain_match_required": True,
            },
            "label_provenance": SIMULATED_FIELD,
            "relevance_guardrail": {
                "semantic_weight": 0.80,
                "maximum_positive_operational_weight": 0.20,
                "return_rate_penalty_weight": 0.0,
                "monotonic_constraints": [1, 1, 1, 1, 1],
                "intent_source": "fixed token-boundary product-type aliases in query_text",
                "query_id_or_product_target_used": False,
            },
        },
        "counts": {
            "products": 240,
            "product_types": 20,
            "product_type_counts": dict(TYPE_ITEM_COUNTS),
            "images": 240,
            "queries": 6,
            "ground_truth_judgments": 1440,
            "training_records": 1440,
            "train_query_groups": 4,
            "validation_query_groups": 2,
        },
        "rebuild": {
            "working_directory": "demos/function-chain-rerank/backend",
            "prepare_command": (
                "uv run python -m function_chain_demo.synthetic_source prepare "
                "--cache ../../../artifacts/runtime/synthetic-commerce-catalog-r1/cache "
                "--output ../../../artifacts/runtime/synthetic-commerce-catalog-r1/source-manifest.json"
            ),
            "generate_command": (
                "uv run python -m function_chain_demo.dataset generate "
                "--source-manifest "
                "../../../artifacts/runtime/synthetic-commerce-catalog-r1/source-manifest.json "
                "--source-cache ../../../artifacts/runtime/synthetic-commerce-catalog-r1/cache"
            ),
        },
        "hash_algorithm": "SHA-256",
        "files": files,
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_bytes(canonical_json(manifest))
    return manifest_path


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetValidationError(f"Cannot read valid JSON: {path}") from exc


def _exact_keys(value: object, keys: set[str], context: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise DatasetValidationError(f"{context} keys are invalid")
    return value


def _validate_file_inventory(root: Path, manifest: dict[str, Any]) -> None:
    entries = manifest.get("files")
    if not isinstance(entries, list) or not entries:
        raise DatasetValidationError("Manifest files must be a non-empty list")
    declared: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"path", "bytes", "sha256", "media_type"}:
            raise DatasetValidationError("Manifest file entry is invalid")
        relative = entry["path"]
        if (
            not isinstance(relative, str)
            or not relative
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or relative in declared
        ):
            raise DatasetValidationError(f"Invalid manifest path: {relative}")
        declared.add(relative)
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise DatasetValidationError(f"Missing or unsafe manifest file: {relative}")
        if path.stat().st_size != entry["bytes"] or sha256_file(path) != entry["sha256"]:
            raise DatasetValidationError(f"Manifest hash or byte size mismatch: {relative}")
        if relative.startswith("images/") and (
            entry["media_type"] != IMAGE_MIME or not relative.endswith(".jpg")
        ):
            raise DatasetValidationError(f"Image media contract is invalid: {relative}")
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    if actual != declared:
        raise DatasetValidationError("Manifest inventory does not match dataset files")

    hash_lines = (root / "hash-manifest.sha256").read_text().splitlines()
    hashed: set[str] = set()
    for line in hash_lines:
        digest, separator, relative = line.partition("  ")
        if not separator or relative in hashed or len(digest) != 64:
            raise DatasetValidationError("Hash manifest syntax is invalid")
        path = root / relative
        if not path.is_file() or sha256_file(path) != digest:
            raise DatasetValidationError(f"Hash manifest mismatch: {relative}")
        hashed.add(relative)
    expected = declared - {"hash-manifest.sha256"}
    if hashed != expected:
        raise DatasetValidationError("Hash manifest coverage is invalid")


def load_dataset(root: Path = DEFAULT_DATASET_ROOT) -> CatalogDataset:
    manifest_path = root / "manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = _read_json(manifest_path)
    if (
        manifest.get("schema_version") != 2
        or manifest.get("dataset_id") != DATASET_ID
        or manifest.get("revision") != DATASET_REVISION
        or manifest.get("seed") != DATASET_SEED
        or manifest.get("generator_version") != GENERATOR_VERSION
        or manifest.get("hash_algorithm") != "SHA-256"
    ):
        raise DatasetValidationError("Manifest fixed identity is invalid")
    if manifest.get("field_provenance") != product_provenance():
        raise DatasetValidationError("Manifest field provenance is invalid")
    if manifest.get("feature_order") != list(FEATURE_NAMES):
        raise DatasetValidationError("Manifest feature order is invalid")
    simulation = manifest.get("simulation_boundary", {})
    if (
        simulation.get("real_product_metadata") is not False
        or simulation.get("real_product_photos") is not False
        or simulation.get("real_transaction_data") is not False
        or simulation.get("placeholder_images") is not True
        or simulation.get("placeholder_image_provenance") != SOURCE_IMAGE_FIELD
        or simulation.get("label") != SIMULATED_FIELD
        or simulation.get("training_labels") != SIMULATED_FIELD
    ):
        raise DatasetValidationError("Simulation boundary is invalid")
    profile_contract = simulation.get("operational_profile_contract", {})
    if (
        profile_contract.get("version") != PROFILE_VERSION
        or profile_contract.get("key") != "item_id"
        or profile_contract.get("profiled_item_count") != len(ITEM_PROFILE_NAMES)
        or profile_contract.get("query_identity_used") is not False
        or profile_contract.get("vector_rank_used") is not False
        or profile_contract.get("frozen_target_used") is not False
    ):
        raise DatasetValidationError("Operational profile contract is invalid")
    training_contract = manifest.get("training", {})
    if training_contract.get("semantic_score_preprocessing") != {
        "operation": "round_decimal",
        "decimal": SEMANTIC_SCORE_DECIMALS,
        "applied_before_label_and_model_training": True,
        "milvus_function_chain_match_required": True,
    }:
        raise DatasetValidationError("Semantic score preprocessing contract is invalid")
    license_info = manifest.get("license", {})
    if (
        license_info.get("development_controlling_spdx") != "CC0-1.0"
        or license_info.get("conflict_record") != "none"
        or license_info.get("publication_review_required") is not False
        or license_info.get("endorsement") is not False
    ):
        raise DatasetValidationError("License and release contract is invalid")
    _validate_file_inventory(root, manifest)

    source = _read_json(root / "source-objects.json")
    if (
        source.get("revision") != DATASET_REVISION
        or source.get("selection_rule") != selection_rule()
        or len(source.get("items", [])) != 240
    ):
        raise DatasetValidationError("Frozen source object manifest is invalid")
    expected_static_objects = [dict(static_object) for static_object in STATIC_OBJECTS]
    if source.get("static_objects") != expected_static_objects:
        raise DatasetValidationError("Frozen metadata and license source objects are invalid")
    source_items = source["items"]
    for key in ("item_id", "selected_image_id", "object_path"):
        if len({item[key] for item in source_items}) != 240:
            raise DatasetValidationError(f"Source {key} must be globally unique")
    expected_sequences = list(range(1, 241))
    if [item["sequence"] for item in source_items] != expected_sequences:
        raise DatasetValidationError("Source sequences are not stable")
    expected_types = [str(item["product_type"]) for item in build_items()]
    if [item["official_metadata"]["product_type"] for item in source_items] != expected_types:
        raise DatasetValidationError("Source product type order is invalid")

    contract = load_contract(_read_json(root / "embedding-contract.json"))
    products_doc = _read_json(root / "products.json")
    _exact_keys(products_doc, {"dataset_id", "revision", "products"}, "products document")
    if products_doc["dataset_id"] != DATASET_ID or products_doc["revision"] != DATASET_REVISION:
        raise DatasetValidationError("Products document identity is invalid")
    raw_products = products_doc["products"]
    if not isinstance(raw_products, list) or len(raw_products) != 240:
        raise DatasetValidationError("Dataset must contain exactly 240 products")
    products: list[Product] = []
    for index, (item, source_item) in enumerate(
        zip(raw_products, source_items, strict=True), start=1
    ):
        _exact_keys(item, PRODUCT_KEYS, f"product {index}")
        if item["id"] != index or item["item_id"] != source_item["item_id"]:
            raise DatasetValidationError(f"Product {index} identity is invalid")
        metadata = source_item["official_metadata"]
        if any(item[field] != metadata[field] for field in OFFICIAL_FIELDS):
            raise DatasetValidationError(f"Product {index} changes or infers synthetic metadata")
        if item["field_provenance"] != item_product_provenance(source_item):
            raise DatasetValidationError(f"Product {index} provenance is invalid")
        expected_signals = simulated_signals(item["item_id"])
        if any(item[field] != expected_signals[field] for field in expected_signals):
            raise DatasetValidationError(f"Product {index} simulated signals are invalid")
        expected_image_path = f"images/{index:03d}-{item['item_id']}.jpg"
        image_fields = {
            "main_image_id": source_item["main_image_id"],
            "selected_image_id": source_item["selected_image_id"],
            "image_role": source_item["image_role"],
            "source_object_path": source_item["object_path"],
            "source_url": source_item["url"],
            "image_path": expected_image_path,
            "image_mime": IMAGE_MIME,
            "image_width": source_item["image_width"],
            "image_height": source_item["image_height"],
            "image_bytes": source_item["bytes"],
            "image_etag": source_item["http_head"]["etag"],
            "image_md5": source_item["md5"],
            "image_sha256": source_item["sha256"],
        }
        if any(item[field] != value for field, value in image_fields.items()):
            raise DatasetValidationError(f"Product {index} source image metadata is invalid")
        image_path = root / expected_image_path
        if (
            image_path.stat().st_size != item["image_bytes"]
            or sha256_file(image_path) != item["image_sha256"]
            or jpeg_dimensions(image_path) != (item["image_width"], item["image_height"])
        ):
            raise DatasetValidationError(f"Product {index} JPEG verification failed")
        expected_embedding = encode_document(_embedding_document(source_item), contract)
        embedding = tuple(float(value) for value in item["embedding"])
        if embedding != expected_embedding or not math.isclose(
            sum(value * value for value in embedding), 1.0, abs_tol=2e-6
        ):
            raise DatasetValidationError(f"Product {index} embedding is invalid")
        products.append(
            Product(
                **{
                    **item,
                    "bullet_points": tuple(item["bullet_points"]),
                    "embedding": embedding,
                    "field_provenance": dict(item["field_provenance"]),
                }
            )
        )

    queries_doc = _read_json(root / "queries.json")
    _exact_keys(queries_doc, {"dataset_id", "revision", "queries"}, "queries document")
    if queries_doc["dataset_id"] != DATASET_ID or queries_doc["revision"] != DATASET_REVISION:
        raise DatasetValidationError("Queries document identity is invalid")
    raw_queries = queries_doc["queries"]
    if not isinstance(raw_queries, list) or len(raw_queries) != 6:
        raise DatasetValidationError("Dataset must contain exactly six queries")
    queries: list[DemoQuery] = []
    for item, spec in zip(raw_queries, QUERY_SPECS, strict=True):
        _exact_keys(item, QUERY_KEYS, "query")
        if any(item[key] != value for key, value in spec.items()):
            raise DatasetValidationError("Natural query contract changed")
        embedding = tuple(float(value) for value in item["embedding"])
        if embedding != encode_query(item["query_text"], contract):
            raise DatasetValidationError(f"Query embedding is invalid: {item['id']}")
        queries.append(DemoQuery(**{**item, "embedding": embedding}))
    if {query.split for query in queries} != {"train", "validation"}:
        raise DatasetValidationError("Query group splits are invalid")

    training_doc = _read_json(root / "training-records.json")
    _exact_keys(
        training_doc,
        {"dataset_id", "revision", "feature_order", "label_provenance", "training_records"},
        "training document",
    )
    if (
        training_doc["dataset_id"] != DATASET_ID
        or training_doc["revision"] != DATASET_REVISION
        or training_doc["feature_order"] != list(FEATURE_NAMES)
        or training_doc["label_provenance"] != SIMULATED_FIELD
    ):
        raise DatasetValidationError("Training document contract is invalid")
    raw_records = training_doc["training_records"]
    if not isinstance(raw_records, list) or len(raw_records) != 1440:
        raise DatasetValidationError("Training data must contain six complete query groups")
    training_records: list[TrainingRecord] = []
    expected_pairs = [(query, product) for query in queries for product in products]
    for item, (query, product) in zip(raw_records, expected_pairs, strict=True):
        _exact_keys(item, TRAINING_KEYS, "training record")
        expected_features = _training_features(query.embedding, product.as_milvus_row())
        features = tuple(float(value) for value in item["features"])
        if (
            item["query_id"] != query.id
            or item["product_id"] != product.id
            or item["split"] != query.split
            or features != expected_features
            or item["label"] != _training_label(features)
            or item["label_provenance"] != SIMULATED_FIELD
        ):
            raise DatasetValidationError("Training record is not reproducible")
        training_records.append(TrainingRecord(**{**item, "features": features}))

    ground_truth = load_ground_truth(root)
    if ground_truth != build_ground_truth(raw_queries, source_items):
        raise DatasetValidationError("Relevance ground truth is not reproducible")

    expected_counts = {
        "products": 240,
        "product_types": 20,
        "product_type_counts": dict(TYPE_ITEM_COUNTS),
        "images": 240,
        "queries": 6,
        "ground_truth_judgments": 1440,
        "training_records": 1440,
        "train_query_groups": 4,
        "validation_query_groups": 2,
    }
    if manifest.get("counts") != expected_counts:
        raise DatasetValidationError("Manifest counts are invalid")
    return CatalogDataset(
        root=root,
        manifest=manifest,
        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        products=tuple(products),
        queries=tuple(queries),
        training_records=tuple(training_records),
        embedding_contract=contract,
        source_objects=source,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate = subparsers.add_parser("generate")
    generate.add_argument("--output", type=Path, default=DEFAULT_DATASET_ROOT)
    generate.add_argument("--source-manifest", type=Path, required=True)
    generate.add_argument("--source-cache", type=Path, required=True)
    generate.add_argument("--force", action="store_true")
    validate = subparsers.add_parser("validate")
    validate.add_argument("--root", type=Path, default=DEFAULT_DATASET_ROOT)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.command == "generate":
        manifest_path = generate_dataset(
            args.output, args.source_manifest, args.source_cache, force=args.force
        )
        dataset = load_dataset(manifest_path.parent)
    else:
        dataset = load_dataset(args.root)
    print(json.dumps(dataset.public_identity(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
