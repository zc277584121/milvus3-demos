import json
import shutil
from pathlib import Path

import pytest

from function_chain_demo.synthetic_source import (
    TARGET_TYPES,
    TYPE_COLORS,
)
from function_chain_demo.catalog import DATASET
from function_chain_demo.catalog_curation import (
    PARTIAL,
    STRONG,
    classify_intent_candidate,
)
from function_chain_demo.dataset import (
    DEFAULT_DATASET_ROOT,
    DERIVED_SIMULATED_FIELD,
    DERIVED_SIMULATED_FIELDS,
    OFFICIAL_FIELDS,
    SIMULATED_FIELD,
    SIMULATED_FIELDS,
    DatasetValidationError,
    generate_dataset,
    load_dataset,
    simulated_signals,
)
from function_chain_demo.text_embedding import (
    cosine,
    semantic_attribute_types,
)


def file_hashes(root: Path) -> dict[str, str]:
    import hashlib

    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*")
        if path.is_file()
    }


def materialize_source_cache(root: Path) -> Path:
    cache = root / "source-cache"
    for product in DATASET.products:
        destination = cache / product.source_object_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(DEFAULT_DATASET_ROOT / product.image_path, destination)
    static_names = {
        "dataset_readme": "SYNTHETIC-README.md",
        "controlling_license": "LICENSE.txt",
    }
    for source in DATASET.source_objects["static_objects"]:
        name = static_names.get(source["kind"])
        if name:
            destination = cache / source["object_path"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(DEFAULT_DATASET_ROOT / "licenses" / name, destination)
    return cache


def test_dataset_rebuild_is_byte_for_byte_identical(tmp_path) -> None:
    rebuilt = tmp_path / "synthetic-commerce-catalog-r1"
    source_cache = materialize_source_cache(tmp_path)
    generate_dataset(
        rebuilt,
        DEFAULT_DATASET_ROOT / "source-objects.json",
        source_cache,
    )

    assert file_hashes(rebuilt) == file_hashes(DEFAULT_DATASET_ROOT)
    assert load_dataset(rebuilt).manifest_sha256 == DATASET.manifest_sha256


def test_fixed_source_inventory_and_jpeg_hashes_are_complete() -> None:
    source = DATASET.source_objects
    items = source["items"]
    image_entries = [
        entry for entry in DATASET.manifest["files"] if entry["media_type"] == "image/jpeg"
    ]

    assert source["selection_rule"]["ordered_product_types"] == list(TARGET_TYPES)
    expected_type_counts = source["selection_rule"]["type_item_counts"]
    assert sum(expected_type_counts.values()) == 240
    assert len(items) == len(image_entries) == 240
    assert len({item["item_id"] for item in items}) == 240
    assert len({item["selected_image_id"] for item in items}) == 240
    assert len({item["object_path"] for item in items}) == 240
    assert len({entry["path"] for entry in image_entries}) == 240
    # Solid-color placeholders are identical within a product type (one color per
    # type), so there are exactly 20 distinct image digests across the catalog.
    assert len({entry["sha256"] for entry in image_entries}) == 20
    actual_type_counts = {
        product_type: sum(
            item["official_metadata"]["product_type"] == product_type for item in items
        )
        for product_type in TARGET_TYPES
    }
    assert actual_type_counts == expected_type_counts
    assert {item["image_role"] for item in items} == {"main"}
    assert {item["metadata_provenance"]["title"] for item in items} == {
        "synthetic_authored_metadata"
    }
    assert all(
        item["image_width"] == 640 and item["image_height"] == 480 for item in items
    )
    assert {item["http_head"].get("media_type") for item in items} == {"image/jpeg"}
    assert set(TYPE_COLORS) == set(TARGET_TYPES)


def test_intent_curation_is_auditable_and_not_a_production_ranker() -> None:
    items = DATASET.source_objects["items"]
    contract = DATASET.source_objects["selection_rule"]["intent_curation"]
    labels = {item["selection_class"] for item in items}

    assert contract["query_id_rules"] is False
    assert contract["item_id_rules"] is False
    assert contract["production_ranking_consumers"] == []
    assert labels == {
        "strong",
        "partial",
        "intra_type_hard_negative",
        "catalog_background",
    }
    assert sum(item["selection_class"] == "strong" for item in items) == 36
    assert sum(item["selection_class"] == "partial" for item in items) == 6
    assert sum(item["selection_class"] == "intra_type_hard_negative" for item in items) == 7


def test_bronze_intent_rejects_gold_and_accepts_shared_authored_image_evidence() -> None:
    gold = classify_intent_candidate(
        "BED",
        "Modern King Bed Frame, Gold Finished Metal",
        "gold brushed gold metal modern bed",
    )
    shared_bronze_family = classify_intent_candidate(
        "BED",
        "Classic Metal Bed Frame with Headboard",
        "classic metal bed frame",
        bronze_family_image_evidence=True,
    )

    assert gold is not None and gold.label == PARTIAL
    assert shared_bronze_family is not None and shared_bronze_family.label == STRONG
    assert "WARM_METAL_BED_FINISH" not in semantic_attribute_types(
        "Modern King Bed Frame, Gold Finished Metal"
    )


def test_official_and_simulated_field_boundaries_are_reproducible() -> None:
    for product, source in zip(DATASET.products, DATASET.source_objects["items"], strict=True):
        official = source["official_metadata"]
        assert all(
            list(getattr(product, field)) == official[field]
            if field == "bullet_points"
            else getattr(product, field) == official[field]
            for field in OFFICIAL_FIELDS
        )
        expected_signals = simulated_signals(product.item_id)
        assert all(getattr(product, field) == expected_signals[field] for field in expected_signals)
        assert all(product.field_provenance[field] == SIMULATED_FIELD for field in SIMULATED_FIELDS)
        assert all(
            product.field_provenance[field] == DERIVED_SIMULATED_FIELD
            for field in DERIVED_SIMULATED_FIELDS
        )
    assert all(product.color for product in DATASET.products)
    assert DATASET.manifest["simulation_boundary"]["real_transaction_data"] is False
    assert DATASET.manifest["simulation_boundary"]["real_product_metadata"] is False
    assert DATASET.manifest["simulation_boundary"]["real_product_photos"] is False
    assert DATASET.manifest["simulation_boundary"]["placeholder_images"] is True
    assert DATASET.manifest["training"]["label_provenance"] == SIMULATED_FIELD
    profile_contract = DATASET.manifest["simulation_boundary"]["operational_profile_contract"]
    assert profile_contract["key"] == "item_id"
    assert profile_contract["version"] == "synthetic-operational-profiles-v1"
    assert profile_contract["profiled_item_count"] == 36
    assert profile_contract["query_identity_used"] is False
    assert profile_contract["vector_rank_used"] is False
    assert profile_contract["frozen_target_used"] is False
    assert set(profile_contract["templates"]) == {"H", "G", "S", "N", "R"}


def test_queries_are_natural_and_training_is_grouped_without_answer_keys() -> None:
    raw_queries = json.loads((DEFAULT_DATASET_ROOT / "queries.json").read_text())["queries"]
    forbidden = {"business_target_id", "baseline_reference_id", "hard_negative_ids"}
    split_by_query = {query.id: query.split for query in DATASET.queries}

    assert all(not (set(query) & forbidden) for query in raw_queries)
    assert all(
        "SYN-" not in query.query_text
        and not any(character.isdigit() for character in query.query_text)
        for query in DATASET.queries
    )
    assert {query.split for query in DATASET.queries} == {"train", "validation"}
    assert len({(record.query_id, record.product_id) for record in DATASET.training_records}) == 1440
    assert all(
        record.split == split_by_query[record.query_id] for record in DATASET.training_records
    )
    assert all(
        sum(record.query_id == query.id for record in DATASET.training_records) == 240
        for query in DATASET.queries
    )


def test_queries_declare_expected_type_and_keep_it_ahead_of_operational_signals() -> None:
    expected_types = ("DESK", "RUG", "BACKPACK", "HEADPHONES", "UMBRELLA", "BED")
    for query, expected_type in zip(DATASET.queries, expected_types, strict=True):
        assert query.expected_type == expected_type
        vector_order = sorted(
            DATASET.products,
            key=lambda product: (-cosine(query.embedding, product.embedding), product.id),
        )
        # Pure BGE keeps the expected category at the top; it does not guarantee
        # a perfect same-type top-6 (that residual ambiguity is the demo's point).
        assert vector_order[0].product_type == expected_type
        assert sum(product.product_type == expected_type for product in vector_order[:6]) >= 5


def test_notice_declares_synthetic_catalog_and_placeholder_images() -> None:
    notice = (DEFAULT_DATASET_ROOT / "NOTICE.md").read_text()
    conflict = (DEFAULT_DATASET_ROOT / "LICENSE-CONFLICT.md").read_text()

    assert "Synthetic Commerce Catalog" in notice
    assert "not derived from any real retailer" in notice
    assert "solid-color JPEG placeholder" in notice
    assert "CC0 1.0" in notice
    assert "https://creativecommons.org/publicdomain/zero/1.0/" in notice
    assert "not product photographs" in conflict
    assert DATASET.manifest["license"]["publication_review_required"] is False
    assert DATASET.manifest["license"]["development_controlling_spdx"] == "CC0-1.0"


def test_tampered_data_is_rejected_by_both_hash_layers(tmp_path) -> None:
    tampered_root = tmp_path / "tampered"
    shutil.copytree(DEFAULT_DATASET_ROOT, tampered_root)
    products_path = tampered_root / "products.json"
    document = json.loads(products_path.read_text())
    document["products"][0]["title"] = "Tampered"
    products_path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(DatasetValidationError, match="hash or byte size mismatch"):
        load_dataset(tampered_root)
