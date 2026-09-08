import pytest

from function_chain_demo.synthetic_source import TARGET_TYPES
from function_chain_demo.catalog import (
    DATASET,
    EMBEDDING_DIMENSION,
    FEATURE_NAMES,
    PRODUCTS,
    QUERIES,
    get_query,
)


def test_catalog_uses_fixed_synthetic_identity_and_feature_order() -> None:
    assert DATASET.manifest["revision"] == "synthetic-commerce-catalog-r1"
    assert DATASET.manifest["seed"] == 20260820
    assert [product.id for product in PRODUCTS] == list(range(1, 241))
    expected_counts = DATASET.source_objects["selection_rule"]["type_item_counts"]
    assert {
        product_type: sum(product.product_type == product_type for product in PRODUCTS)
        for product_type in TARGET_TYPES
    } == expected_counts
    assert len({product.item_id for product in PRODUCTS}) == 240
    assert len({product.selected_image_id for product in PRODUCTS}) == 240
    assert len({product.source_object_path for product in PRODUCTS}) == 240
    assert len({product.image_path for product in PRODUCTS}) == 240
    assert len(QUERIES) == 6
    assert EMBEDDING_DIMENSION == 1024
    assert FEATURE_NAMES == (
        "semantic_score",
        "rating",
        "popularity",
        "price_affinity",
        "freshness",
    )
    assert len(get_query("compact-dark-wood-desk").embedding) == EMBEDDING_DIMENSION


def test_unknown_query_has_no_implicit_fallback() -> None:
    with pytest.raises(KeyError):
        get_query("unknown")
