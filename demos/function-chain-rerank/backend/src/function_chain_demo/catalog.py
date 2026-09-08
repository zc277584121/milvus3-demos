"""Strictly validated catalog exports used by the backend runtime."""

from __future__ import annotations

from typing import Final

from function_chain_demo.dataset import (
    DEFAULT_DATASET_ROOT,
    EMBEDDING_DIMENSION,
    FEATURE_NAMES,
    CatalogDataset,
    DemoQuery,
    Product,
    TrainingRecord,
    load_dataset,
)

__all__ = [
    "DATASET",
    "DATASET_IDENTITY",
    "EMBEDDING_DIMENSION",
    "FEATURE_NAMES",
    "PRODUCTS",
    "QUERIES",
    "TRAINING_RECORDS",
    "get_query",
]

DATASET: Final[CatalogDataset] = load_dataset(DEFAULT_DATASET_ROOT)
PRODUCTS: Final[tuple[Product, ...]] = DATASET.products
QUERIES: Final[tuple[DemoQuery, ...]] = DATASET.queries
TRAINING_RECORDS: Final[tuple[TrainingRecord, ...]] = DATASET.training_records
DATASET_IDENTITY: Final[dict[str, object]] = DATASET.public_identity()


def get_query(query_id: str) -> DemoQuery:
    for query in QUERIES:
        if query.id == query_id:
            return query
    raise KeyError(query_id)
