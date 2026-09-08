"""Reproducible grouped XGBoost model generation and metadata."""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

import numpy as np
import xgboost as xgb

from function_chain_demo.business_gates import business_gate_report, direction_report
from function_chain_demo.catalog import DATASET, FEATURE_NAMES
from function_chain_demo.dataset import CatalogDataset

MODEL_SEED = 42
MODEL_ROUNDS = 128
MODEL_VERSION = "synthetic-commerce-catalog-r1-reranker-v1"
MODEL_PARAMETERS: Final[dict[str, object]] = {
    "max_depth": 2,
    "eta": 0.08,
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "nthread": 1,
    "seed": MODEL_SEED,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "tree_method": "hist",
    "monotone_constraints": "(1,1,1,1,1)",
}


@dataclass(frozen=True)
class TrainingMatrices:
    train_features: np.ndarray
    train_labels: np.ndarray
    validation_features: np.ndarray
    validation_labels: np.ndarray
    validation_query_ids_by_row: tuple[str, ...]
    train_query_ids: tuple[str, ...]
    validation_query_ids: tuple[str, ...]


@dataclass(frozen=True)
class ModelArtifact:
    path: Path
    sha256: str
    byte_size: int
    model_version: str
    xgboost_version: str
    seed: int
    rounds: int
    parameters: dict[str, object]
    feature_names: tuple[str, ...]
    dataset_revision: str
    training_records_sha256: str
    training_record_count: int
    train_query_ids: tuple[str, ...]
    validation_query_ids: tuple[str, ...]
    metrics: dict[str, float]
    direction_gate: dict[str, object]
    business_gate: dict[str, object]

    def public_metadata(self) -> dict[str, object]:
        metadata = asdict(self)
        metadata.pop("path")
        metadata["feature_names"] = list(self.feature_names)
        metadata["train_query_ids"] = list(self.train_query_ids)
        metadata["validation_query_ids"] = list(self.validation_query_ids)
        return metadata


def generate_training_data(dataset: CatalogDataset = DATASET) -> TrainingMatrices:
    """Build fixed train and validation matrices without row-level random splitting."""
    train_records = [record for record in dataset.training_records if record.split == "train"]
    validation_records = [
        record for record in dataset.training_records if record.split == "validation"
    ]
    query_splits = {query.id: query.split for query in dataset.queries}
    train_query_ids = tuple(
        sorted(query_id for query_id, split in query_splits.items() if split == "train")
    )
    validation_query_ids = tuple(
        sorted(query_id for query_id, split in query_splits.items() if split == "validation")
    )
    if set(train_query_ids) & set(validation_query_ids):
        raise ValueError("Query groups cannot cross train and validation splits")
    return TrainingMatrices(
        train_features=np.asarray([record.features for record in train_records], dtype=np.float32),
        train_labels=np.asarray([record.label for record in train_records], dtype=np.float32),
        validation_features=np.asarray(
            [record.features for record in validation_records], dtype=np.float32
        ),
        validation_labels=np.asarray(
            [record.label for record in validation_records], dtype=np.float32
        ),
        validation_query_ids_by_row=tuple(record.query_id for record in validation_records),
        train_query_ids=train_query_ids,
        validation_query_ids=validation_query_ids,
    )


def _rmse(labels: np.ndarray, predictions: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(labels - predictions))))


def _ndcg_at_10(
    labels: np.ndarray,
    predictions: np.ndarray,
    query_ids: tuple[str, ...],
) -> float:
    scores: list[float] = []
    for query_id in sorted(set(query_ids)):
        indexes = [index for index, value in enumerate(query_ids) if value == query_id]
        predicted_order = sorted(indexes, key=lambda index: (-predictions[index], index))[:10]
        ideal_order = sorted(indexes, key=lambda index: (-labels[index], index))[:10]

        def dcg(order: list[int]) -> float:
            return sum(
                (2.0 ** float(labels[index]) - 1.0) / math.log2(rank + 2.0)
                for rank, index in enumerate(order)
            )

        ideal = dcg(ideal_order)
        scores.append(dcg(predicted_order) / ideal if ideal else 1.0)
    return sum(scores) / len(scores)


def _file_hash(dataset: CatalogDataset, relative_path: str) -> str:
    for entry in dataset.manifest["files"]:
        if entry["path"] == relative_path:
            return str(entry["sha256"])
    raise ValueError(f"Dataset manifest does not contain {relative_path}")


def _ranking_metrics(dataset: CatalogDataset, model: xgb.Booster) -> dict[str, float]:
    changed = 0
    relevant_changed = 0
    relevant_top1 = 0
    validation_relevant_top1 = 0
    validation_queries = 0
    for query in dataset.queries:
        expected_type = query.expected_type
        records = sorted(
            (record for record in dataset.training_records if record.query_id == query.id),
            key=lambda record: (-record.features[0], record.product_id),
        )[:20]
        predictions = model.predict(
            xgb.DMatrix(np.asarray([record.features for record in records], dtype=np.float32))
        )
        business_indexes = np.argsort(-predictions, kind="stable")
        vector_ids = [record.product_id for record in records]
        business_ids = [records[int(index)].product_id for index in business_indexes]
        changed += int(vector_ids != business_ids)
        vector_relevant_ids = [
            product_id
            for product_id in vector_ids
            if dataset.product_by_id(product_id).product_type == expected_type
        ]
        business_relevant_ids = [
            product_id
            for product_id in business_ids
            if dataset.product_by_id(product_id).product_type == expected_type
        ]
        relevant_changed += int(vector_relevant_ids != business_relevant_ids)
        business_top_type = dataset.product_by_id(business_ids[0]).product_type
        is_relevant = business_top_type == expected_type
        relevant_top1 += int(is_relevant)
        if query.split == "validation":
            validation_queries += 1
            validation_relevant_top1 += int(is_relevant)
    query_count = len(dataset.queries)
    return {
        "all_query_changed_order_rate": round(changed / query_count, 8),
        "all_query_relevant_candidate_changed_order_rate": round(relevant_changed / query_count, 8),
        "all_query_relevant_top1_rate": round(relevant_top1 / query_count, 8),
        "validation_relevant_top1_rate": round(validation_relevant_top1 / validation_queries, 8),
    }


def train_model(model_path: Path, dataset: CatalogDataset = DATASET) -> ModelArtifact:
    matrices = generate_training_data(dataset)
    train_matrix = xgb.DMatrix(matrices.train_features, label=matrices.train_labels)
    model = xgb.train(
        MODEL_PARAMETERS,
        train_matrix,
        num_boost_round=MODEL_ROUNDS,
    )
    train_predictions = model.predict(train_matrix)
    validation_predictions = model.predict(xgb.DMatrix(matrices.validation_features))
    metrics = {
        "train_rmse": round(_rmse(matrices.train_labels, train_predictions), 8),
        "validation_rmse": round(_rmse(matrices.validation_labels, validation_predictions), 8),
        "validation_ndcg_at_10": round(
            _ndcg_at_10(
                matrices.validation_labels,
                validation_predictions,
                matrices.validation_query_ids_by_row,
            ),
            8,
        ),
        **_ranking_metrics(dataset, model),
    }
    direction_gate = direction_report(dataset, model)
    business_gate = business_gate_report(dataset, model)
    if not all(bool(result["passed"]) for result in direction_gate.values()):
        raise RuntimeError("XGBoost model failed the five-feature monotonicity gate")
    if not business_gate["passed"]:
        raise RuntimeError("XGBoost model failed the frozen business-semantic gate")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(model_path)
    model_bytes = model_path.read_bytes()
    return ModelArtifact(
        path=model_path,
        sha256=hashlib.sha256(model_bytes).hexdigest(),
        byte_size=len(model_bytes),
        model_version=MODEL_VERSION,
        xgboost_version=xgb.__version__,
        seed=MODEL_SEED,
        rounds=MODEL_ROUNDS,
        parameters=dict(MODEL_PARAMETERS),
        feature_names=FEATURE_NAMES,
        dataset_revision=str(dataset.manifest["revision"]),
        training_records_sha256=_file_hash(dataset, "training-records.json"),
        training_record_count=len(dataset.training_records),
        train_query_ids=matrices.train_query_ids,
        validation_query_ids=matrices.validation_query_ids,
        metrics=metrics,
        direction_gate=direction_gate,
        business_gate=business_gate,
    )


def load_model(model_path: Path) -> xgb.Booster:
    model = xgb.Booster()
    model.load_model(model_path)
    return model
