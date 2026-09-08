"""Offline oracle used to verify, never serve, the Milvus rerank order."""

from pathlib import Path

import numpy as np
import xgboost as xgb

from function_chain_demo.service import SearchComparison


def local_model_order(model_path: Path, comparison: SearchComparison) -> list[int]:
    model = xgb.Booster()
    model.load_model(model_path)
    features = np.asarray(
        [
            [
                product.score,
                product.rating,
                product.inventory,
                product.return_rate,
                product.freshness,
            ]
            for product in comparison.vector_order
        ],
        dtype=np.float32,
    )
    predictions = model.predict(xgb.DMatrix(features))
    indices = np.argsort(-predictions, kind="stable")
    return [comparison.vector_order[int(index)].id for index in indices]
