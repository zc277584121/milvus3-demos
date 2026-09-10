"""CPU-only BGE-M3 ONNX dense embedding with a local-cache-first loader."""

from __future__ import annotations

import contextlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from structarray_hybrid_demo.config import (
    MODEL_BATCH_SIZE,
    MODEL_DIMENSION,
    MODEL_ID,
    MODEL_REVISION,
    RuntimeConfig,
)


class ModelContractError(RuntimeError):
    """Raised when the local ONNX model violates the fixed contract."""


@dataclass(frozen=True)
class ModelIdentity:
    model_id: str
    revision: str
    vector_dimension: int
    dense_output_name: str
    device: str

    def public_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class EmbeddingBatchReport:
    vector_count: int
    vector_dimension: int
    norm_min: float
    norm_max: float

    def public_dict(self) -> dict[str, object]:
        return asdict(self)


class BgeM3OnnxEmbedder:
    """Load the quantized BGE-M3 ONNX model and return L2-normalized dense vectors."""

    def __init__(self, config: RuntimeConfig) -> None:
        self._config = config
        self._session: Any = None
        self._tokenizer: Any = None
        self._output_names: list[str] = []
        self._dimension = MODEL_DIMENSION
        self._batch_size = MODEL_BATCH_SIZE

    @property
    def loaded(self) -> bool:
        return self._session is not None

    @property
    def cache_available(self) -> bool:
        try:
            return self._model_snapshot().is_dir()
        except ModelContractError:
            return False

    @property
    def identity(self) -> ModelIdentity:
        return ModelIdentity(
            model_id=MODEL_ID,
            revision=MODEL_REVISION,
            vector_dimension=self._dimension,
            dense_output_name="dense_vecs",
            device="cpu",
        )

    def _model_snapshot(self) -> Path:
        snapshot = (
            self._config.hf_hub_cache
            / f"models--{MODEL_ID.replace('/', '--')}"
            / "snapshots"
            / MODEL_REVISION
        )
        if not snapshot.is_dir():
            raise ModelContractError(f"Missing local BGE-M3 ONNX snapshot: {snapshot}")
        return snapshot

    def load(self) -> None:
        """Load ONNX Runtime session and tokenizer from the local snapshot."""
        if self._session is not None:
            return
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ModelContractError("onnxruntime is required for the BGE-M3 ONNX backend") from exc
        try:
            from tokenizers import Tokenizer
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ModelContractError("tokenizers is required for the BGE-M3 ONNX backend") from exc

        snapshot = self._model_snapshot()
        tokenizer_path = snapshot / "tokenizer.json"
        model_path = snapshot / "model_quantized.onnx"
        if not tokenizer_path.is_file() or not model_path.is_file():
            raise ModelContractError(f"BGE-M3 ONNX snapshot is incomplete: {snapshot}")

        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        tokenizer.enable_padding(pad_id=1, pad_token="<pad>")
        tokenizer.enable_truncation(max_length=8192)

        session = ort.InferenceSession(str(model_path))
        output_names = [output.name for output in session.get_outputs()]
        if "dense_vecs" not in output_names:
            raise ModelContractError("BGE-M3 ONNX model is missing the dense_vecs output")

        self._tokenizer = tokenizer
        self._session = session
        self._output_names = output_names

    def _encode_batch(self, texts: list[str]) -> list[list[float]]:
        encoded = self._tokenizer.encode_batch(texts)
        input_ids = np.asarray([item.ids for item in encoded], dtype=np.int64)
        attention_mask = np.asarray([item.attention_mask for item in encoded], dtype=np.int64)
        outputs = self._session.run(
            None, {"input_ids": input_ids, "attention_mask": attention_mask}
        )
        dense_index = self._output_names.index("dense_vecs")
        dense = np.asarray(outputs[dense_index], dtype=np.float32)
        if dense.ndim != 2 or dense.shape[0] != len(texts):
            raise ModelContractError("BGE-M3 ONNX dense output has an unexpected shape")
        norms = np.linalg.norm(dense, axis=1, keepdims=True)
        norms = np.where(norms == 0.0, 1.0, norms)
        normalized = dense / norms
        return [list(row) for row in normalized]

    def embed(self, texts: list[str]) -> tuple[list[list[float]], EmbeddingBatchReport]:
        """Embed an arbitrary number of texts in fixed-size batches."""
        if self._session is None:
            self.load()
        if not texts:
            raise ModelContractError("Cannot embed an empty text batch")
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            vectors.extend(self._encode_batch(batch))
        matrix = np.asarray(vectors, dtype=np.float32)
        report = EmbeddingBatchReport(
            vector_count=int(matrix.shape[0]),
            vector_dimension=int(matrix.shape[1]),
            norm_min=float(np.min(np.linalg.norm(matrix, axis=1))),
            norm_max=float(np.max(np.linalg.norm(matrix, axis=1))),
        )
        return vectors, report

    def unload(self) -> None:
        self._session = None
        self._tokenizer = None
        with contextlib.suppress(Exception):
            import gc

            gc.collect()
