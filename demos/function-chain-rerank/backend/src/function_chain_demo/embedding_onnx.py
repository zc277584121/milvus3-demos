"""Offline ONNX BGE-M3 dense embeddings (pure semantic, no anchor dims).

This module replaces the earlier "dense + type/attribute anchor" embedding with a
plain, frozen BGE-M3 dense vector. The anchor block was a deterministic overlay of
extra dimensions that hard-coded product-type and attribute matches; it forced
same-type items to near-identical similarity (``0.9999``) and was hard to explain.
We removed it so the demo shows what a real semantic encoder returns, honestly,
including its known weakness: BGE separates *categories* well (RUG vs DESK) but
does not sharply separate *within-category attribute intent* (on-ear vs in-ear
headphones, neutral vs vibrant rug). That residual ambiguity is the whole point of
the downstream business Function Chain, and it is surfaced — not hidden — by the
relaxed offline gates and the UI.

Layout
------
``dense(1024)`` only. The encoder already L2-normalizes each dense vector, so
cosine similarity is a plain dot product. This keeps the Milvus schema at a single
``embedding`` FLOAT_VECTOR column with ``dimension = 1024``.

Properties the dataset loader relies on:
- ``dimension`` is exactly 1024.
- Vectors are L2-normalized so cosine similarity is a plain dot product.
- Inference is deterministic (bit-identical across runs and thread counts), so
  the strict loader still compares embeddings for exact equality.

The model identity is verified by HF-cache blob filename (the content SHA-256);
no heavyweight re-hash is performed on every load.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np

# --- Dense BGE-M3 identity -------------------------------------------------
EMBEDDING_DIMENSION: Final = 1024
EMBEDDING_VERSION: Final = "bge-m3-onnx-int8-dense-l2-v3"
EMBEDDING_MODEL_REPO: Final = "gpahal/bge-m3-onnx-int8"
EMBEDDING_MODEL_REVISION: Final = "2b34e84df040034d4b9eabb62383a87c18955822"
ONNX_FILE: Final = "model_quantized.onnx"
TOKENIZER_FILE: Final = "tokenizer.json"
ONNX_SHA256: Final = "16de7ea1146ca427e14938ec3e9abfdcaff0e6ac76434cd693ac35d761250bcb"
TOKENIZER_SHA256: Final = "249df0778f236f6ece390de0de746838ef25b9d6954b68c2ee71249e0a9d8fd4"

# Document fields are joined in this fixed order; missing/empty fields are skipped.
FIELD_ORDER: Final[tuple[str, ...]] = (
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

_NORMALIZATION: Final = "l2"
_BATCH_SIZE: Final = 32
_MAX_LENGTH: Final = 8192


@dataclass(frozen=True)
class EmbeddingContract:
    """Frozen identity of the pure BGE-M3 dense embedding used to build the catalog."""

    dimension: int
    version: str
    model_repo: str
    model_revision: str
    onnx_file: str
    tokenizer_file: str
    onnx_sha256: str
    tokenizer_sha256: str
    field_order: tuple[str, ...]
    normalization: str

    def public_metadata(self) -> dict[str, object]:
        return {
            "dimension": self.dimension,
            "version": self.version,
            "model_repo": self.model_repo,
            "model_revision": self.model_revision,
            "onnx_file": self.onnx_file,
            "tokenizer_file": self.tokenizer_file,
            "onnx_sha256": self.onnx_sha256,
            "tokenizer_sha256": self.tokenizer_sha256,
            "field_order": list(self.field_order),
            "normalization": self.normalization,
        }

    def as_json(self) -> dict[str, object]:
        return self.public_metadata()


def default_model_cache() -> Path:
    """Resolve the Hugging Face hub cache directory, honouring common env vars."""
    configured = os.environ.get("HF_HUB_CACHE")
    if configured:
        return Path(configured)
    hf_home = Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser()
    return hf_home / "hub"


def _snapshot_dir() -> Path:
    cache = default_model_cache()
    snapshot = (
        cache
        / f"models--{EMBEDDING_MODEL_REPO.replace('/', '--')}"
        / "snapshots"
        / EMBEDDING_MODEL_REVISION
    )
    if not snapshot.is_dir():
        raise RuntimeError(
            f"Missing local BGE ONNX snapshot: {snapshot}. "
            "Download it with huggingface-cli or the memsearch ONNX provider first."
        )
    return snapshot


def _verify_blob(path: Path, expected_sha256: str, name: str) -> Path:
    if not path.is_file():
        raise RuntimeError(f"Missing ONNX model file: {path}")
    resolved = path.resolve()
    if resolved.name != expected_sha256:
        raise RuntimeError(
            f"{name} content hash mismatch: expected blob {expected_sha256[:16]}…, "
            f"got {resolved.name[:16]}…"
        )
    return resolved


class _BgeEncoder:
    """Process-local lazy loader for the ONNX session and tokenizer."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._loaded = False
        self._session: Any = None
        self._tokenizer: Any = None
        self._output_names: list[str] = []

    def _load(self) -> None:
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            import onnxruntime as ort
            from tokenizers import Tokenizer

            snapshot = _snapshot_dir()
            onnx_path = _verify_blob(snapshot / ONNX_FILE, ONNX_SHA256, "onnx")
            tokenizer_path = _verify_blob(snapshot / TOKENIZER_FILE, TOKENIZER_SHA256, "tokenizer")

            tokenizer = Tokenizer.from_file(str(tokenizer_path))
            tokenizer.enable_padding(pad_id=1, pad_token="<pad>")
            tokenizer.enable_truncation(max_length=_MAX_LENGTH)

            options = ort.SessionOptions()
            # NOTE: leave thread counts at onnxruntime defaults. Multi-threaded and
            # single-threaded runs are bit-identical for this model (verified), and
            # the default intra-op parallelism is ~an order of magnitude faster for
            # a 24-layer xlm-roberta encoder than forcing one thread.
            session = ort.InferenceSession(str(onnx_path), sess_options=options)
            output_names = [output.name for output in session.get_outputs()]
            if "dense_vecs" not in output_names:
                raise RuntimeError(f"ONNX model lacks dense_vecs output: {output_names}")

            self._tokenizer = tokenizer
            self._session = session
            self._output_names = output_names
            self._loaded = True

    def encode(self, texts: list[str]) -> np.ndarray:
        self._load()
        assert self._session is not None and self._tokenizer is not None
        encoded = self._tokenizer.encode_batch(texts)
        input_ids = np.asarray([item.ids for item in encoded], dtype=np.int64)
        attention_mask = np.asarray([item.attention_mask for item in encoded], dtype=np.int64)
        outputs = self._session.run(
            None, {"input_ids": input_ids, "attention_mask": attention_mask}
        )
        dense = outputs[self._output_names.index("dense_vecs")]
        norms = np.linalg.norm(dense, axis=1, keepdims=True)
        return (dense / norms).astype(np.float32)


_ENCODER: Final = _BgeEncoder()


def _document_text(fields: Mapping[str, object]) -> str:
    parts: list[str] = []
    for field in FIELD_ORDER:
        value = fields.get(field)
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            for item in value:
                if isinstance(item, str) and item.strip():
                    parts.append(item.strip())
        elif isinstance(value, str) and value.strip():
            parts.append(value.strip())
    return " ".join(parts)


def build_contract(documents: Iterable[Mapping[str, object]]) -> EmbeddingContract:
    """Return the fixed pure-BGE-M3 contract (documents only validate non-emptiness)."""
    documents = list(documents)
    if not documents:
        raise ValueError("At least one catalog document is required")
    return EmbeddingContract(
        dimension=EMBEDDING_DIMENSION,
        version=EMBEDDING_VERSION,
        model_repo=EMBEDDING_MODEL_REPO,
        model_revision=EMBEDDING_MODEL_REVISION,
        onnx_file=ONNX_FILE,
        tokenizer_file=TOKENIZER_FILE,
        onnx_sha256=ONNX_SHA256,
        tokenizer_sha256=TOKENIZER_SHA256,
        field_order=FIELD_ORDER,
        normalization=_NORMALIZATION,
    )


def load_contract(value: Mapping[str, object]) -> EmbeddingContract:
    expected = build_contract([{"title": "probe"}]).public_metadata()
    if set(value) != set(expected):
        raise ValueError("Embedding contract keys are invalid")
    if any(value[key] != expected[key] for key in expected):
        raise ValueError("Embedding contract does not match the fixed BGE-M3 implementation")
    return EmbeddingContract(
        dimension=int(value["dimension"]),
        version=str(value["version"]),
        model_repo=str(value["model_repo"]),
        model_revision=str(value["model_revision"]),
        onnx_file=str(value["onnx_file"]),
        tokenizer_file=str(value["tokenizer_file"]),
        onnx_sha256=str(value["onnx_sha256"]),
        tokenizer_sha256=str(value["tokenizer_sha256"]),
        field_order=tuple(str(item) for item in value["field_order"]),  # type: ignore[arg-type]
        normalization=str(value["normalization"]),
    )


def encode_document(fields: Mapping[str, object], contract: EmbeddingContract) -> tuple[float, ...]:
    text = _document_text(fields)
    if not text:
        raise ValueError("Document has no text to embed")
    dense = _ENCODER.encode([text])[0]
    return tuple(round(float(value), 8) for value in dense)


def encode_query(query_text: str, contract: EmbeddingContract) -> tuple[float, ...]:
    if not query_text.strip():
        raise ValueError("Query text cannot be empty")
    dense = _ENCODER.encode([query_text.strip()])[0]
    return tuple(round(float(value), 8) for value in dense)


def cosine(left: Iterable[float], right: Iterable[float]) -> float:
    left_values = tuple(left)
    right_values = tuple(right)
    if len(left_values) != len(right_values):
        raise ValueError("Cosine vectors must have the same dimension")
    return round(sum(a * b for a, b in zip(left_values, right_values, strict=True)), 8)
