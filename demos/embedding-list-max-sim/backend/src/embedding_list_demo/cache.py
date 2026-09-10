"""Validated on-disk cache for ColSmol page multi-vectors."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from embedding_list_demo.config import VECTOR_DIMENSION
from embedding_list_demo.manual import ManualManifest, sha256_file
from embedding_list_demo.model import ModelIdentity

CACHE_MANIFEST_FILENAME = "embedding-manifest.json"
CACHE_TENSOR_FILENAME = "page-embeddings.safetensors"


class CacheContractError(RuntimeError):
    """Raised when an existing cache cannot be trusted or safely replaced."""


@dataclass(frozen=True)
class CacheLoad:
    hit: bool
    vectors: list[torch.Tensor] | None
    manifest: dict[str, object] | None


def _cache_key_payload(manual: ManualManifest, identity: ModelIdentity) -> dict[str, object]:
    return {
        "schema_version": 2,
        "dataset": {
            "dataset_id": manual.dataset_id,
            "revision": manual.revision,
            "manifest_sha256": manual.manifest_sha256,
            "source_spec_sha256": manual.source_spec_sha256,
            "pdf_sha256": manual.pdf_sha256,
            "queries_sha256": manual.queries_sha256,
            "render_dpi": manual.render_dpi,
            "pdf_page_index_base": manual.pdf_page_index_base,
            "generator_version": manual.generator_version,
            "pages": [
                {"page_id": page.page_id, "image_sha256": page.image_sha256}
                for page in manual.pages
            ],
        },
        "model": identity.public_dict(),
        "storage_dtype": "float32",
        "vector_dimension": VECTOR_DIMENSION,
    }


def cache_key(manual: ManualManifest, identity: ModelIdentity) -> str:
    """Return the canonical cache identity for manual and model inputs."""

    payload = json.dumps(
        _cache_key_payload(manual, identity),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


class PageEmbeddingCache:
    """Read, write, validate, and purge one exact derived cache directory."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def load(self, manual: ManualManifest, identity: ModelIdentity) -> CacheLoad:
        """Return a validated cache hit, a clean miss, or fail on corrupt residue."""

        if not self.root.exists():
            return CacheLoad(hit=False, vectors=None, manifest=None)
        if self.root.is_symlink() or not self.root.is_dir():
            raise CacheContractError(f"Unsafe embedding cache root: {self.root}")
        manifest_path = self.root / CACHE_MANIFEST_FILENAME
        tensor_path = self.root / CACHE_TENSOR_FILENAME
        if not manifest_path.is_file() or not tensor_path.is_file():
            raise CacheContractError("Embedding cache directory is incomplete")
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_key = cache_key(manual, identity)
        if value.get("cache_key") != expected_key:
            raise CacheContractError("Embedding cache key differs from the current inputs")
        if value.get("tensor_sha256") != sha256_file(tensor_path):
            raise CacheContractError("Embedding cache tensor SHA-256 verification failed")
        tensors = load_file(tensor_path, device="cpu")
        expected_pages = [page.page_id for page in manual.pages]
        if sorted(tensors) != sorted(expected_pages):
            raise CacheContractError("Embedding cache tensor keys differ from manual page IDs")
        vectors: list[torch.Tensor] = []
        manifest_pages = value.get("pages")
        if not isinstance(manifest_pages, list):
            raise CacheContractError("Embedding cache page manifest has an unexpected shape")
        page_metadata = {str(item["page_id"]): item for item in manifest_pages}
        for page_id in expected_pages:
            tensor = tensors[page_id].to(dtype=torch.float32).contiguous()
            metadata = page_metadata.get(page_id)
            if metadata is None:
                raise CacheContractError(f"Embedding cache metadata is missing: {page_id}")
            if list(tensor.shape) != metadata.get("shape"):
                raise CacheContractError(f"Embedding cache shape differs: {page_id}")
            if tensor.ndim != 2 or tensor.shape[1] != VECTOR_DIMENSION:
                raise CacheContractError(f"Embedding cache dimension differs: {page_id}")
            if not torch.isfinite(tensor).all():
                raise CacheContractError(f"Embedding cache contains non-finite values: {page_id}")
            vectors.append(tensor)
        return CacheLoad(hit=True, vectors=vectors, manifest=value)

    def save(
        self,
        *,
        manual: ManualManifest,
        identity: ModelIdentity,
        vectors: list[torch.Tensor],
        inference_report: dict[str, object],
    ) -> dict[str, object]:
        """Atomically publish page tensors and a complete cache manifest."""

        if self.root.exists() or self.root.is_symlink():
            raise CacheContractError(f"Refusing to overwrite embedding cache: {self.root}")
        if len(vectors) != len(manual.pages):
            raise CacheContractError("Embedding tensor count differs from manual page count")
        self.root.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=".embedding-cache-", dir=self.root.parent))
        try:
            tensor_path = temporary / CACHE_TENSOR_FILENAME
            tensors: dict[str, torch.Tensor] = {}
            pages: list[dict[str, object]] = []
            for page, vector in zip(manual.pages, vectors, strict=True):
                tensor = vector.detach().to(device="cpu", dtype=torch.float32).contiguous()
                if tensor.ndim != 2 or tensor.shape[1] != VECTOR_DIMENSION:
                    raise CacheContractError(f"Cannot cache unexpected tensor: {page.page_id}")
                tensors[page.page_id] = tensor
                pages.append(
                    {
                        "page_id": page.page_id,
                        "pdf_page_index": page.pdf_page_index,
                        "printed_page": page.printed_page,
                        "page_image_sha256": page.image_sha256,
                        "shape": list(tensor.shape),
                        "storage_dtype": "float32",
                    }
                )
            save_file(tensors, tensor_path)
            manifest: dict[str, object] = {
                "schema_version": 2,
                "cache_key": cache_key(manual, identity),
                "dataset_id": manual.dataset_id,
                "dataset_revision": manual.revision,
                "dataset_manifest_sha256": manual.manifest_sha256,
                "dataset_pdf_sha256": manual.pdf_sha256,
                "model": identity.public_dict(),
                "inference": inference_report,
                "tensor_filename": CACHE_TENSOR_FILENAME,
                "tensor_sha256": sha256_file(tensor_path),
                "pages": pages,
            }
            (temporary / CACHE_MANIFEST_FILENAME).write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, self.root)
            return manifest
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise

    def purge(self) -> bool:
        """Delete only this validated derived cache directory."""

        if not self.root.exists():
            return False
        if self.root.is_symlink() or not self.root.is_dir():
            raise CacheContractError(f"Refusing to purge unsafe embedding cache: {self.root}")
        shutil.rmtree(self.root)
        return True
