from __future__ import annotations

from pathlib import Path

import pytest
import torch

from embedding_list_demo.cache import CacheContractError, PageEmbeddingCache, cache_key
from embedding_list_demo.config import PAGE_COUNT, default_dataset_root
from embedding_list_demo.manual import load_manual
from embedding_list_demo.model import ModelIdentity, SnapshotRecord


def fake_identity() -> ModelIdentity:
    adapter = SnapshotRecord("adapter", "a" * 40, "c" * 40, "/adapter")
    base = SnapshotRecord("base", "b" * 40, "c" * 40, "/base")
    return ModelIdentity(
        model_id="test-model",
        adapter=adapter,
        base=base,
        inference_dtype="float32",
        inference_device="cpu",
        vector_dimension=128,
        colpali_engine_version="test",
        torch_version="test",
        transformers_version="test",
    )


def vectors() -> list[torch.Tensor]:
    return [torch.full((index + 2, 128), float(index + 1)) for index in range(PAGE_COUNT)]


def test_embedding_cache_round_trip_and_manifest(tmp_path: Path) -> None:
    manual = load_manual(default_dataset_root())
    identity = fake_identity()
    cache = PageEmbeddingCache(tmp_path / "cache")

    miss = cache.load(manual, identity)
    manifest = cache.save(
        manual=manual,
        identity=identity,
        vectors=vectors(),
        inference_report={"vector_counts": list(range(2, PAGE_COUNT + 2))},
    )
    hit = cache.load(manual, identity)

    assert miss.hit is False
    assert hit.hit is True
    assert hit.manifest == manifest
    assert manifest["cache_key"] == cache_key(manual, identity)
    assert hit.vectors is not None
    assert [list(tensor.shape) for tensor in hit.vectors] == [
        [index + 2, 128] for index in range(PAGE_COUNT)
    ]
    assert manifest["pages"][0]["pdf_page_index"] == 64  # type: ignore[index]
    assert manifest["pages"][0]["printed_page"] == 51  # type: ignore[index]
    assert cache.purge() is True
    assert cache.purge() is False


def test_embedding_cache_rejects_tensor_tampering(tmp_path: Path) -> None:
    manual = load_manual(default_dataset_root())
    identity = fake_identity()
    cache = PageEmbeddingCache(tmp_path / "cache")
    cache.save(manual=manual, identity=identity, vectors=vectors(), inference_report={})
    tensor_path = cache.root / "page-embeddings.safetensors"
    tensor_path.write_bytes(tensor_path.read_bytes() + b"tamper")

    with pytest.raises(CacheContractError, match="SHA-256"):
        cache.load(manual, identity)


def test_embedding_cache_refuses_overwrite(tmp_path: Path) -> None:
    manual = load_manual(default_dataset_root())
    identity = fake_identity()
    cache = PageEmbeddingCache(tmp_path / "cache")
    cache.save(manual=manual, identity=identity, vectors=vectors(), inference_report={})

    with pytest.raises(CacheContractError, match="Refusing to overwrite"):
        cache.save(manual=manual, identity=identity, vectors=vectors(), inference_report={})
