from __future__ import annotations

from pathlib import Path

import pytest
import torch

from embedding_list_demo.config import (
    MODEL_ADAPTER_REVISION,
    MODEL_BASE_ID,
    MODEL_BASE_REVISION,
    MODEL_DTYPE,
    MODEL_ID,
    VECTOR_DIMENSION,
    ConfigurationError,
    RuntimeConfig,
    default_dataset_root,
    default_hf_hub_cache,
)
from embedding_list_demo.model import ModelContractError, nonzero_vectors, resolve_model_identity


def _hub_directory(model_id: str) -> str:
    return "models--" + model_id.replace("/", "--")


def _offline_model_present() -> bool:
    hub = default_hf_hub_cache()
    adapter = hub / _hub_directory(MODEL_ID) / "snapshots" / MODEL_ADAPTER_REVISION
    base = hub / _hub_directory(MODEL_BASE_ID) / "snapshots" / MODEL_BASE_REVISION
    return adapter.is_dir() and base.is_dir()


@pytest.mark.skipif(
    not _offline_model_present(),
    reason="Local offline ColSmol model cache is not installed",
)
def test_installed_offline_model_identity_is_exact(tmp_path: Path) -> None:
    config = RuntimeConfig(
        runtime_root=tmp_path,
        hf_hub_cache=default_hf_hub_cache(),
        device="cpu",
    )

    identity = resolve_model_identity(config)

    assert identity.adapter.revision == MODEL_ADAPTER_REVISION
    assert identity.base.revision == MODEL_BASE_REVISION
    assert identity.inference_dtype == MODEL_DTYPE
    assert identity.inference_device == "cpu"
    assert identity.vector_dimension == VECTOR_DIMENSION
    assert Path(identity.adapter.snapshot_path).is_dir()
    assert Path(identity.base.snapshot_path).is_dir()


def test_nonzero_vectors_filters_padding_and_records_float32() -> None:
    source = torch.zeros((3, 128), dtype=torch.float16)
    source[1, 0] = 1
    source[2, 1] = 2

    result = nonzero_vectors(source)

    assert result.shape == (2, 128)
    assert result.dtype == torch.float32
    assert result.device.type == "cpu"
    assert result.is_contiguous()


def test_nonzero_vectors_rejects_wrong_dimension_and_empty() -> None:
    with pytest.raises(ModelContractError, match="Unexpected embedding shape"):
        nonzero_vectors(torch.ones((2, 64)))
    with pytest.raises(ModelContractError, match="no finite nonzero"):
        nonzero_vectors(torch.zeros((2, 128)))


def test_environment_configuration_is_explicitly_cpu_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("COLSMOL_DEVICE", raising=False)
    monkeypatch.delenv("EMBEDDING_LIST_DATASET_DIR", raising=False)

    config = RuntimeConfig.from_environment()

    assert config.device == "cpu"
    assert config.dataset_root == default_dataset_root()

    monkeypatch.setenv("COLSMOL_DEVICE", "cuda:0")
    with pytest.raises(ConfigurationError, match="must be cpu"):
        RuntimeConfig.from_environment()

    with pytest.raises(ConfigurationError, match="must be cpu"):
        RuntimeConfig(
            runtime_root=default_dataset_root(),
            hf_hub_cache=default_hf_hub_cache(),
            device="cuda:0",
        )
