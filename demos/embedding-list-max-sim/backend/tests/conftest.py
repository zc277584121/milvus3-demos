from __future__ import annotations

from pathlib import Path

import pytest

from embedding_list_demo.config import RuntimeConfig, default_hf_hub_cache


@pytest.fixture
def runtime_config(tmp_path: Path) -> RuntimeConfig:
    return RuntimeConfig(
        runtime_root=tmp_path / "runtime",
        hf_hub_cache=default_hf_hub_cache(),
        device="cpu",
    )
