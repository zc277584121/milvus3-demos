from __future__ import annotations

import pytest

from minhash_near_dup_demo.config import RuntimeConfig
from minhash_near_dup_demo.data import DatasetBundle, build_dataset


@pytest.fixture(scope="session")
def config() -> RuntimeConfig:
    return RuntimeConfig.from_environment()


@pytest.fixture(scope="session")
def bundle(config: RuntimeConfig) -> DatasetBundle:
    return build_dataset(config)
