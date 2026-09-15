from __future__ import annotations

import pytest

from structarray_hybrid_demo.config import RuntimeConfig
from structarray_hybrid_demo.data import DatasetBundle, build_dataset


@pytest.fixture(scope="session")
def config() -> RuntimeConfig:
    return RuntimeConfig.from_environment()


@pytest.fixture(scope="session")
def bundle(config: RuntimeConfig) -> DatasetBundle:
    return build_dataset(config)
