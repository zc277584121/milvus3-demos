from __future__ import annotations

import pytest

from movie_discovery_demo.config import RuntimeConfig
from movie_discovery_demo.data import DatasetBundle, build_dataset


@pytest.fixture(scope="session")
def config() -> RuntimeConfig:
    return RuntimeConfig.from_environment()


@pytest.fixture(scope="session")
def bundle(config: RuntimeConfig) -> DatasetBundle:
    return build_dataset(config)
