from __future__ import annotations

import pytest

from structarray_hybrid_demo.config import RuntimeConfig
from structarray_hybrid_demo.data import DatasetBundle, build_dataset


@pytest.fixture(scope="session")
def config() -> RuntimeConfig:
    return RuntimeConfig.from_environment()


@pytest.fixture(scope="session")
def covla_data_present(config: RuntimeConfig) -> bool:
    return config.sample_path.is_file() and config.prefix_path.is_file()


@pytest.fixture(scope="session")
def bundle(config: RuntimeConfig, covla_data_present: bool) -> DatasetBundle:
    if not covla_data_present:
        pytest.skip("CoVLA data is not committed; set COVLA_DATA_DIR to run data-backed tests")
    return build_dataset(config)
