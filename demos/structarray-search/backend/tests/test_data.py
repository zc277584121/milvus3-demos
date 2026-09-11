from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from structarray_hybrid_demo.config import (
    EXPECTED_PREFIX_SHA256,
    EXPECTED_SAMPLE_SHA256,
    SAMPLE_SIZE,
    RuntimeConfig,
)
from structarray_hybrid_demo.data import DataContractError, build_dataset


def test_runtime_config_keeps_sample_under_data_root() -> None:
    config = RuntimeConfig.from_environment()
    assert config.sample_path.is_relative_to(config.data_root)
    assert config.prefix_path.is_relative_to(config.data_root)
    assert config.sample_path.name == "video_data_100_samples.json"
    assert config.prefix_path.name == "video_data_10_samples.json"


def test_real_bundle_has_30_videos_and_some_observations(bundle, covla_data_present: bool) -> None:
    if not covla_data_present:
        pytest.skip("CoVLA data is not committed; set COVLA_DATA_DIR to run data-backed tests")
    assert bundle.video_count == 30
    assert bundle.sample_size == SAMPLE_SIZE
    assert bundle.sample_sha256 == EXPECTED_SAMPLE_SHA256
    assert bundle.prefix_sha256 == EXPECTED_PREFIX_SHA256
    assert bundle.observation_count > 500
    # At least 27 of 30 videos contribute at least one searchable observation.
    assert sum(1 for video in bundle.videos if len(video.observations) > 0) >= 27
    assert max(len(video.observations) for video in bundle.videos) <= 128


def test_prefix_preserves_10_record_order(config: RuntimeConfig, covla_data_present: bool) -> None:
    if not covla_data_present:
        pytest.skip("CoVLA data is not committed; set COVLA_DATA_DIR to run data-backed tests")
    bundle = build_dataset(config)
    first10_ids = [video.video_id for video in bundle.videos[:10]]
    # Rebuilt inside build_dataset already; assert monotonic ordinal mapping.
    assert [video.source_ordinal for video in bundle.videos[:10]] == list(range(10))
    assert len(set(first10_ids)) == 10


def test_data_contract_rejects_wrong_sample_sha(tmp_path: Path) -> None:
    sample = tmp_path / "video_data_100_samples.json"
    prefix = tmp_path / "video_data_10_samples.json"
    sample.write_text('[{"video_id": "0000000000000000", "video_summary": "x"}]')
    prefix.write_text("[]")
    config = replace(
        RuntimeConfig.from_environment(),
        data_root=tmp_path,
        sample_path=sample,
        prefix_path=prefix,
    )
    with pytest.raises(DataContractError, match="SHA-256 differs"):
        build_dataset(config)
