from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from structarray_hybrid_demo.config import (
    EXPECTED_DATASET_ID,
    EXPECTED_DATASET_VERSION,
    EXPECTED_OBSERVATION_COUNT,
    EXPECTED_VIDEO_COUNT,
    SAMPLE_SIZE,
    RuntimeConfig,
)
from structarray_hybrid_demo.data import DataContractError, build_dataset


def test_runtime_config_keeps_manifest_under_data_root() -> None:
    config = RuntimeConfig.from_environment()
    assert config.manifest_path.is_relative_to(config.data_root)
    assert config.manifest_path.name == "scenes.json"


def test_synthetic_bundle_has_fixed_videos_observations_and_frames() -> None:
    bundle = build_dataset(RuntimeConfig.from_environment())
    assert bundle.dataset_id == EXPECTED_DATASET_ID
    assert bundle.dataset_version == EXPECTED_DATASET_VERSION
    assert bundle.video_count == EXPECTED_VIDEO_COUNT == SAMPLE_SIZE
    assert bundle.observation_count == EXPECTED_OBSERVATION_COUNT
    assert bundle.evidence_frame_count == EXPECTED_VIDEO_COUNT
    assert all(len(video.observations) == 18 for video in bundle.videos)
    assert all(
        observation.raw_frame_file == observation.annotated_frame_file
        for video in bundle.videos
        for observation in video.observations
    )


def test_synthetic_ground_truth_has_three_videos_per_preset() -> None:
    bundle = build_dataset(RuntimeConfig.from_environment())
    groups = {
        "intersection-white-truck": ("intersection", "truck", "white"),
        "bridge-white-van": ("bridge", "van", "white"),
        "ramp-white-truck": ("ramp", "truck", "white"),
        "residential-black-suv": ("local_residential", "suv", "black"),
    }
    for scene, object_type, color in groups.values():
        matched = [
            video
            for video in bundle.videos
            if scene in video.video_summary.lower()
            and any(
                observation.object_type == object_type and observation.color == color
                for observation in video.observations
            )
        ]
        assert len(matched) == 3


def test_data_contract_rejects_wrong_dataset_identity(tmp_path: Path) -> None:
    images = tmp_path / "images"
    images.mkdir()
    manifest = tmp_path / "scenes.json"
    manifest.write_text('{"dataset_id":"wrong","version":1}', encoding="utf-8")
    config = replace(
        RuntimeConfig.from_environment(),
        data_root=tmp_path,
        manifest_path=manifest,
    )
    with pytest.raises(DataContractError, match="Unexpected synthetic dataset identity"):
        build_dataset(config)
