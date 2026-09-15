"""Deterministic loader for the checked-in synthetic driving dataset."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from structarray_hybrid_demo.config import (
    EXPECTED_DATASET_ID,
    EXPECTED_DATASET_VERSION,
    EXPECTED_OBSERVATION_COUNT,
    EXPECTED_VIDEO_COUNT,
    FRAME_DIRECTORY,
    OBSERVATIONS_MAX_CAPACITY,
    RuntimeConfig,
)


class DataContractError(ValueError):
    """Raised when the synthetic manifest or its assets violate the fixed contract."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _required_str(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DataContractError(f"{field} must be a non-empty string")
    return value.strip()


def _required_int(value: object, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise DataContractError(f"{field} must be an integer")
    return value


@dataclass(frozen=True)
class ObservationRecord:
    index: int
    description: str
    object_type: str
    frame_id: int
    image_id: str
    bbox: tuple[int, int, int, int]
    vehicle_type: str
    color: str
    orientation: str
    lights_on: str
    v_ego: float
    a_ego: float
    clip_id: str
    raw_frame_file: str | None
    annotated_frame_file: str | None

    @property
    def has_evidence(self) -> bool:
        return self.raw_frame_file is not None and self.annotated_frame_file is not None


@dataclass(frozen=True)
class VideoRecord:
    video_id: str
    video_summary: str
    source_ordinal: int
    observations: tuple[ObservationRecord, ...]


@dataclass(frozen=True)
class DatasetBundle:
    videos: tuple[VideoRecord, ...]
    dataset_id: str
    dataset_version: int
    manifest_sha256: str
    evidence_frame_count: int

    @property
    def observation_count(self) -> int:
        return sum(len(video.observations) for video in self.videos)

    @property
    def video_count(self) -> int:
        return len(self.videos)


def _parse_object(value: object, field: str) -> tuple[str, str, str, str, str, tuple[int, ...]]:
    if not isinstance(value, list) or len(value) != 6:
        raise DataContractError(f"{field} must contain six object fields")
    object_type = _required_str(value[0], f"{field}.object_type")
    color = _required_str(value[1], f"{field}.color")
    vehicle_type = _required_str(value[2], f"{field}.vehicle_type")
    orientation = _required_str(value[3], f"{field}.orientation")
    lights_on = _required_str(value[4], f"{field}.lights_on")
    bbox_value = value[5]
    if not isinstance(bbox_value, list) or len(bbox_value) != 4:
        raise DataContractError(f"{field}.bbox must contain four integers")
    bbox = tuple(_required_int(item, f"{field}.bbox") for item in bbox_value)
    x1, y1, x2, y2 = bbox
    if not (0 <= x1 < x2 <= 960 and 0 <= y1 < y2 <= 420):
        raise DataContractError(f"{field}.bbox must stay inside the 960x420 frame")
    return object_type, color, vehicle_type, orientation, lights_on, bbox


def _parse_phases(raw: object) -> tuple[tuple[int, str], ...]:
    if not isinstance(raw, list) or not raw:
        raise DataContractError("observation_phases must be a non-empty list")
    phases: list[tuple[int, str]] = []
    for index, phase in enumerate(raw):
        if not isinstance(phase, dict):
            raise DataContractError(f"observation_phases[{index}] must be an object")
        phases.append(
            (
                _required_int(phase.get("frame_offset"), "frame_offset"),
                _required_str(phase.get("motion"), "motion"),
            )
        )
    return tuple(phases)


def _parse_scene(
    raw: object,
    source_ordinal: int,
    phases: tuple[tuple[int, str], ...],
    frame_directory: Path,
) -> VideoRecord:
    if not isinstance(raw, dict):
        raise DataContractError(f"scenes[{source_ordinal}] must be an object")
    video_id = _required_str(raw.get("video_id"), "video_id")
    if video_id != f"synthetic-drive-{source_ordinal + 1:03d}":
        raise DataContractError(f"Unexpected ordered video_id: {video_id}")
    summary = _required_str(raw.get("summary"), "summary")
    if "||" not in summary or "road type:" not in summary.lower():
        raise DataContractError(f"Summary lacks the parent-search structure: {video_id}")
    _required_str(raw.get("image_prompt"), "image_prompt")

    image_name = _required_str(raw.get("image"), "image")
    if image_name != Path(image_name).name or not image_name.endswith(".jpg"):
        raise DataContractError(f"Unsafe image name: {image_name}")
    image_path = frame_directory / image_name
    if not image_path.is_file():
        raise DataContractError(f"Missing synthetic frame: {image_path}")

    raw_objects = raw.get("objects")
    if not isinstance(raw_objects, list) or not raw_objects:
        raise DataContractError(f"objects must be a non-empty list: {video_id}")

    observations: list[ObservationRecord] = []
    for phase_index, (frame_offset, motion) in enumerate(phases):
        frame_id = (source_ordinal + 1) * 1000 + frame_offset
        for object_index, raw_object in enumerate(raw_objects):
            object_type, color, vehicle_type, orientation, lights_on, bbox = _parse_object(
                raw_object, f"{video_id}.objects[{object_index}]"
            )
            observation_index = phase_index * len(raw_objects) + object_index
            description = (
                f"A {color} {vehicle_type} {motion} {orientation}; "
                f"the {object_type} has its lights {lights_on}."
            )
            observations.append(
                ObservationRecord(
                    index=observation_index,
                    description=description,
                    object_type=object_type,
                    frame_id=frame_id,
                    image_id=f"{video_id}-representative-frame",
                    bbox=bbox,
                    vehicle_type=vehicle_type,
                    color=color,
                    orientation=orientation,
                    lights_on=lights_on,
                    v_ego=round(8.0 + source_ordinal * 0.35 + phase_index * 0.8, 2),
                    a_ego=(-0.4, 0.1, 0.35)[phase_index % 3],
                    clip_id=f"{video_id}-clip-{phase_index + 1}",
                    raw_frame_file=image_name,
                    annotated_frame_file=image_name,
                )
            )

    if len(observations) > OBSERVATIONS_MAX_CAPACITY:
        raise DataContractError(f"Too many observations for {video_id}: {len(observations)}")
    return VideoRecord(
        video_id=video_id,
        video_summary=summary,
        source_ordinal=source_ordinal,
        observations=tuple(observations),
    )


def build_dataset(config: RuntimeConfig) -> DatasetBundle:
    """Load and validate the original synthetic manifest and representative frames."""
    if not config.manifest_path.is_file():
        raise DataContractError(f"Missing synthetic manifest: {config.manifest_path}")
    try:
        manifest = json.loads(config.manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DataContractError(f"Invalid synthetic manifest JSON: {exc}") from exc
    if not isinstance(manifest, dict):
        raise DataContractError("Synthetic manifest must be a JSON object")

    dataset_id = _required_str(manifest.get("dataset_id"), "dataset_id")
    dataset_version = _required_int(manifest.get("version"), "version")
    if dataset_id != EXPECTED_DATASET_ID or dataset_version != EXPECTED_DATASET_VERSION:
        raise DataContractError(
            f"Unexpected synthetic dataset identity: {dataset_id}@{dataset_version}"
        )

    raw_scenes = manifest.get("scenes")
    if not isinstance(raw_scenes, list) or len(raw_scenes) != EXPECTED_VIDEO_COUNT:
        raise DataContractError(f"Synthetic manifest must contain {EXPECTED_VIDEO_COUNT} scenes")
    phases = _parse_phases(manifest.get("observation_phases"))
    frame_directory = (config.data_root / FRAME_DIRECTORY).resolve()
    if not frame_directory.is_dir():
        raise DataContractError(f"Missing synthetic frame directory: {frame_directory}")
    videos = tuple(
        _parse_scene(scene, index, phases, frame_directory)
        for index, scene in enumerate(raw_scenes)
    )
    image_names = {
        observation.annotated_frame_file
        for video in videos
        for observation in video.observations
        if observation.annotated_frame_file is not None
    }
    if len(image_names) != EXPECTED_VIDEO_COUNT:
        raise DataContractError(
            f"Expected {EXPECTED_VIDEO_COUNT} distinct frames, got {len(image_names)}"
        )
    bundle = DatasetBundle(
        videos=videos,
        dataset_id=dataset_id,
        dataset_version=dataset_version,
        manifest_sha256=_sha256(config.manifest_path),
        evidence_frame_count=len(image_names),
    )
    if bundle.observation_count != EXPECTED_OBSERVATION_COUNT:
        raise DataContractError(
            f"Expected {EXPECTED_OBSERVATION_COUNT} observations, got {bundle.observation_count}"
        )
    return bundle
