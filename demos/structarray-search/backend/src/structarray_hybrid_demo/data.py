"""Deterministic, read-only conversion of the approved CoVLA 30-video slice."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, replace
from pathlib import Path

from structarray_hybrid_demo.config import (
    ANNOTATED_FRAME_DIRECTORY,
    EXPECTED_MAX_OBSERVATIONS,
    EXPECTED_MIN_OBSERVATIONS,
    EXPECTED_PREFIX_SHA256,
    EXPECTED_SAMPLE_SHA256,
    EXPECTED_VIDEO_COUNT,
    RAW_FRAME_DIRECTORY,
    SAMPLE_OFFSET,
    SAMPLE_SIZE,
    RuntimeConfig,
)

FRAME_FILE_PATTERN = re.compile(
    r"^(?P<video_id>[0-9a-f]{16})_.*_frame_(?P<frame_id>\d{6})"
    r"(?P<annotated>_annotated)?\.jpg$"
)


class DataContractError(ValueError):
    """Raised when source data violates the approved deterministic contract."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _clean(value: object) -> object:
    """Strip the leading/trailing single quotes the 100-record file wraps scalars in."""
    if isinstance(value, str) and len(value) >= 2 and value[0] == "'" and value[-1] == "'":
        return value[1:-1]
    return value


def _clean_str(value: object, field: str) -> str:
    cleaned = _clean(value)
    if not isinstance(cleaned, str):
        raise DataContractError(f"{field} must be a string, got {type(cleaned).__name__}")
    return cleaned


def _clean_int(value: object, field: str) -> int:
    cleaned = _clean(value)
    try:
        return int(cleaned)
    except (TypeError, ValueError) as exc:
        raise DataContractError(f"{field} must be an integer, got {cleaned!r}") from exc


def _clean_float(value: object, field: str) -> float:
    cleaned = _clean(value)
    try:
        return float(cleaned)
    except (TypeError, ValueError) as exc:
        raise DataContractError(f"{field} must be a number, got {cleaned!r}") from exc


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
    sample_size: int
    sample_sha256: str
    prefix_sha256: str
    evidence_frame_count: int

    @property
    def observation_count(self) -> int:
        return sum(len(video.observations) for video in self.videos)

    @property
    def video_count(self) -> int:
        return len(self.videos)


def build_frame_index(config: RuntimeConfig) -> dict[tuple[str, int], str]:
    """Map (video_id, frame_id) → annotated file name; raw reuses the bare stem.

    Keyed on annotated file names (which always exist alongside raw files), so a
    single pass over one directory yields both assets.
    """
    directory = (config.data_root / ANNOTATED_FRAME_DIRECTORY).resolve()
    if not directory.is_dir():
        raise DataContractError(f"Missing annotated frame directory: {directory}")
    index: dict[tuple[str, int], str] = {}
    with os.scandir(directory) as entries:
        for entry in entries:
            match = FRAME_FILE_PATTERN.fullmatch(entry.name)
            if match is None:
                continue
            key = (match.group("video_id"), int(match.group("frame_id")))
            index.setdefault(key, entry.name)
    return index


def _parse_video(raw: dict[str, object], source_ordinal: int) -> VideoRecord:
    video_id = _clean_str(raw.get("video_id"), "video_id")
    if len(video_id) != 16 or any(char not in "0123456789abcdef" for char in video_id):
        raise DataContractError(f"Unexpected video_id shape: {video_id!r}")
    video_summary = _clean_str(raw.get("video_summary"), "video_summary")
    if not video_summary.strip():
        raise DataContractError(f"Empty video_summary for {video_id}")

    observations: list[ObservationRecord] = []
    for index, raw_observation in enumerate(raw.get("object_list", [])):
        if not isinstance(raw_observation, dict):
            raise DataContractError(f"object_list entry {index} is not an object")
        description = _clean_str(raw_observation.get("description"), "description").strip()
        if not description:
            continue  # empty-description objects carry no searchable text
        observation = ObservationRecord(
            index=index,
            description=description,
            object_type=_clean_str(raw_observation.get("object_type"), "object_type"),
            frame_id=_clean_int(raw_observation.get("frame_id"), "frame_id"),
            image_id=_clean_str(raw_observation.get("image_id"), "image_id"),
            bbox=(
                _clean_int(raw_observation.get("bbox_x1"), "bbox_x1"),
                _clean_int(raw_observation.get("bbox_y1"), "bbox_y1"),
                _clean_int(raw_observation.get("bbox_x2"), "bbox_x2"),
                _clean_int(raw_observation.get("bbox_y2"), "bbox_y2"),
            ),
            vehicle_type=_clean_str(raw_observation.get("vehicle_type"), "vehicle_type"),
            color=_clean_str(raw_observation.get("color"), "color"),
            orientation=_clean_str(raw_observation.get("orientation"), "orientation"),
            lights_on=_clean_str(raw_observation.get("lights_on"), "lights_on"),
            v_ego=_clean_float(raw_observation.get("vEgo"), "vEgo"),
            a_ego=_clean_float(raw_observation.get("aEgo"), "aEgo"),
            clip_id=_clean_str(raw_observation.get("clip_id"), "clip_id"),
            raw_frame_file=None,
            annotated_frame_file=None,
        )
        observations.append(observation)

    return VideoRecord(
        video_id=video_id,
        video_summary=video_summary,
        source_ordinal=source_ordinal,
        observations=tuple(observations),
    )


def _with_evidence(
    observation: ObservationRecord, raw_frame_file: str, annotated_frame_file: str
) -> ObservationRecord:
    return replace(
        observation, raw_frame_file=raw_frame_file, annotated_frame_file=annotated_frame_file
    )


def _with_observations(
    video: VideoRecord, observations: tuple[ObservationRecord, ...]
) -> VideoRecord:
    return replace(video, observations=observations)


def build_dataset(config: RuntimeConfig) -> DatasetBundle:
    """Load and verify the approved 100-record file and return its 30-video prefix."""
    if not config.sample_path.is_file():
        raise DataContractError(f"Missing CoVLA sample file: {config.sample_path}")
    if not config.prefix_path.is_file():
        raise DataContractError(f"Missing CoVLA prefix file: {config.prefix_path}")

    sample_sha256 = _sha256(config.sample_path)
    if sample_sha256 != EXPECTED_SAMPLE_SHA256:
        raise DataContractError(
            f"CoVLA sample SHA-256 differs: expected={EXPECTED_SAMPLE_SHA256}, "
            f"actual={sample_sha256}"
        )
    prefix_sha256 = _sha256(config.prefix_path)
    if prefix_sha256 != EXPECTED_PREFIX_SHA256:
        raise DataContractError(
            f"CoVLA prefix SHA-256 differs: expected={EXPECTED_PREFIX_SHA256}, "
            f"actual={prefix_sha256}"
        )

    sample = json.loads(config.sample_path.read_text(encoding="utf-8"))
    prefix = json.loads(config.prefix_path.read_text(encoding="utf-8"))
    if not isinstance(sample, list) or len(sample) < SAMPLE_SIZE:
        raise DataContractError("CoVLA sample must be a list of at least 30 videos")
    if not isinstance(prefix, list) or len(prefix) != 10:
        raise DataContractError("CoVLA prefix must contain exactly 10 videos")

    prefix_ids = [_clean_str(video.get("video_id"), "video_id") for video in prefix]
    sample_ids = [_clean_str(video.get("video_id"), "video_id") for video in sample[:10]]
    if sample_ids != prefix_ids:
        raise DataContractError("The 100-record file does not preserve the 10-record prefix")

    videos = tuple(
        _parse_video(video, ordinal)
        for ordinal, video in enumerate(sample[SAMPLE_OFFSET : SAMPLE_OFFSET + SAMPLE_SIZE])
    )
    annotated_index = build_frame_index(config)
    evidence_count = 0
    raw_directory = (config.data_root / RAW_FRAME_DIRECTORY).resolve()
    if not raw_directory.is_dir():
        raise DataContractError(f"Missing raw frame directory: {raw_directory}")
    resolved_videos: list[VideoRecord] = []
    for video in videos:
        resolved_observations: list[ObservationRecord] = []
        for observation in video.observations:
            annotated_file = annotated_index.get((video.video_id, observation.frame_id))
            if annotated_file is not None:
                raw_file = annotated_file.removesuffix("_annotated.jpg") + ".jpg"
                if not (raw_directory / raw_file).is_file():
                    raise DataContractError(f"Missing raw evidence frame: {raw_file}")
                observation = _with_evidence(observation, raw_file, annotated_file)
                evidence_count += 1
            resolved_observations.append(observation)
        resolved_videos.append(_with_observations(video, tuple(resolved_observations)))
    videos = tuple(resolved_videos)

    bundle = DatasetBundle(
        videos=videos,
        sample_size=SAMPLE_SIZE,
        sample_sha256=sample_sha256,
        prefix_sha256=prefix_sha256,
        evidence_frame_count=evidence_count,
    )
    if bundle.video_count != EXPECTED_VIDEO_COUNT:
        raise DataContractError(f"Expected {EXPECTED_VIDEO_COUNT} videos, got {bundle.video_count}")
    if not EXPECTED_MIN_OBSERVATIONS <= bundle.observation_count <= EXPECTED_MAX_OBSERVATIONS:
        raise DataContractError(
            f"Observation count out of approved range: {bundle.observation_count}"
        )
    return bundle
