from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from structarray_hybrid_demo.config import RuntimeConfig
from structarray_hybrid_demo.evidence import EvidencePathError, EvidenceResolver
from structarray_hybrid_demo.main import create_app


@pytest.fixture
def evidence_config(tmp_path: Path) -> RuntimeConfig:
    raw_dir = tmp_path / "yolo_result/behavior_frames_output/frames"
    annotated_dir = tmp_path / "yolo_result/draw_boxes/frame_wise"
    raw_dir.mkdir(parents=True)
    annotated_dir.mkdir(parents=True)
    (raw_dir / "0000b7dc6478371b_0-22_lane_keep_frame_000000.jpg").write_bytes(b"\xff\xd8fake")
    (annotated_dir / "0000b7dc6478371b_0-22_lane_keep_frame_000000_annotated.jpg").write_bytes(
        b"\xff\xd8fake"
    )
    sample = tmp_path / "video_data_100_samples.json"
    prefix = tmp_path / "video_data_10_samples.json"
    sample.write_text("[]")
    prefix.write_text("[]")
    return replace(
        RuntimeConfig.from_environment(),
        data_root=tmp_path,
        sample_path=sample,
        prefix_path=prefix,
    )


def test_resolver_rejects_unsafe_names(evidence_config: RuntimeConfig) -> None:
    resolver = EvidenceResolver(
        config=evidence_config, allowed_names=["0000b7dc6478371b_0-22_lane_keep_frame_000000.jpg"]
    )
    with pytest.raises(EvidencePathError, match="kind"):
        resolver.resolve(kind="evil", file_name="anything.jpg")
    with pytest.raises(EvidencePathError, match="not allowlisted"):
        resolver.resolve(kind="raw", file_name="other_frame.jpg")
    with pytest.raises(EvidencePathError, match="safe frame name"):
        resolver.resolve(
            kind="raw",
            file_name="../0000b7dc6478371b_0-22_lane_keep_frame_000000.jpg",
        )


def test_resolver_resolves_allowlisted_frame(evidence_config: RuntimeConfig) -> None:
    resolver = EvidenceResolver(
        config=evidence_config,
        allowed_names=[
            "0000b7dc6478371b_0-22_lane_keep_frame_000000.jpg",
            "0000b7dc6478371b_0-22_lane_keep_frame_000000_annotated.jpg",
        ],
    )
    raw = resolver.resolve(kind="raw", file_name="0000b7dc6478371b_0-22_lane_keep_frame_000000.jpg")
    annotated = resolver.resolve(
        kind="annotated",
        file_name="0000b7dc6478371b_0-22_lane_keep_frame_000000_annotated.jpg",
    )
    assert raw.name.endswith(".jpg")
    assert "_annotated" in annotated.name


def test_evidence_endpoint_serves_allowlisted_frame(evidence_config: RuntimeConfig) -> None:
    class EvidenceService:
        def __init__(self, config: RuntimeConfig) -> None:
            self.config = config

        def evidence_frame_names(self) -> list[str]:
            return [
                "0000b7dc6478371b_0-22_lane_keep_frame_000000.jpg",
                "0000b7dc6478371b_0-22_lane_keep_frame_000000_annotated.jpg",
            ]

    app = create_app(service=EvidenceService(evidence_config))
    client = TestClient(app)

    ok = client.get("/api/v1/evidence/raw/0000b7dc6478371b_0-22_lane_keep_frame_000000.jpg")
    assert ok.status_code == 200
    assert ok.headers["content-type"] == "image/jpeg"

    missing = client.get("/api/v1/evidence/raw/not_allowed.jpg")
    assert missing.status_code == 404
