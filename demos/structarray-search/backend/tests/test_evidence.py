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
    image_dir = tmp_path / "images"
    image_dir.mkdir(parents=True)
    (image_dir / "synthetic-drive-001.jpg").write_bytes(b"\xff\xd8fake")
    manifest = tmp_path / "scenes.json"
    manifest.write_text("{}")
    return replace(
        RuntimeConfig.from_environment(),
        data_root=tmp_path,
        manifest_path=manifest,
    )


def test_resolver_rejects_unsafe_names(evidence_config: RuntimeConfig) -> None:
    resolver = EvidenceResolver(config=evidence_config, allowed_names=["synthetic-drive-001.jpg"])
    with pytest.raises(EvidencePathError, match="kind"):
        resolver.resolve(kind="evil", file_name="anything.jpg")
    with pytest.raises(EvidencePathError, match="not allowlisted"):
        resolver.resolve(kind="raw", file_name="other_frame.jpg")
    with pytest.raises(EvidencePathError, match="safe frame name"):
        resolver.resolve(
            kind="raw",
            file_name="../synthetic-drive-001.jpg",
        )


def test_resolver_resolves_allowlisted_frame(evidence_config: RuntimeConfig) -> None:
    resolver = EvidenceResolver(
        config=evidence_config,
        allowed_names=["synthetic-drive-001.jpg"],
    )
    raw = resolver.resolve(kind="raw", file_name="synthetic-drive-001.jpg")
    annotated = resolver.resolve(kind="annotated", file_name="synthetic-drive-001.jpg")
    assert raw == annotated


def test_evidence_endpoint_serves_allowlisted_frame(evidence_config: RuntimeConfig) -> None:
    class EvidenceService:
        def __init__(self, config: RuntimeConfig) -> None:
            self.config = config

        def evidence_frame_names(self) -> list[str]:
            return ["synthetic-drive-001.jpg"]

    app = create_app(service=EvidenceService(evidence_config))
    client = TestClient(app)

    ok = client.get("/api/v1/evidence/raw/synthetic-drive-001.jpg")
    assert ok.status_code == 200
    assert ok.headers["content-type"] == "image/jpeg"

    missing = client.get("/api/v1/evidence/raw/not_allowed.jpg")
    assert missing.status_code == 404
