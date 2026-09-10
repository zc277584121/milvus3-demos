from __future__ import annotations

import json
from pathlib import Path

import pytest

from embedding_list_demo.cli import LifecycleContractError, _load_preflight
from embedding_list_demo.config import COLLECTION_NAME


def write_audit(path: Path, **overrides: object) -> None:
    value: dict[str, object] = {
        "server_version": "3.0.0",
        "expected_server_version": "3.0.0",
        "target_collection": COLLECTION_NAME,
        "target_exists": False,
        "raw_collection_names": ["unrelated"],
        "raw_file_resource_names": ["unrelated-resource"],
    }
    value.update(overrides)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_cleanup_preflight_requires_exact_absent_before_evidence(tmp_path: Path) -> None:
    audit = tmp_path / "raw-before.json"
    write_audit(audit)

    loaded = _load_preflight(audit)

    assert loaded["target_exists"] is False
    assert loaded["target_collection"] == COLLECTION_NAME


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"server_version": "3.0-beta"}, "server_version"),
        ({"target_collection": "wrong"}, "target_collection"),
        ({"target_exists": True}, "target_exists"),
        ({"raw_collection_names": "wrong"}, "raw SDK name lists"),
    ],
)
def test_cleanup_preflight_rejects_changed_or_unsafe_evidence(
    tmp_path: Path,
    overrides: dict[str, object],
    message: str,
) -> None:
    audit = tmp_path / "raw-before.json"
    write_audit(audit, **overrides)

    with pytest.raises(LifecycleContractError, match=message):
        _load_preflight(audit)
