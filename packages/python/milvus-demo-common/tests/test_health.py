from __future__ import annotations

from typing import Any

from milvus_demo_common.health import PyMilvusHealthProbe
from milvus_demo_common.settings import MilvusSettings


class FakeClient:
    def __init__(self, version: str = "3.0.0", error: Exception | None = None) -> None:
        self.version = version
        self.error = error
        self.closed = False

    def get_server_version(self, timeout: float | None = None) -> str:
        if self.error is not None:
            raise self.error
        return self.version

    def close(self) -> None:
        self.closed = True


def test_probe_accepts_exact_milvus_version() -> None:
    client = FakeClient()
    captured: dict[str, Any] = {}

    def factory(**kwargs: Any) -> FakeClient:
        captured.update(kwargs)
        return client

    settings = MilvusSettings(
        milvus_uri="http://milvus.test:19530",
        milvus_token="private-token",
    )
    result = PyMilvusHealthProbe(settings, factory).check()

    assert result.healthy is True
    assert result.server_version == "3.0.0"
    assert result.error_code is None
    assert captured["token"] == "private-token"
    assert client.closed is True


def test_probe_rejects_wrong_version() -> None:
    client = FakeClient(version="3.0-beta")
    result = PyMilvusHealthProbe(MilvusSettings(), lambda **_: client).check()

    assert result.healthy is False
    assert result.server_version == "3.0-beta"
    assert result.error_code == "version_mismatch"
    assert client.closed is True


def test_probe_sanitizes_connection_errors() -> None:
    client = FakeClient(error=RuntimeError("token=do-not-leak"))
    result = PyMilvusHealthProbe(MilvusSettings(), lambda **_: client).check()

    assert result.healthy is False
    assert result.server_version is None
    assert result.error_code == "RuntimeError"
    assert "do-not-leak" not in repr(result)
    assert client.closed is True


def test_settings_repr_masks_token() -> None:
    settings = MilvusSettings(milvus_token="private-token")

    assert "private-token" not in repr(settings)
    assert settings.token_value() == "private-token"
