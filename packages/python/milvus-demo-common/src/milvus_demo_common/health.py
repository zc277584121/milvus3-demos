"""Real Milvus server health probing with sanitized results."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from pymilvus import MilvusClient

from milvus_demo_common.settings import MilvusSettings


class MilvusClientProtocol(Protocol):
    """Small client surface required by the health probe."""

    def get_server_version(self, timeout: float | None = None) -> str | dict: ...

    def close(self) -> None: ...


ClientFactory = Callable[..., MilvusClientProtocol]


@dataclass(frozen=True, slots=True)
class MilvusHealthResult:
    """Sanitized result returned to API layers."""

    healthy: bool
    server_version: str | None
    latency_ms: float
    error_code: str | None = None


class PyMilvusHealthProbe:
    """Open a short-lived client and verify the exact server version."""

    def __init__(
        self,
        settings: MilvusSettings,
        client_factory: ClientFactory = MilvusClient,
    ) -> None:
        self._settings = settings
        self._client_factory = client_factory

    def check(self) -> MilvusHealthResult:
        """Return health without exposing the URI, token, or exception message."""

        started = time.perf_counter()
        client: MilvusClientProtocol | None = None
        try:
            client = self._client_factory(
                uri=self._settings.milvus_uri,
                token=self._settings.token_value(),
                timeout=self._settings.milvus_timeout_seconds,
            )
            raw_version = client.get_server_version(timeout=self._settings.milvus_timeout_seconds)
            if not isinstance(raw_version, str):
                raise TypeError("Milvus returned a non-string server version")
            healthy = raw_version == self._settings.milvus_expected_version
            return MilvusHealthResult(
                healthy=healthy,
                server_version=raw_version,
                latency_ms=self._elapsed_ms(started),
                error_code=None if healthy else "version_mismatch",
            )
        except Exception as exc:
            return MilvusHealthResult(
                healthy=False,
                server_version=None,
                latency_ms=self._elapsed_ms(started),
                error_code=type(exc).__name__,
            )
        finally:
            if client is not None:
                client.close()

    @staticmethod
    def _elapsed_ms(started: float) -> float:
        return round((time.perf_counter() - started) * 1000, 2)
