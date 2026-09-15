"""Allowlisted, read-only evidence frame resolution for the hybrid demo."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from structarray_hybrid_demo.config import FRAME_DIRECTORY, RuntimeConfig


class EvidencePathError(ValueError):
    """Raised when a requested evidence path is not an inserted observation asset."""


class EvidenceResolver:
    """Resolve allowlisted raw/annotated frame file names to source paths."""

    def __init__(
        self,
        *,
        config: RuntimeConfig | None = None,
        allowed_names: Iterable[str] = (),
    ) -> None:
        self.config = config or RuntimeConfig.from_environment()
        self.allowed = frozenset(allowed_names)

    def resolve(self, *, kind: str, file_name: str) -> Path:
        if kind not in {"raw", "annotated"}:
            raise EvidencePathError("Evidence kind must be 'raw' or 'annotated'")
        if file_name != Path(file_name).name or not file_name.endswith(".jpg"):
            raise EvidencePathError("Evidence file name is not a safe frame name")
        if file_name not in self.allowed:
            raise EvidencePathError("Evidence path is not allowlisted by the prepared sample")

        data_root = self.config.data_root
        directory = (data_root / FRAME_DIRECTORY).resolve()
        resolved = (directory / file_name).resolve(strict=True)
        if not resolved.is_relative_to(directory) or not resolved.is_file():
            raise EvidencePathError("Evidence path escaped the read-only data root")
        return resolved
