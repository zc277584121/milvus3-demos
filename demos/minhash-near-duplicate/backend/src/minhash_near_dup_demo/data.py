"""Deterministic loader for the checked-in original-short-stories dataset."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from minhash_near_dup_demo.config import (
    CONTENT_MAX_LENGTH,
    DOCUMENTS_FILE_NAME,
    EXPECTED_DATASET_ID,
    EXPECTED_DATASET_VERSION,
    EXPECTED_DOCUMENT_COUNT,
    PRESETS_FILE_NAME,
    RuntimeConfig,
)


class DataContractError(ValueError):
    """Raised when the dataset or presets violate the fixed contract."""


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


@dataclass(frozen=True)
class DocumentRecord:
    doc_id: str
    title: str
    cluster: str
    variant: str
    content: str


@dataclass(frozen=True)
class PresetRecord:
    id: str
    label: str
    title: str
    text: str
    description: str

    def public_dict(self) -> dict[str, str]:
        return {
            "id": self.id,
            "label": self.label,
            "title": self.title,
            "text": self.text,
            "description": self.description,
        }


@dataclass(frozen=True)
class DatasetBundle:
    documents: tuple[DocumentRecord, ...]
    presets: tuple[PresetRecord, ...]
    dataset_id: str
    dataset_version: int
    documents_sha256: str
    presets_sha256: str

    @property
    def document_count(self) -> int:
        return len(self.documents)


def _parse_document(raw: object, index: int) -> DocumentRecord:
    if not isinstance(raw, dict):
        raise DataContractError(f"documents[{index}] must be an object")
    doc_id = _required_str(raw.get("id"), f"documents[{index}].id")
    if doc_id != Path(doc_id).name or not doc_id.endswith(".txt"):
        raise DataContractError(f"Unsafe document id: {doc_id}")
    title = _required_str(raw.get("title"), f"{doc_id}.title")
    cluster = _required_str(raw.get("cluster"), f"{doc_id}.cluster")
    variant = _required_str(raw.get("variant"), f"{doc_id}.variant")
    content = _required_str(raw.get("content"), f"{doc_id}.content")
    if len(content) > CONTENT_MAX_LENGTH:
        raise DataContractError(
            f"{doc_id} exceeds the {CONTENT_MAX_LENGTH}-character content limit"
        )
    if len(title) > 256:
        raise DataContractError(f"{doc_id} title exceeds the 256-character limit")
    return DocumentRecord(
        doc_id=doc_id, title=title, cluster=cluster, variant=variant, content=content
    )


def _parse_preset(raw: object, index: int) -> PresetRecord:
    if not isinstance(raw, dict):
        raise DataContractError(f"presets[{index}] must be an object")
    preset_id = _required_str(raw.get("id"), f"presets[{index}].id")
    label = _required_str(raw.get("label"), f"{preset_id}.label")
    title = _required_str(raw.get("title"), f"{preset_id}.title")
    text = _required_str(raw.get("text"), f"{preset_id}.text")
    description = _required_str(raw.get("description"), f"{preset_id}.description")
    if len(text) > CONTENT_MAX_LENGTH:
        raise DataContractError(
            f"preset {preset_id} exceeds the {CONTENT_MAX_LENGTH}-character limit"
        )
    return PresetRecord(id=preset_id, label=label, title=title, text=text, description=description)


def build_dataset(config: RuntimeConfig) -> DatasetBundle:
    """Load and validate the original-short-stories documents and presets."""
    documents_path = config.dataset_root / DOCUMENTS_FILE_NAME
    presets_path = config.dataset_root / PRESETS_FILE_NAME
    if not documents_path.is_file():
        raise DataContractError(f"Missing documents file: {documents_path}")
    if not presets_path.is_file():
        raise DataContractError(f"Missing presets file: {presets_path}")

    try:
        documents_raw = json.loads(documents_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DataContractError(f"Invalid documents JSON: {exc}") from exc
    if not isinstance(documents_raw, dict):
        raise DataContractError("documents.json must be a JSON object")
    dataset_id = _required_str(documents_raw.get("dataset_id"), "dataset_id")
    if dataset_id != EXPECTED_DATASET_ID:
        raise DataContractError(f"Unexpected dataset identity: {dataset_id}")
    raw_documents = documents_raw.get("documents")
    if not isinstance(raw_documents, list) or len(raw_documents) != EXPECTED_DOCUMENT_COUNT:
        raise DataContractError(f"documents.json must contain {EXPECTED_DOCUMENT_COUNT} documents")
    documents = tuple(_parse_document(raw, index) for index, raw in enumerate(raw_documents))
    doc_ids = [document.doc_id for document in documents]
    if len(doc_ids) != len(set(doc_ids)):
        raise DataContractError("Document ids must be unique")

    try:
        presets_raw = json.loads(presets_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DataContractError(f"Invalid presets JSON: {exc}") from exc
    if not isinstance(presets_raw, dict):
        raise DataContractError("presets.json must be a JSON object")
    raw_presets = presets_raw.get("presets")
    if not isinstance(raw_presets, list) or not raw_presets:
        raise DataContractError("presets.json must contain a non-empty preset list")
    presets = tuple(_parse_preset(raw, index) for index, raw in enumerate(raw_presets))
    preset_ids = [preset.id for preset in presets]
    if len(preset_ids) != len(set(preset_ids)):
        raise DataContractError("Preset ids must be unique")

    return DatasetBundle(
        documents=documents,
        presets=presets,
        dataset_id=dataset_id,
        dataset_version=EXPECTED_DATASET_VERSION,
        documents_sha256=_sha256(documents_path),
        presets_sha256=_sha256(presets_path),
    )
