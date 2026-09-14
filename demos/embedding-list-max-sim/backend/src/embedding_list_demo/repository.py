"""Exact project-GA Milvus repository for page multi-vector retrieval."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Protocol

import numpy as np
import torch
from pymilvus import DataType, MilvusClient
from pymilvus.client.embedding_list import EmbeddingList

from embedding_list_demo.config import (
    ANNS_FIELD,
    COLLECTION_NAME,
    INDEX_NAME,
    MAX_PATCHES_PER_PAGE,
    METRIC_TYPE,
    MILVUS_EXPECTED_VERSION,
    MILVUS_TIMEOUT_SECONDS,
    MILVUS_URI,
    VECTOR_DIMENSION,
)
from embedding_list_demo.manual import ManualManifest

INDEX_BUILD_POLL_SECONDS = 0.1


class RepositoryContractError(RuntimeError):
    """Raised when Milvus or namespace state differs from the fixed contract."""


class MilvusClientLike(Protocol):
    def close(self) -> None: ...

    def get_server_version(self) -> str: ...

    def list_collections(self, **kwargs: Any) -> list[Any]: ...

    def list_file_resources(self, **kwargs: Any) -> list[Any]: ...

    def has_collection(self, collection_name: str, **kwargs: Any) -> bool: ...

    def create_schema(self, **kwargs: Any) -> Any: ...

    def create_struct_field_schema(self, **kwargs: Any) -> Any: ...

    def prepare_index_params(self, **kwargs: Any) -> Any: ...

    def create_collection(self, **kwargs: Any) -> Any: ...

    def insert(self, **kwargs: Any) -> Any: ...

    def flush(self, **kwargs: Any) -> Any: ...

    def load_collection(self, **kwargs: Any) -> Any: ...

    def describe_collection(self, collection_name: str, **kwargs: Any) -> Any: ...

    def list_indexes(self, collection_name: str, **kwargs: Any) -> list[Any]: ...

    def describe_index(self, collection_name: str, index_name: str, **kwargs: Any) -> Any: ...

    def get_collection_stats(self, collection_name: str, **kwargs: Any) -> Any: ...

    def search(self, **kwargs: Any) -> list[list[dict[str, Any]]]: ...

    def drop_collection(self, collection_name: str, **kwargs: Any) -> Any: ...


ClientFactory = Callable[[], MilvusClientLike]


def default_client_factory() -> MilvusClient:
    """Connect only to the project Milvus 3.0.0 GA endpoint."""

    return MilvusClient(uri=MILVUS_URI, timeout=MILVUS_TIMEOUT_SECONDS)


def normalize_sdk_names(items: Iterable[object], *, operation: str) -> tuple[str, ...]:
    """Normalize documented string and object-with-name SDK return shapes."""

    names: list[str] = []
    for item in items:
        name = item if isinstance(item, str) else getattr(item, "name", None)
        if not isinstance(name, str) or not name:
            raise TypeError(f"{operation} returned an item without a nonempty string name")
        names.append(name)
    if len(names) != len(set(names)):
        raise RepositoryContractError(f"{operation} returned duplicate names")
    return tuple(sorted(names))


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.name
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "to_dict"):
        return _jsonable(value.to_dict())
    return str(value)


@dataclass(frozen=True)
class RawAudit:
    server_version: str
    collection_item_sdk_types: tuple[str, ...]
    raw_collection_names: tuple[str, ...]
    file_resource_item_sdk_types: tuple[str, ...]
    raw_file_resource_names: tuple[str, ...]
    target_exists: bool
    index_item_sdk_types: tuple[str, ...]
    index_names: tuple[str, ...]
    indexes: tuple[dict[str, Any], ...]
    schema: dict[str, Any] | None
    collection_stats: dict[str, Any] | None

    def public_dict(self) -> dict[str, object]:
        return {
            "milvus_uri": MILVUS_URI,
            "server_version": self.server_version,
            "expected_server_version": MILVUS_EXPECTED_VERSION,
            "collection_item_sdk_types": list(self.collection_item_sdk_types),
            "raw_collection_names": list(self.raw_collection_names),
            "file_resource_item_sdk_types": list(self.file_resource_item_sdk_types),
            "raw_file_resource_names": list(self.raw_file_resource_names),
            "target_collection": COLLECTION_NAME,
            "target_exists": self.target_exists,
            "index_item_sdk_types": list(self.index_item_sdk_types),
            "index_names": list(self.index_names),
            "indexes": list(self.indexes),
            "schema": self.schema,
            "collection_stats": self.collection_stats,
        }


@dataclass(frozen=True)
class PrepareResult:
    server_version: str
    insert_result: dict[str, Any]
    audit: RawAudit
    latency_ms: dict[str, float]

    def public_dict(self) -> dict[str, object]:
        return {
            "status": "prepared",
            "collection_name": COLLECTION_NAME,
            "index_name": INDEX_NAME,
            "anns_field": ANNS_FIELD,
            "metric_type": METRIC_TYPE,
            "server_version": self.server_version,
            "insert_result": self.insert_result,
            "audit": self.audit.public_dict(),
            "latency_ms": self.latency_ms,
        }


@dataclass(frozen=True)
class MilvusHit:
    page_id: str
    pdf_page_index: int
    printed_page: int
    title: str
    section: str
    document_identifier: str
    ntrs_id: int
    ntrs_record_url: str
    distribution: str
    rights_determination: str
    contains_third_party_material: bool
    pdf_sha256: str
    image_sha256: str
    score: float

    def public_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class CleanupResult:
    dropped: bool
    index_names_before: tuple[str, ...]
    raw_collection_names_before: tuple[str, ...]
    raw_collection_names_after: tuple[str, ...]
    raw_file_resource_names_before: tuple[str, ...]
    raw_file_resource_names_after: tuple[str, ...]
    target_absent_after: bool
    unknown_collections_unchanged: bool
    file_resources_unchanged: bool

    def public_dict(self) -> dict[str, object]:
        value = asdict(self)
        for key in (
            "index_names_before",
            "raw_collection_names_before",
            "raw_collection_names_after",
            "raw_file_resource_names_before",
            "raw_file_resource_names_after",
        ):
            value[key] = list(value[key])
        value["status"] = "clean"
        value["collection_name"] = COLLECTION_NAME
        return value


def _schema_and_index(client: MilvusClientLike) -> tuple[Any, Any]:
    schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field("page_id", DataType.VARCHAR, is_primary=True, max_length=64)
    schema.add_field("pdf_page_index", DataType.INT32)
    schema.add_field("printed_page", DataType.INT32)
    schema.add_field("title", DataType.VARCHAR, max_length=256)
    schema.add_field("section", DataType.VARCHAR, max_length=256)
    schema.add_field("document_identifier", DataType.VARCHAR, max_length=64)
    schema.add_field("ntrs_id", DataType.INT64)
    schema.add_field("ntrs_record_url", DataType.VARCHAR, max_length=256)
    schema.add_field("distribution", DataType.VARCHAR, max_length=32)
    schema.add_field("rights_determination", DataType.VARCHAR, max_length=64)
    schema.add_field("contains_third_party_material", DataType.BOOL)
    schema.add_field("pdf_sha256", DataType.VARCHAR, max_length=64)
    schema.add_field("image_sha256", DataType.VARCHAR, max_length=64)
    patch = client.create_struct_field_schema()
    patch.add_field("patch_index", DataType.INT32)
    patch.add_field("patch_embedding", DataType.FLOAT_VECTOR, dim=VECTOR_DIMENSION)
    schema.add_field(
        "patches",
        DataType.ARRAY,
        element_type=DataType.STRUCT,
        struct_schema=patch,
        max_capacity=MAX_PATCHES_PER_PAGE,
    )
    indexes = client.prepare_index_params()
    indexes.add_index(
        field_name=ANNS_FIELD,
        index_name=INDEX_NAME,
        index_type="HNSW",
        metric_type=METRIC_TYPE,
        params={"M": 8, "efConstruction": 64},
    )
    return schema, indexes


def _rows(manual: ManualManifest, vectors: list[torch.Tensor]) -> list[dict[str, object]]:
    if len(manual.pages) != len(vectors):
        raise RepositoryContractError("Page embedding count differs from the manual")
    rows: list[dict[str, object]] = []
    for page, tensor in zip(manual.pages, vectors, strict=True):
        array = tensor.detach().cpu().numpy().astype(np.float32, copy=False)
        if array.ndim != 2 or array.shape[1] != VECTOR_DIMENSION:
            raise RepositoryContractError(f"Unexpected page embedding shape: {page.page_id}")
        if array.shape[0] > MAX_PATCHES_PER_PAGE:
            raise RepositoryContractError(f"Page exceeds Milvus patch capacity: {page.page_id}")
        rows.append(
            {
                "page_id": page.page_id,
                "pdf_page_index": page.pdf_page_index,
                "printed_page": page.printed_page,
                "title": page.title,
                "section": page.section,
                "document_identifier": manual.document_identifier,
                "ntrs_id": manual.ntrs_id,
                "ntrs_record_url": manual.ntrs_record_url,
                "distribution": manual.distribution,
                "rights_determination": manual.rights_determination,
                "contains_third_party_material": manual.contains_third_party_material,
                "pdf_sha256": manual.pdf_sha256,
                "image_sha256": page.image_sha256,
                "patches": [
                    {"patch_index": index, "patch_embedding": vector.tolist()}
                    for index, vector in enumerate(array)
                ],
            }
        )
    return rows


class EmbeddingListRepository:
    """Own one exact Collection created by this backend process."""

    def __init__(self, client_factory: ClientFactory | None = None) -> None:
        self._client_factory = client_factory or default_client_factory
        self._cleanup_authorized = False
        self._owns_collection = False

    @property
    def owns_collection(self) -> bool:
        return self._owns_collection

    def _audit_client(self, client: MilvusClientLike) -> RawAudit:
        raw_collections = list(client.list_collections())
        collection_names = normalize_sdk_names(
            raw_collections,
            operation="list_collections()",
        )
        raw_resources = list(client.list_file_resources())
        resource_names = normalize_sdk_names(
            raw_resources,
            operation="list_file_resources()",
        )
        target_exists = COLLECTION_NAME in collection_names
        raw_indexes = list(client.list_indexes(COLLECTION_NAME)) if target_exists else []
        index_names = normalize_sdk_names(raw_indexes, operation="list_indexes()")
        indexes = tuple(
            _jsonable(client.describe_index(COLLECTION_NAME, name)) for name in index_names
        )
        schema = _jsonable(client.describe_collection(COLLECTION_NAME)) if target_exists else None
        stats = _jsonable(client.get_collection_stats(COLLECTION_NAME)) if target_exists else None
        if schema is not None and not isinstance(schema, dict):
            raise RepositoryContractError("describe_collection() returned an unexpected shape")
        if stats is not None and not isinstance(stats, dict):
            raise RepositoryContractError("get_collection_stats() returned an unexpected shape")
        return RawAudit(
            server_version=client.get_server_version(),
            collection_item_sdk_types=tuple(
                sorted({type(item).__name__ for item in raw_collections})
            ),
            raw_collection_names=collection_names,
            file_resource_item_sdk_types=tuple(
                sorted({type(item).__name__ for item in raw_resources})
            ),
            raw_file_resource_names=resource_names,
            target_exists=target_exists,
            index_item_sdk_types=tuple(sorted({type(item).__name__ for item in raw_indexes})),
            index_names=index_names,
            indexes=indexes,
            schema=schema,
            collection_stats=stats,
        )

    def raw_audit(self) -> RawAudit:
        """Inspect only project GA and the exact target namespace."""

        client = self._client_factory()
        try:
            return self._audit_client(client)
        finally:
            client.close()

    def _wait_for_index(self, client: MilvusClientLike) -> None:
        deadline = time.monotonic() + MILVUS_TIMEOUT_SECONDS
        while True:
            names = normalize_sdk_names(
                client.list_indexes(COLLECTION_NAME),
                operation="list_indexes()",
            )
            if names != (INDEX_NAME,):
                raise RepositoryContractError(
                    f"Milvus index names differ: expected={[INDEX_NAME]}, actual={list(names)}"
                )
            index = _jsonable(client.describe_index(COLLECTION_NAME, INDEX_NAME))
            if not isinstance(index, dict):
                raise RepositoryContractError("describe_index() returned an unexpected shape")
            if index.get("state") == "Finished":
                return
            if time.monotonic() >= deadline:
                raise RepositoryContractError(
                    f"Milvus index did not finish: state={index.get('state')}"
                )
            time.sleep(INDEX_BUILD_POLL_SECONDS)

    def prepare(self, manual: ManualManifest, vectors: list[torch.Tensor]) -> PrepareResult:
        """Create and populate the exact Collection without replacing prior state."""

        started = time.perf_counter()
        client = self._client_factory()
        created = False
        try:
            preflight_started = time.perf_counter()
            before = self._audit_client(client)
            if before.server_version != MILVUS_EXPECTED_VERSION:
                raise RepositoryContractError(
                    f"Expected Milvus {MILVUS_EXPECTED_VERSION}, got {before.server_version}"
                )
            if before.target_exists:
                raise RepositoryContractError(
                    f"Refusing to replace existing Collection {COLLECTION_NAME}"
                )
            self._cleanup_authorized = True
            preflight_ms = (time.perf_counter() - preflight_started) * 1000

            create_started = time.perf_counter()
            schema, indexes = _schema_and_index(client)
            client.create_collection(
                collection_name=COLLECTION_NAME,
                schema=schema,
                index_params=indexes,
                consistency_level="Strong",
            )
            created = True
            self._owns_collection = True
            create_ms = (time.perf_counter() - create_started) * 1000

            insert_started = time.perf_counter()
            insert_result = _jsonable(
                client.insert(collection_name=COLLECTION_NAME, data=_rows(manual, vectors))
            )
            client.flush(collection_name=COLLECTION_NAME)
            client.load_collection(collection_name=COLLECTION_NAME)
            self._wait_for_index(client)
            insert_ms = (time.perf_counter() - insert_started) * 1000

            after = self._audit_client(client)
            if not after.target_exists or after.index_names != (INDEX_NAME,):
                raise RepositoryContractError("Prepared Collection or exact index is absent")
            if after.collection_stats is None or int(
                after.collection_stats.get("row_count", -1)
            ) != len(manual.pages):
                raise RepositoryContractError(
                    "Prepared Collection row count differs from page count"
                )
            if not isinstance(insert_result, dict):
                raise RepositoryContractError("Milvus insert result is not a mapping")
            return PrepareResult(
                server_version=after.server_version,
                insert_result=insert_result,
                audit=after,
                latency_ms={
                    "server_preflight": round(preflight_ms, 3),
                    "create_collection_and_index": round(create_ms, 3),
                    "insert_flush_load_index": round(insert_ms, 3),
                    "total": round((time.perf_counter() - started) * 1000, 3),
                },
            )
        except Exception:
            if created and client.has_collection(COLLECTION_NAME):
                client.drop_collection(COLLECTION_NAME)
            self._owns_collection = False
            raise
        finally:
            client.close()

    def adopt(self, page_count: int) -> bool:
        """Take ownership of a pre-existing exact Collection after validating it.

        This is used on startup so a container restart can reuse the Collection
        it (or a prior identical run) already built instead of refusing to
        search. Returns True when the Collection was adopted, False when it is
        absent (caller should then prepare from scratch).
        """

        client = self._client_factory()
        try:
            audit = self._audit_client(client)
            if audit.server_version != MILVUS_EXPECTED_VERSION:
                raise RepositoryContractError(
                    f"Expected Milvus {MILVUS_EXPECTED_VERSION}, got {audit.server_version}"
                )
            if not audit.target_exists:
                return False
            if audit.index_names != (INDEX_NAME,):
                raise RepositoryContractError(
                    f"Pre-existing Collection index differs: expected={[INDEX_NAME]}, "
                    f"actual={list(audit.index_names)}"
                )
            stats = audit.collection_stats
            if stats is None or int(stats.get("row_count", -1)) != page_count:
                raise RepositoryContractError(
                    "Pre-existing Collection row count differs from page count"
                )
            self._owns_collection = True
            self._cleanup_authorized = True
            return True
        finally:
            client.close()

    def search(self, query_vectors: torch.Tensor, *, limit: int) -> tuple[MilvusHit, ...]:
        """Return Milvus-ranked pages without application-side sorting or reranking."""

        if not self._owns_collection:
            raise RepositoryContractError("This process does not own a prepared Collection")
        array = query_vectors.detach().cpu().numpy().astype(np.float32, copy=False)
        query = EmbeddingList(dim=VECTOR_DIMENSION, dtype=np.dtype("float32"))
        for vector in array:
            query.add(vector)
        client = self._client_factory()
        try:
            results = client.search(
                collection_name=COLLECTION_NAME,
                data=[query],
                anns_field=ANNS_FIELD,
                search_params={"metric_type": METRIC_TYPE, "params": {"ef": 64}},
                limit=limit,
                output_fields=[
                    "page_id",
                    "pdf_page_index",
                    "printed_page",
                    "title",
                    "section",
                    "document_identifier",
                    "ntrs_id",
                    "ntrs_record_url",
                    "distribution",
                    "rights_determination",
                    "contains_third_party_material",
                    "pdf_sha256",
                    "image_sha256",
                ],
            )
        finally:
            client.close()
        if len(results) != 1:
            raise RepositoryContractError("Milvus search returned an unexpected query batch shape")
        hits: list[MilvusHit] = []
        for raw_hit in results[0]:
            entity = raw_hit.get("entity", {})
            page_id = raw_hit.get("page_id") or entity.get("page_id") or raw_hit.get("id")
            if not isinstance(page_id, str):
                raise RepositoryContractError("Milvus hit is missing page_id")
            hits.append(
                MilvusHit(
                    page_id=page_id,
                    pdf_page_index=int(entity["pdf_page_index"]),
                    printed_page=int(entity["printed_page"]),
                    title=str(entity["title"]),
                    section=str(entity["section"]),
                    document_identifier=str(entity["document_identifier"]),
                    ntrs_id=int(entity["ntrs_id"]),
                    ntrs_record_url=str(entity["ntrs_record_url"]),
                    distribution=str(entity["distribution"]),
                    rights_determination=str(entity["rights_determination"]),
                    contains_third_party_material=bool(entity["contains_third_party_material"]),
                    pdf_sha256=str(entity["pdf_sha256"]),
                    image_sha256=str(entity["image_sha256"]),
                    score=float(raw_hit["distance"]),
                )
            )
        return tuple(hits)

    def cleanup(self) -> CleanupResult:
        """Drop only the exact Collection when this process owns cleanup authority."""

        client = self._client_factory()
        try:
            before = self._audit_client(client)
            if before.target_exists and not self._cleanup_authorized:
                raise RepositoryContractError(
                    f"Refusing to drop unowned Collection {COLLECTION_NAME}"
                )
            unknown_before = set(before.raw_collection_names) - {COLLECTION_NAME}
            dropped = before.target_exists
            if dropped:
                client.drop_collection(COLLECTION_NAME)
            after = self._audit_client(client)
            target_absent = not after.target_exists
            unknown_unchanged = set(after.raw_collection_names) == unknown_before
            resources_unchanged = before.raw_file_resource_names == after.raw_file_resource_names
            if not target_absent:
                raise RepositoryContractError("Exact Collection remains after cleanup")
            if not unknown_unchanged:
                raise RepositoryContractError("A non-demo Collection name changed during cleanup")
            if not resources_unchanged:
                raise RepositoryContractError("FileResource names changed during cleanup")
            self._owns_collection = False
            self._cleanup_authorized = False
            return CleanupResult(
                dropped=dropped,
                index_names_before=before.index_names,
                raw_collection_names_before=before.raw_collection_names,
                raw_collection_names_after=after.raw_collection_names,
                raw_file_resource_names_before=before.raw_file_resource_names,
                raw_file_resource_names_after=after.raw_file_resource_names,
                target_absent_after=target_absent,
                unknown_collections_unchanged=unknown_unchanged,
                file_resources_unchanged=resources_unchanged,
            )
        finally:
            client.close()
