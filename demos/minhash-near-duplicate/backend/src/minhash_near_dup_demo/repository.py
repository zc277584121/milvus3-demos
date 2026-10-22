"""Milvus 3.0 MinHash + LSH provisioning and near-duplicate search."""

from __future__ import annotations

import time
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Protocol

from pymilvus import DataType, Function, FunctionType, MilvusClient

from minhash_near_dup_demo.config import (
    COLLECTION_NAME,
    HASH_FUNCTION,
    INDEX_BUILD_POLL_SECONDS,
    INDEX_NAME,
    INDEX_TYPE,
    METRIC_TYPE,
    MH_ELEMENT_BIT_WIDTH,
    MH_LSH_BAND,
    MILVUS_EXPECTED_VERSION,
    MILVUS_TIMEOUT_SECONDS,
    MILVUS_URI,
    NUM_HASHES,
    REFINE_K,
    SEARCH_WITH_JACCARD,
    SHINGLE_SIZE,
    SIGNATURE_DIMENSION,
    SIGNATURE_FIELD,
    TEXT_FIELD,
    TOKEN_LEVEL,
    WITH_RAW_DATA,
)
from minhash_near_dup_demo.data import DocumentRecord


class RepositoryContractError(RuntimeError):
    """Raised when Milvus or namespace state differs from the fixed contract."""


class SearchContractError(ValueError):
    """Raised when a search cannot be executed honestly against the prepared collection."""


class MilvusClientLike(Protocol):
    def close(self) -> None: ...

    def get_server_version(self) -> str: ...

    def list_collections(self, **kwargs: Any) -> list[Any]: ...

    def has_collection(self, collection_name: str, **kwargs: Any) -> bool: ...

    def create_schema(self, **kwargs: Any) -> Any: ...

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


ClientFactory = Any


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
    raw_collection_names: tuple[str, ...]
    target_exists: bool
    index_names: tuple[str, ...]
    indexes: tuple[dict[str, Any], ...]
    schema: dict[str, Any] | None
    collection_stats: dict[str, Any] | None

    def public_dict(self) -> dict[str, object]:
        return {
            "milvus_uri": MILVUS_URI,
            "server_version": self.server_version,
            "expected_server_version": MILVUS_EXPECTED_VERSION,
            "raw_collection_names": list(self.raw_collection_names),
            "target_collection": COLLECTION_NAME,
            "target_exists": self.target_exists,
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
            "text_field": TEXT_FIELD,
            "signature_field": SIGNATURE_FIELD,
            "metric_type": METRIC_TYPE,
            "server_version": self.server_version,
            "insert_result": self.insert_result,
            "audit": self.audit.public_dict(),
            "latency_ms": self.latency_ms,
        }


@dataclass(frozen=True)
class MilvusHit:
    doc_id: str
    title: str
    content: str
    distance: float

    def similarity(self) -> float:
        """Estimated Jaccard similarity.

        Milvus's ``MHJACCARD`` metric returns the estimated Jaccard similarity
        directly in the ``distance`` field: larger is more similar, and
        identical signatures score 1.0. This is verified against the real
        Milvus 3.0.0 GA endpoint — an exact copy scores 1.0, a one-word edit
        about 0.86, and unrelated text about 0.5. Applying a ``1 - distance``
        transform would invert the scale and is therefore wrong for this
        metric.
        """
        return round(self.distance, 6)

    def public_dict(self, *, threshold: float) -> dict[str, object]:
        similarity = self.similarity()
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "content": self.content,
            "estimated_jaccard": similarity,
            "raw_distance": round(self.distance, 6),
            "above_threshold": similarity >= threshold,
        }


@dataclass(frozen=True)
class CleanupResult:
    dropped: bool
    index_names_before: tuple[str, ...]
    raw_collection_names_before: tuple[str, ...]
    raw_collection_names_after: tuple[str, ...]
    target_absent_after: bool
    unknown_collections_unchanged: bool

    def public_dict(self) -> dict[str, object]:
        value = asdict(self)
        for key in (
            "index_names_before",
            "raw_collection_names_before",
            "raw_collection_names_after",
        ):
            value[key] = list(value[key])
        value["status"] = "clean"
        value["collection_name"] = COLLECTION_NAME
        return value


def _schema_and_index(client: MilvusClientLike) -> tuple[Any, Any]:
    schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field("doc_id", DataType.VARCHAR, is_primary=True, max_length=64)
    schema.add_field("title", DataType.VARCHAR, max_length=256)
    schema.add_field(TEXT_FIELD, DataType.VARCHAR, max_length=2000, enable_analyzer=True)
    schema.add_field(SIGNATURE_FIELD, DataType.BINARY_VECTOR, dim=SIGNATURE_DIMENSION)

    function = Function(
        name="minhash_signature_function",
        function_type=FunctionType.MINHASH,
        input_field_names=[TEXT_FIELD],
        output_field_names=[SIGNATURE_FIELD],
        description="MinHash signatures over the analyzed content field",
        params={
            "num_hashes": NUM_HASHES,
            "shingle_size": SHINGLE_SIZE,
            "hash_function": HASH_FUNCTION,
            "token_level": TOKEN_LEVEL,
        },
    )
    schema.add_function(function)

    indexes = client.prepare_index_params()
    indexes.add_index(
        field_name=SIGNATURE_FIELD,
        index_name=INDEX_NAME,
        index_type=INDEX_TYPE,
        metric_type=METRIC_TYPE,
        params={
            "mh_element_bit_width": MH_ELEMENT_BIT_WIDTH,
            "mh_lsh_band": MH_LSH_BAND,
            "with_raw_data": WITH_RAW_DATA,
        },
    )
    return schema, indexes


def _rows(documents: tuple[DocumentRecord, ...]) -> list[dict[str, str]]:
    return [
        {
            "doc_id": document.doc_id,
            "title": document.title,
            TEXT_FIELD: document.content,
        }
        for document in documents
    ]


class MinHashRepository:
    """Own one exact Collection created by this backend process."""

    def __init__(self, client_factory: Any = None) -> None:
        self._client_factory = client_factory or default_client_factory
        self._cleanup_authorized = False
        self._owns_collection = False

    @property
    def owns_collection(self) -> bool:
        return self._owns_collection

    def _audit_client(self, client: MilvusClientLike) -> RawAudit:
        raw_collections = list(client.list_collections())
        collection_names = normalize_sdk_names(raw_collections, operation="list_collections()")
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
            raw_collection_names=collection_names,
            target_exists=target_exists,
            index_names=index_names,
            indexes=indexes,
            schema=schema,
            collection_stats=stats,
        )

    def raw_audit(self) -> RawAudit:
        client = self._client_factory()
        try:
            return self._audit_client(client)
        finally:
            client.close()

    def _wait_for_index(self, client: MilvusClientLike) -> None:
        deadline = time.monotonic() + MILVUS_TIMEOUT_SECONDS
        while True:
            names = normalize_sdk_names(
                client.list_indexes(COLLECTION_NAME), operation="list_indexes()"
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

    def prepare(self, documents: tuple[DocumentRecord, ...]) -> PrepareResult:
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
                client.insert(collection_name=COLLECTION_NAME, data=_rows(documents))
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
            ) != len(documents):
                raise RepositoryContractError(
                    "Prepared Collection row count differs from document count"
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

    def adopt(self, document_count: int) -> bool:
        """Take ownership of a pre-existing exact Collection after validating it."""
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
            if stats is None or int(stats.get("row_count", -1)) != document_count:
                raise RepositoryContractError(
                    "Pre-existing Collection row count differs from document count"
                )
            self._owns_collection = True
            self._cleanup_authorized = True
            return True
        finally:
            client.close()

    def search(self, query_text: str, *, limit: int) -> tuple[MilvusHit, ...]:
        """Return Milvus-ranked candidates by estimated Jaccard similarity."""
        if not self._owns_collection:
            raise RepositoryContractError("This process does not own a prepared Collection")
        client = self._client_factory()
        try:
            results = client.search(
                collection_name=COLLECTION_NAME,
                data=[query_text],
                anns_field=SIGNATURE_FIELD,
                search_params={
                    "metric_type": METRIC_TYPE,
                    "params": {
                        "mh_search_with_jaccard": SEARCH_WITH_JACCARD,
                        "refine_k": REFINE_K,
                    },
                },
                limit=limit,
                output_fields=["doc_id", "title", TEXT_FIELD],
            )
        finally:
            client.close()
        if len(results) != 1:
            raise RepositoryContractError("Milvus search returned an unexpected query batch shape")
        hits: list[MilvusHit] = []
        for raw_hit in results[0]:
            entity = raw_hit.get("entity", {})
            doc_id = raw_hit.get("doc_id") or entity.get("doc_id") or raw_hit.get("id")
            if not isinstance(doc_id, str):
                raise RepositoryContractError("Milvus hit is missing doc_id")
            hits.append(
                MilvusHit(
                    doc_id=doc_id,
                    title=str(entity["title"]),
                    content=str(entity[TEXT_FIELD]),
                    distance=float(raw_hit["distance"]),
                )
            )
        return tuple(hits)

    def cleanup(self) -> CleanupResult:
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
            if not target_absent:
                raise RepositoryContractError("Exact Collection remains after cleanup")
            if not unknown_unchanged:
                raise RepositoryContractError("A non-demo Collection name changed during cleanup")
            self._owns_collection = False
            self._cleanup_authorized = False
            return CleanupResult(
                dropped=dropped,
                index_names_before=before.index_names,
                raw_collection_names_before=before.raw_collection_names,
                raw_collection_names_after=after.raw_collection_names,
                target_absent_after=target_absent,
                unknown_collections_unchanged=unknown_unchanged,
            )
        finally:
            client.close()
