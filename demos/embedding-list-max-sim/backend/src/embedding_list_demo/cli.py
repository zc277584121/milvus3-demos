"""Raw lifecycle audit and evidence-authorized cleanup commands."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from pymilvus import MilvusClient

from embedding_list_demo.config import (
    COLLECTION_NAME,
    INDEX_NAME,
    MILVUS_EXPECTED_VERSION,
    MILVUS_TIMEOUT_SECONDS,
    MILVUS_URI,
    RuntimeConfig,
)
from embedding_list_demo.repository import EmbeddingListRepository, normalize_sdk_names


class LifecycleContractError(RuntimeError):
    """Raised when lifecycle evidence cannot authorize exact cleanup."""


def _load_preflight(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise LifecycleContractError("Cleanup preflight audit must be a regular JSON file")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise LifecycleContractError("Cleanup preflight audit must contain a JSON object")
    required = {
        "server_version": MILVUS_EXPECTED_VERSION,
        "expected_server_version": MILVUS_EXPECTED_VERSION,
        "target_collection": COLLECTION_NAME,
        "target_exists": False,
    }
    for key, expected in required.items():
        if value.get(key) != expected:
            raise LifecycleContractError(
                f"Cleanup preflight differs for {key}: expected={expected!r}, "
                f"actual={value.get(key)!r}"
            )
    if not isinstance(value.get("raw_collection_names"), list) or not isinstance(
        value.get("raw_file_resource_names"), list
    ):
        raise LifecycleContractError("Cleanup preflight is missing raw SDK name lists")
    return value


def cleanup_from_preflight(path: Path) -> dict[str, object]:
    """Drop only the target proven absent before this lifecycle and purge owned runtime."""

    preflight = _load_preflight(path)
    before_collections = tuple(sorted(str(item) for item in preflight["raw_collection_names"]))
    before_resources = tuple(sorted(str(item) for item in preflight["raw_file_resource_names"]))
    if COLLECTION_NAME in before_collections:
        raise LifecycleContractError("Preflight unexpectedly contains the target Collection")

    client = MilvusClient(uri=MILVUS_URI, timeout=MILVUS_TIMEOUT_SECONDS)
    try:
        if client.get_server_version() != MILVUS_EXPECTED_VERSION:
            raise LifecycleContractError("Project GA version changed before cleanup")
        raw_current_collections = list(client.list_collections())
        current_collections = normalize_sdk_names(
            raw_current_collections,
            operation="list_collections()",
        )
        raw_current_resources = list(client.list_file_resources())
        current_resources = normalize_sdk_names(
            raw_current_resources,
            operation="list_file_resources()",
        )
        current_unknown = tuple(name for name in current_collections if name != COLLECTION_NAME)
        if current_unknown != before_collections:
            raise LifecycleContractError("Non-demo Collection names changed before cleanup")
        if current_resources != before_resources:
            raise LifecycleContractError("FileResource names changed before cleanup")

        dropped = COLLECTION_NAME in current_collections
        index_names: tuple[str, ...] = ()
        if dropped:
            raw_indexes = list(client.list_indexes(COLLECTION_NAME))
            index_names = normalize_sdk_names(raw_indexes, operation="list_indexes()")
            if index_names != (INDEX_NAME,):
                raise LifecycleContractError(
                    f"Refusing cleanup with unexpected target indexes: {list(index_names)}"
                )
            client.drop_collection(COLLECTION_NAME)

        final_collections = normalize_sdk_names(
            list(client.list_collections()),
            operation="list_collections()",
        )
        final_resources = normalize_sdk_names(
            list(client.list_file_resources()),
            operation="list_file_resources()",
        )
        if final_collections != before_collections or final_resources != before_resources:
            raise LifecycleContractError("Raw SDK names did not return to the preflight state")
    finally:
        client.close()

    runtime_root = RuntimeConfig.from_environment().runtime_root
    runtime_removed = False
    if runtime_root.exists():
        if runtime_root.is_symlink() or not runtime_root.is_dir():
            raise LifecycleContractError("Refusing to remove an unsafe runtime root")
        shutil.rmtree(runtime_root)
        runtime_removed = True
    return {
        "status": "clean",
        "collection_name": COLLECTION_NAME,
        "dropped": dropped,
        "index_names_before": list(index_names),
        "raw_collection_names_after": list(final_collections),
        "raw_file_resource_names_after": list(final_resources),
        "runtime_root": str(runtime_root),
        "runtime_removed": runtime_removed,
        "runtime_exists_after": runtime_root.exists(),
        "preflight_audit": str(path.resolve()),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="embedding-list-lifecycle")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("raw-audit", description="Audit project GA raw SDK names")
    cleanup = subparsers.add_parser(
        "cleanup",
        description="Clean only resources authorized by an absent-before raw audit",
    )
    cleanup.add_argument("--preflight-audit", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "raw-audit":
        result = EmbeddingListRepository().raw_audit().public_dict()
    else:
        result = cleanup_from_preflight(args.preflight_audit)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
