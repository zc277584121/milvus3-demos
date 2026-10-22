"""Strict CLI for data checks and the fixed MinHash lifecycle."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from minhash_near_dup_demo.config import RuntimeConfig
from minhash_near_dup_demo.data import DataContractError, build_dataset
from minhash_near_dup_demo.service import MinHashService


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="minhash-near-duplicate")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("data-check", description="Validate the original-short-stories dataset")
    subparsers.add_parser("status", description="Inspect the fixed MinHash collection")
    subparsers.add_parser("prepare", description="Prepare the fixed MinHash collection")
    subparsers.add_parser("cleanup", description="Drop only the exact demo collection")

    query = subparsers.add_parser("query", description="Run MinHash near-duplicate search")
    query.add_argument("--query", required=True)
    query.add_argument("--threshold", type=float, default=0.85)
    query.add_argument("--limit", type=int, default=5)
    return parser


def _print_json(value: object) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = RuntimeConfig.from_environment()
    service = MinHashService(config=config)

    if args.command == "data-check":
        try:
            bundle = build_dataset(config)
        except DataContractError as exc:
            print(f"data-check failed: {exc}", file=sys.stderr)
            return 1
        _print_json(
            {
                "status": "passed",
                "dataset_id": bundle.dataset_id,
                "dataset_version": bundle.dataset_version,
                "document_count": bundle.document_count,
                "preset_count": len(bundle.presets),
                "documents_sha256": bundle.documents_sha256,
                "presets_sha256": bundle.presets_sha256,
            }
        )
        return 0

    if args.command == "status":
        _print_json(service.status())
        return 0

    if args.command == "prepare":
        try:
            _print_json(service.prepare())
        except Exception as exc:  # noqa: BLE001 - CLI boundary
            print(f"prepare failed: {exc}", file=sys.stderr)
            return 1
        return 0

    if args.command == "cleanup":
        try:
            _print_json(service.cleanup())
        except Exception as exc:  # noqa: BLE001 - CLI boundary
            print(f"cleanup failed: {exc}", file=sys.stderr)
            return 1
        return 0

    if args.command == "query":
        try:
            _print_json(
                service.search(
                    query=args.query,
                    threshold=args.threshold,
                    limit=args.limit,
                ).public_dict()
            )
        except Exception as exc:  # noqa: BLE001 - CLI boundary
            print(f"query failed: {exc}", file=sys.stderr)
            return 1
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
