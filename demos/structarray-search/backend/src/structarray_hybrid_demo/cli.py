"""Strict CLI for CoVLA data checks and the fixed hybrid lifecycle."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from structarray_hybrid_demo.config import RuntimeConfig
from structarray_hybrid_demo.data import DataContractError, build_dataset
from structarray_hybrid_demo.service import StructArrayHybridService


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="structarray-hybrid")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("data-check", description="Validate the approved CoVLA 30-video slice")
    subparsers.add_parser("status", description="Inspect the fixed hybrid collection")
    subparsers.add_parser("prepare", description="Embed and prepare the fixed collection")
    subparsers.add_parser("cleanup", description="Drop only the exact demo collection")

    query = subparsers.add_parser("query", description="Run all three semantic search paths")
    query.add_argument("--query", required=True)
    query.add_argument("--limit", type=int, default=8)
    query.add_argument("--parent-weight", type=float, default=0.5)
    return parser


def _print_json(value: object) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = RuntimeConfig.from_environment()
    service = StructArrayHybridService(config=config)

    if args.command == "data-check":
        try:
            bundle = build_dataset(config)
        except DataContractError as exc:
            print(f"data-check failed: {exc}", file=sys.stderr)
            return 1
        _print_json(
            {
                "status": "passed",
                "video_count": bundle.video_count,
                "observation_count": bundle.observation_count,
                "sample_sha256": bundle.sample_sha256,
                "prefix_sha256": bundle.prefix_sha256,
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
                    limit=args.limit,
                    parent_weight=args.parent_weight,
                ).public_dict()
            )
        except Exception as exc:  # noqa: BLE001 - CLI boundary
            print(f"query failed: {exc}", file=sys.stderr)
            return 1
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
