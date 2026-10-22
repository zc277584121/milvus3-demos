"""Strict CLI for movie data checks and the fixed lifecycle."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from movie_discovery_demo.config import RuntimeConfig
from movie_discovery_demo.data import DataContractError, build_dataset
from movie_discovery_demo.service import MovieDiscoveryService


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="movie-discovery")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("data-check", description="Validate the 90-movie catalog")
    subparsers.add_parser("status", description="Inspect the fixed movie collection")
    subparsers.add_parser("prepare", description="Embed and prepare the fixed collection")
    subparsers.add_parser("cleanup", description="Drop only the exact demo collection")

    query = subparsers.add_parser("query", description="Run grouped and list search paths")
    query.add_argument("--query", required=True)
    query.add_argument("--view", choices=["grouped", "list"], default="grouped")
    query.add_argument("--facet", choices=["genre", "decade", "region"], default="genre")
    query.add_argument("--limit", type=int, default=12)
    query.add_argument("--max-runtime", type=int, default=None)
    return parser


def _print_json(value: object) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = RuntimeConfig.from_environment()
    service = MovieDiscoveryService(config=config)

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
                "movie_count": bundle.movie_count,
                "movies_sha256": bundle.movies_sha256,
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
                    view=args.view,
                    facet=args.facet,
                    limit=args.limit,
                    max_runtime=args.max_runtime,
                ).public_dict()
            )
        except Exception as exc:  # noqa: BLE001 - CLI boundary
            print(f"query failed: {exc}", file=sys.stderr)
            return 1
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
