"""Lifecycle and verification commands for the Function Chain backend."""

from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

from function_chain_demo.catalog import QUERIES
from function_chain_demo.config import DemoSettings
from function_chain_demo.modeling import train_model
from function_chain_demo.service import DemoProvisioner, SearchService
from function_chain_demo.verification import local_model_order


def verify(settings: DemoSettings, report_path: Path) -> dict[str, Any]:
    provisioner = DemoProvisioner(settings)
    search_service = SearchService(settings)
    report: dict[str, Any] = {}
    cleanup_error: Exception | None = None
    try:
        manifest = provisioner.prepare()
        comparison = search_service.compare(QUERIES[0].query_text, query_id=QUERIES[0].id)
        with tempfile.TemporaryDirectory(prefix="function-chain-verify-") as temp_dir:
            artifact = train_model(Path(temp_dir) / "xgb-reranker.ubj")
            expected_order = local_model_order(artifact.path, comparison)

        vector_ids = [product.id for product in comparison.vector_order]
        business_ids = [product.id for product in comparison.business_order]
        if vector_ids == business_ids:
            raise AssertionError("The server-side Function Chain did not change the order")
        if business_ids != expected_order:
            raise AssertionError("The server-side order differs from the XGBoost model order")
        report = {
            "status": "passed",
            "manifest": asdict(manifest),
            "search": comparison.public_dict(),
            "vector_order_ids": vector_ids,
            "business_order_ids": business_ids,
            "local_model_order": expected_order,
        }
    finally:
        try:
            provisioner.cleanup()
        except Exception as exc:
            cleanup_error = exc
        cleanup_state = provisioner.inspect_state()
        report["cleanup_state"] = asdict(cleanup_state)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    if cleanup_error is not None:
        raise cleanup_error
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare", help="Create the namespaced model and collection")
    subparsers.add_parser("cleanup", help="Remove only this demo's namespaced resources")
    verify_parser = subparsers.add_parser(
        "verify",
        help="Prepare, run the real server-side comparison, and clean up",
    )
    verify_parser.add_argument(
        "--report",
        type=Path,
        required=True,
        help="JSON report path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = DemoSettings()
    provisioner = DemoProvisioner(settings)
    if args.command == "prepare":
        print(json.dumps(asdict(provisioner.prepare()), indent=2))
    elif args.command == "cleanup":
        provisioner.cleanup()
        print(json.dumps(asdict(provisioner.inspect_state()), indent=2))
    else:
        print(json.dumps(verify(settings, args.report), indent=2))


if __name__ == "__main__":
    main()
