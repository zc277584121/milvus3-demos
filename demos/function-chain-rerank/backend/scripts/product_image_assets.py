#!/usr/bin/env python3
"""Prepare prompts and crop generated product-image contact sheets."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

PRODUCT_TYPE_ORDER = (
    "DESK",
    "RUG",
    "BACKPACK",
    "HEADPHONES",
    "UMBRELLA",
    "BED",
    "SOFA",
    "CHAIR",
    "TABLE",
    "LAMP",
    "HANDBAG",
    "SHOES",
    "BOOT",
    "SANDAL",
    "HAT",
    "SUITCASE",
    "DRINKING_CUP",
    "PILLOW",
    "SHELF",
    "PLANTER",
)

DEFAULT_DATASET_ROOT = Path(__file__).resolve().parents[2] / "data" / (
    "synthetic-commerce-catalog-r1"
)
DEFAULT_STAGING_ROOT = Path(__file__).resolve().parents[4] / "tmp" / "product-imagegen-v1"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def visual_description(product: dict[str, Any]) -> str:
    bullets = "; ".join(str(value) for value in product["bullet_points"])
    return (
        f"{product['description']} Color: {product['color']}. "
        f"Material: {product['material']}. Style: {product['style']}. "
        f"Important visible details: {bullets}."
    )


def build_prompt(product_type: str, products: list[dict[str, Any]], rows: int) -> str:
    columns = 4
    cells = rows * columns
    empty_cells = cells - len(products)
    shape = "square" if rows == 4 else "wide landscape"
    lines = [
        "Use case: product-mockup",
        "Asset type: contact sheet for tiny e-commerce catalog thumbnails",
        (
            f"Primary request: Create ONE {shape} product contact sheet containing an exact "
            f"{columns}-column by {rows}-row grid. The first {len(products)} cells each show "
            f"exactly one different {product_type.lower().replace('_', ' ')} catalog product, "
            "in the precise row-major order listed below. "
            f"The remaining {empty_cells} cells must be completely empty neutral-background cells."
        ),
        "",
        (
            f"Grid layout: {cells} perfectly equal rectangular cells with straight, crisp, "
            "continuous white gutters separating every cell. No product may overlap, cross, "
            "touch, or cast a shadow across a gutter. Treat every cell as an independent catalog "
            "photograph. Do not create one continuous scene spanning multiple cells."
        ),
        "",
        "Scene/backdrop: each cell has the same plain light warm-gray seamless studio background.",
        (
            "Style/medium: clean realistic e-commerce product photography, commercially "
            "plausible but unbranded."
        ),
        (
            "Composition/framing: one complete product or explicitly stated product set centered "
            "in each occupied cell, category-appropriate catalog angle, entire product visible, "
            "generous padding."
        ),
        (
            "Lighting/mood: soft neutral studio lighting, subtle grounding shadow contained "
            "entirely inside its own cell."
        ),
        "",
        "Cell contents, left-to-right and top-to-bottom:",
    ]
    for index, product in enumerate(products):
        row = index // columns + 1
        column = index % columns + 1
        lines.append(
            f"Row {row} column {column}: {visual_description(product)}"
        )
    for index in range(len(products), cells):
        row = index // columns + 1
        column = index % columns + 1
        lines.append(f"Row {row} column {column}: empty light warm-gray studio cell.")
    lines.extend(
        [
            "",
            (
                f"Constraints: exactly {columns} columns and exactly {rows} rows; exactly "
                f"{len(products)} occupied cells; one catalog product or explicitly stated set "
                "per occupied cell; preserve the stated row-major mapping; strong visual "
                "differences matching color, material, style, and construction."
            ),
            (
                "Avoid: labels, numbers, captions, text, logos, brand marks, prices, people, "
                "hands, unrelated props, watermarks, merged panels, irregular collage, shared "
                "rooms, overlapping objects, cropped products, repeated identical products, "
                "text-like marks, and borders inside cells."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def prepare(dataset_root: Path, staging_root: Path) -> None:
    products_doc = read_json(dataset_root / "products.json")
    products = products_doc["products"]
    groups = {
        product_type: [item for item in products if item["product_type"] == product_type]
        for product_type in PRODUCT_TYPE_ORDER
    }
    if sum(len(items) for items in groups.values()) != 240:
        raise RuntimeError("Expected exactly 240 products across the fixed product types")

    prompt_dir = staging_root / "prompts"
    atlas_dir = staging_root / "atlases"
    image_dir = staging_root / "images"
    for directory in (prompt_dir, atlas_dir, image_dir):
        directory.mkdir(parents=True, exist_ok=True)

    manifest_groups = []
    for product_type in PRODUCT_TYPE_ORDER:
        items = groups[product_type]
        rows = 4 if len(items) > 12 else 3
        prompt_name = f"{product_type}.txt"
        atlas_name = f"{product_type}.png"
        (prompt_dir / prompt_name).write_text(
            build_prompt(product_type, items, rows), encoding="utf-8"
        )
        manifest_groups.append(
            {
                "product_type": product_type,
                "rows": rows,
                "columns": 4,
                "prompt_path": f"prompts/{prompt_name}",
                "atlas_path": f"atlases/{atlas_name}",
                "items": [
                    {
                        "id": item["id"],
                        "item_id": item["item_id"],
                        "image_path": item["image_path"],
                        "row": index // 4 + 1,
                        "column": index % 4 + 1,
                    }
                    for index, item in enumerate(items)
                ],
            }
        )
    write_json(
        staging_root / "atlas-manifest.json",
        {
            "dataset_id": products_doc["dataset_id"],
            "revision": products_doc["revision"],
            "groups": manifest_groups,
        },
    )


def identify_dimensions(path: Path) -> tuple[int, int]:
    output = subprocess.check_output(
        ["identify", "-format", "%w %h", str(path)], text=True
    )
    width_text, height_text = output.split()
    return int(width_text), int(height_text)


def crop_atlas(
    atlas_path: Path,
    output_dir: Path,
    items: list[dict[str, Any]],
    rows: int,
    columns: int,
) -> None:
    width, height = identify_dimensions(atlas_path)
    for item in items:
        row = int(item["row"]) - 1
        column = int(item["column"]) - 1
        left = round(column * width / columns)
        right = round((column + 1) * width / columns)
        top = round(row * height / rows)
        bottom = round((row + 1) * height / rows)
        cell_width = right - left
        cell_height = bottom - top
        inset = max(4, round(min(cell_width, cell_height) * 0.025))
        crop_width = cell_width - inset * 2
        crop_height = cell_height - inset * 2
        destination = output_dir / Path(item["image_path"]).name
        destination.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "convert",
                str(atlas_path),
                "-crop",
                f"{crop_width}x{crop_height}+{left + inset}+{top + inset}",
                "+repage",
                "-resize",
                "256x256",
                "-background",
                "#efefed",
                "-gravity",
                "center",
                "-extent",
                "256x256",
                "-strip",
                "-sampling-factor",
                "4:2:0",
                "-quality",
                "86",
                str(destination),
            ],
            check=True,
        )


def crop(staging_root: Path) -> None:
    manifest = read_json(staging_root / "atlas-manifest.json")
    output_dir = staging_root / "images"
    for group in manifest["groups"]:
        atlas_path = staging_root / group["atlas_path"]
        if not atlas_path.is_file():
            raise RuntimeError(f"Missing atlas: {atlas_path}")
        crop_atlas(
            atlas_path,
            output_dir,
            group["items"],
            int(group["rows"]),
            int(group["columns"]),
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate(staging_root: Path) -> None:
    manifest = read_json(staging_root / "atlas-manifest.json")
    expected_names = {
        Path(item["image_path"]).name
        for group in manifest["groups"]
        for item in group["items"]
    }
    image_dir = staging_root / "images"
    actual = {path.name for path in image_dir.glob("*.jpg")}
    if actual != expected_names:
        missing = sorted(expected_names - actual)
        extra = sorted(actual - expected_names)
        raise RuntimeError(f"Image inventory mismatch; missing={missing}, extra={extra}")
    digests = []
    for name in sorted(expected_names):
        path = image_dir / name
        if identify_dimensions(path) != (256, 256):
            raise RuntimeError(f"Unexpected image dimensions: {path}")
        digests.append(sha256_file(path))
    if len(set(digests)) != len(digests):
        raise RuntimeError("Expected every cropped product image to have unique bytes")
    print(f"Validated {len(expected_names)} unique 256x256 JPEG images")


def install_preview_atlas(source: Path, staging_root: Path) -> None:
    destination = staging_root / "atlases" / "DESK.png"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    print(destination)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--staging-root", type=Path, default=DEFAULT_STAGING_ROOT)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare")
    subparsers.add_parser("crop")
    subparsers.add_parser("validate")
    install = subparsers.add_parser("install-preview-atlas")
    install.add_argument("source", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "prepare":
        prepare(args.dataset_root, args.staging_root)
    elif args.command == "crop":
        crop(args.staging_root)
    elif args.command == "validate":
        validate(args.staging_root)
    elif args.command == "install-preview-atlas":
        install_preview_atlas(args.source, args.staging_root)


if __name__ == "__main__":
    main()
