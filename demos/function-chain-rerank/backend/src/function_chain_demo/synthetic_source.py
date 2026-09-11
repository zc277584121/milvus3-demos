"""Deterministic preparation of the fully synthetic Function Chain source set.

There is no external retailer, no download, and no real product metadata or
photography. Every product identifier, brand, title, description, bullet point,
and color is authored by this project. Product images are project-generated
synthetic catalog renderings that are dimension-pinned and hash-pinned.

The emitted source manifest keeps the exact shape ``dataset.generate_dataset``
expects (``revision``, ``selection_rule``, ``static_objects``, ``items``), so the
rest of the pipeline is unchanged except for the honest provenance labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import struct
from pathlib import Path
from typing import Final

from function_chain_demo.catalog_curation import (
    DERIVED_ENGLISH_TITLE_PROVENANCE,
    TYPE_ITEM_COUNTS,
    curation_contract,
)
from function_chain_demo.synthetic_catalog import build_items, validate

DATASET_REVISION: Final = "synthetic-commerce-catalog-r1"
SELECTION_VERSION: Final = "synthetic-commerce-240-authored-intent-v1"

# Provenance labels shared with ``dataset`` (which imports them from here).
SYNTHETIC_METADATA_FIELD: Final = "synthetic_authored_metadata"
SYNTHETIC_IDENTITY_FIELD: Final = "synthetic_catalog_identity"
SYNTHETIC_IMAGE_FIELD: Final = "synthetic_generated_product_image_object"

# Product types in catalog order: the six intent slices first, then the fourteen
# background slices. This is the exact order ``synthetic_catalog.build_items``
# emits, so sequence numbers line up with the frozen type order.
TARGET_TYPES: Final = (
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

PLACEHOLDER_WIDTH: Final = 640
PLACEHOLDER_HEIGHT: Final = 480
PRODUCT_IMAGE_WIDTH: Final = 256
PRODUCT_IMAGE_HEIGHT: Final = 256
DEFAULT_DATASET_ROOT: Final = Path(__file__).resolve().parents[3] / "data" / DATASET_REVISION

# One muted representative color per product type so the placeholder subtly
# hints at the category without pretending to be a product photograph.
TYPE_COLORS: Final[dict[str, tuple[int, int, int]]] = {
    "DESK": (112, 78, 48),
    "RUG": (206, 198, 184),
    "BACKPACK": (190, 60, 50),
    "HEADPHONES": (60, 60, 72),
    "UMBRELLA": (30, 32, 38),
    "BED": (150, 96, 52),
    "SOFA": (90, 110, 150),
    "CHAIR": (140, 120, 90),
    "TABLE": (120, 90, 60),
    "LAMP": (200, 170, 90),
    "HANDBAG": (130, 80, 60),
    "SHOES": (80, 80, 90),
    "BOOT": (110, 90, 80),
    "SANDAL": (170, 150, 120),
    "HAT": (90, 130, 90),
    "SUITCASE": (60, 80, 110),
    "DRINKING_CUP": (140, 160, 170),
    "PILLOW": (190, 180, 200),
    "SHELF": (100, 80, 60),
    "PLANTER": (160, 100, 80),
}

# ---------------------------------------------------------------------------
# Original static texts: the project's own synthetic-catalog README and the
# CC0-1.0 dedication covering the authored catalog content.
# ---------------------------------------------------------------------------

_SYNTHETIC_README: Final = b"""# Synthetic Commerce Catalog

This catalog is entirely fictional and was authored by the milvus3-demos project
for the Function Chain reranking demonstration. It is not derived from any real
retailer: product identifiers, brand names, titles, descriptions, bullet points,
and colors are invented. Every product image is a project-generated synthetic
catalog rendering rather than a photograph of a real product.

Operational values (display price, rating, inventory, return rate, release age,
clicks, sales) are deterministic simulated signals and are labeled as such on
every field. They are not merchant or user facts.
"""

_LICENSE_CC0: Final = b"""# CC0 1.0 Universal

To the extent possible under law, the milvus3-demos project has waived all
copyright and related or neighboring rights to the synthetic catalog content
(metadata, synthetic generated product images, and this notice).

The full legal text is published by Creative Commons at:
https://creativecommons.org/publicdomain/zero/1.0/legalcode

A human-readable summary is available at:
https://creativecommons.org/publicdomain/zero/1.0/
"""


class SourcePreparationError(ValueError):
    """Raised when synthetic source generation violates the contract."""


def canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    ).encode()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def md5_file(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _md5_bytes(data: bytes) -> str:
    return hashlib.md5(data, usedforsecurity=False).hexdigest()


def static_objects() -> tuple[dict[str, object], ...]:
    """Fixed original static texts with precomputed digests (pure constants)."""
    return (
        {
            "kind": "dataset_readme",
            "object_path": "licenses/SYNTHETIC-README.md",
            "bytes": len(_SYNTHETIC_README),
            "sha256": _sha256_bytes(_SYNTHETIC_README),
            "md5": _md5_bytes(_SYNTHETIC_README),
            "media_type": "text/markdown",
            "url": "synthetic://catalog/SYNTHETIC-README.md",
        },
        {
            "kind": "controlling_license",
            "object_path": "licenses/LICENSE.txt",
            "bytes": len(_LICENSE_CC0),
            "sha256": _sha256_bytes(_LICENSE_CC0),
            "md5": _md5_bytes(_LICENSE_CC0),
            "media_type": "text/plain",
            "url": "synthetic://catalog/LICENSE.txt",
        },
    )


STATIC_OBJECTS: Final = static_objects()


def _static_content(kind: str) -> bytes:
    if kind == "dataset_readme":
        return _SYNTHETIC_README
    if kind == "controlling_license":
        return _LICENSE_CC0
    raise SourcePreparationError(f"Unknown static object kind: {kind}")


# ---------------------------------------------------------------------------
# Minimal deterministic solid-color JPEG encoder (baseline, 4:2:0).
#
# The entropy data is a single DC coefficient per 8x8 block; all AC coefficients
# are zero (a perfectly flat image). The Huffman tables below are the exact
# canonical tables Pillow 8.2.0 emits for quality=75 baseline JPEG, so the
# output decodes with any JPEG reader without a Pillow runtime dependency.
# ---------------------------------------------------------------------------

_APP0: Final = bytes.fromhex("ffe000104a46494600010100000100010000")
_DQT: Final = bytes.fromhex(
    "ffdb004300080606070605080707070909080a0c140d0c0b0b0c1912130f141d1a1f1e1d1a1c1c20242e2720222c231c1c2837292c30313434341f27393d38323c2e333432"
    "ffdb0043010909090c0b0c180d0d1832211c213232323232323232323232323232323232323232323232323232323232323232323232323232323232323232323232323232"
)
_DHT: Final = bytes.fromhex(
    "ffc4001f0000010501010101010100000000000000000102030405060708090a0b"
    "ffc400b5100002010303020403050504040000017d01020300041105122131410613516107227114328191a1082342b1c11552d1f02433627282090a161718191a25262728292a3435363738393a434445464748494a535455565758595a636465666768696a737475767778797a838485868788898a92939495969798999aa2a3a4a5a6a7a8a9aab2b3b4b5b6b7b8b9bac2c3c4c5c6c7c8c9cad2d3d4d5d6d7d8d9dae1e2e3e4e5e6e7e8e9eaf1f2f3f4f5f6f7f8f9fa"
    "ffc4001f0100030101010101010101010000000000000102030405060708090a0b"
    "ffc400b51100020102040403040705040400010277000102031104052131061241510761711322328108144291a1b1c109233352f0156272d10a162434e125f11718191a262728292a35363738393a434445464748494a535455565758595a636465666768696a737475767778797a82838485868788898a92939495969798999aa2a3a4a5a6a7a8a9aab2b3b4b5b6b7b8b9bac2c3c4c5c6c7c8c9cad2d3d4d5d6d7d8d9dae2e3e4e5e6e7e8e9eaf2f3f4f5f6f7f8f9fa"
)
_SOS_HEADER: Final = bytes.fromhex("000c03010002110311003f00")

_LUMA_DC: Final = {
    "0": "00",
    "1": "010",
    "2": "011",
    "3": "100",
    "4": "101",
    "5": "110",
    "6": "1110",
    "7": "11110",
    "8": "111110",
    "9": "1111110",
    "10": "11111110",
    "11": "111111110",
}
_CHROMA_DC: Final = {
    "0": "00",
    "1": "01",
    "2": "10",
    "3": "110",
    "4": "1110",
    "5": "11110",
    "6": "111110",
    "7": "1111110",
    "8": "11111110",
    "9": "111111110",
    "10": "1111111110",
    "11": "11111111110",
}
_LUMA_EOB: Final = "1010"
_CHROMA_EOB: Final = "00"


class _BitWriter:
    def __init__(self) -> None:
        self.bits: list[str] = []

    def put(self, value: str) -> None:
        self.bits.extend(value)

    def bytes(self) -> bytes:
        output = bytearray()
        accumulator = 0
        count = 0
        for bit in self.bits:
            accumulator = (accumulator << 1) | (1 if bit == "1" else 0)
            count += 1
            if count == 8:
                output.append(accumulator)
                if accumulator == 0xFF:
                    output.append(0x00)
                accumulator = 0
                count = 0
        if count:
            accumulator = (accumulator << (8 - count)) | ((1 << (8 - count)) - 1)
            output.append(accumulator)
            if accumulator == 0xFF:
                output.append(0x00)
        return bytes(output)


def _dc_bits(coefficient: int, codes: dict[str, str]) -> tuple[str, str]:
    if coefficient == 0:
        return codes["0"], ""
    category = abs(coefficient).bit_length()
    code = codes[str(category)]
    if coefficient > 0:
        amplitude = format(coefficient, f"0{category}b")
    else:
        amplitude = format(coefficient + (1 << category) - 1, f"0{category}b")
    return code, amplitude


def solid_color_jpeg(rgb: tuple[int, int, int], width: int, height: int) -> bytes:
    """Encode one flat-color baseline JPEG (4:2:0) without an image library."""
    if width <= 0 or height <= 0 or width % 16 or height % 16:
        raise SourcePreparationError("Placeholder dimensions must be positive multiples of 16")
    red, green, blue = rgb
    y = 0.299 * red + 0.587 * green + 0.114 * blue
    cb = 128 - 0.168736 * red - 0.331264 * green + 0.5 * blue
    cr = 128 + 0.5 * red - 0.418688 * green - 0.081312 * blue
    y_dc = round(y - 128)
    cb_dc = round(8 * (cb - 128) / 9)
    cr_dc = round(8 * (cr - 128) / 9)

    writer = _BitWriter()
    y_pred = cb_pred = cr_pred = 0
    mcu_count = (width // 16) * (height // 16)
    for _ in range(mcu_count):
        for block_index in range(4):
            coefficient = y_dc if block_index == 0 else 0
            code, amplitude = _dc_bits(coefficient - y_pred, _LUMA_DC)
            writer.put(code)
            writer.put(amplitude)
            writer.put(_LUMA_EOB)
            y_pred = coefficient
        for coefficient, codes, eob, prediction in (
            (cb_dc, _CHROMA_DC, _CHROMA_EOB, cb_pred),
            (cr_dc, _CHROMA_DC, _CHROMA_EOB, cr_pred),
        ):
            code, amplitude = _dc_bits(coefficient - prediction, codes)
            writer.put(code)
            writer.put(amplitude)
            writer.put(eob)
        cb_pred = cb_dc
        cr_pred = cr_dc

    sof0 = (
        b"\xff\xc0\x00\x11\x08"
        + height.to_bytes(2, "big")
        + width.to_bytes(2, "big")
        + bytes.fromhex("03012200021101031101")
    )
    sos = b"\xff\xda" + _SOS_HEADER
    return b"\xff\xd8" + _APP0 + _DQT + sof0 + _DHT + sos + writer.bytes() + b"\xff\xd9"


def jpeg_dimensions(path: Path) -> tuple[int, int]:
    """Read JPEG SOF dimensions without an image-processing dependency."""
    data = path.read_bytes()
    if len(data) < 4 or data[:2] != b"\xff\xd8" or data[-2:] != b"\xff\xd9":
        raise SourcePreparationError(f"Not a complete JPEG file: {path}")
    offset = 2
    sof_markers = {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}
    while offset + 4 <= len(data):
        if data[offset] != 0xFF:
            offset += 1
            continue
        marker = data[offset + 1]
        offset += 2
        if marker in {0xD8, 0xD9} or 0xD0 <= marker <= 0xD7:
            continue
        if offset + 2 > len(data):
            break
        length = struct.unpack(">H", data[offset : offset + 2])[0]
        if marker in sof_markers and offset + 7 <= len(data):
            height, width = struct.unpack(">HH", data[offset + 3 : offset + 7])
            return width, height
        if length < 2:
            break
        offset += length
    raise SourcePreparationError(f"JPEG has no supported SOF marker: {path}")


def selection_rule() -> dict[str, object]:
    items = build_items()
    return {
        "version": SELECTION_VERSION,
        "ordered_product_types": list(TARGET_TYPES),
        "type_item_counts": dict(TYPE_ITEM_COUNTS),
        "authored_catalog_version": "synthetic-commerce-catalog-v1",
        "language": "authored English only",
        "image_policy": (
            "every selected record carries one hash-pinned 256x256 project-generated "
            "synthetic catalog JPEG; these are not photographs of real products"
        ),
        "candidate_order": "fixed authored catalog order with stable SYN- sequence ids",
        "selection_order": (
            "intent slices first (strong, partial, intra_type_hard_negative), then background "
            "types, in the authored catalog order"
        ),
        "global_uniqueness": ["item_id", "selected_image_id", "object_path"],
        "intent_curation": curation_contract(),
        "synthetic_item_count": len(items),
    }


def build_source_items(image_root: Path = DEFAULT_DATASET_ROOT) -> list[dict[str, object]]:
    """Assemble manifest items from the catalog and canonical generated images."""
    items: list[dict[str, object]] = []
    for catalog_item in build_items():
        sequence = int(catalog_item["sequence"])
        item_id = str(catalog_item["id"])
        product_type = str(catalog_item["product_type"])
        object_path = f"images/{sequence:03d}-{item_id}.jpg"
        source_path = image_root / object_path
        if not source_path.is_file():
            raise SourcePreparationError(f"Missing canonical product image: {source_path}")
        image_bytes = source_path.read_bytes()
        if jpeg_dimensions(source_path) != (PRODUCT_IMAGE_WIDTH, PRODUCT_IMAGE_HEIGHT):
            raise SourcePreparationError(f"Product image dimensions are invalid: {source_path}")
        md5 = _md5_bytes(image_bytes)
        items.append(
            {
                "item_id": item_id,
                "sequence": sequence,
                "official_metadata": {
                    "title": catalog_item["title"],
                    "product_type": product_type,
                    "brand": catalog_item["brand"],
                    "color": catalog_item["color"],
                    "material": catalog_item["material"],
                    "style": catalog_item["style"],
                    "node_name": catalog_item["node_name"],
                    "description": catalog_item["description"],
                    "bullet_points": list(catalog_item["bullet_points"]),
                },
                "selection_class": catalog_item["selection_class"],
                "selection_reason": catalog_item["selection_reason"],
                "metadata_provenance": {"title": SYNTHETIC_METADATA_FIELD},
                "main_image_id": f"SYN-IMG-{sequence:03d}",
                "selected_image_id": f"SYN-IMG-{sequence:03d}",
                "image_role": "main",
                "object_path": object_path,
                "url": f"synthetic://catalog/{item_id}/generated-product.jpg",
                "bytes": len(image_bytes),
                "md5": md5,
                "sha256": _sha256_bytes(image_bytes),
                "image_width": PRODUCT_IMAGE_WIDTH,
                "image_height": PRODUCT_IMAGE_HEIGHT,
                "http_head": {"etag": md5, "media_type": "image/jpeg"},
            }
        )
    return items


def prepare_source(
    cache: Path, output: Path, image_root: Path = DEFAULT_DATASET_ROOT
) -> dict[str, object]:
    """Copy canonical synthetic images and static texts, then emit the manifest."""
    cache.mkdir(parents=True, exist_ok=True)
    items = build_source_items(image_root)
    validate()

    for static_object in STATIC_OBJECTS:
        destination = cache / str(static_object["object_path"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        content = _static_content(str(static_object["kind"]))
        if (
            len(content) != int(static_object["bytes"])
            or _sha256_bytes(content) != static_object["sha256"]
        ):
            raise SourcePreparationError("Static object digest mismatch")
        destination.write_bytes(content)

    for item in items:
        destination = cache / str(item["object_path"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(image_root / str(item["object_path"]), destination)
        if (
            destination.stat().st_size != int(item["bytes"])
            or sha256_file(destination) != item["sha256"]
            or jpeg_dimensions(destination) != (PRODUCT_IMAGE_WIDTH, PRODUCT_IMAGE_HEIGHT)
        ):
            raise SourcePreparationError(
                f"Product image verification failed: {item['object_path']}"
            )

    manifest = {
        "schema_version": 1,
        "dataset": "Synthetic Commerce Catalog",
        "revision": DATASET_REVISION,
        "source_type": "synthetic_catalog_with_simulated_operations",
        "selection_rule": selection_rule(),
        "static_objects": [dict(item) for item in STATIC_OBJECTS],
        "items": items,
        "derived_localized_title_summary_provenance": DERIVED_ENGLISH_TITLE_PROVENANCE,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_json(manifest))
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--cache", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--image-root", type=Path, default=DEFAULT_DATASET_ROOT)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = prepare_source(args.cache, args.output, args.image_root)
    print(json.dumps({"revision": manifest["revision"], "items": len(manifest["items"])}))


if __name__ == "__main__":
    main()
