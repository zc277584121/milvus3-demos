from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from embedding_list_demo.config import default_dataset_root
from embedding_list_demo.manual import (
    DATASET_REVISION,
    DISTRIBUTION,
    PAGE_COUNT,
    QUERY_SPECS,
    RIGHTS_DETERMINATION,
    SOURCE_PDF_SHA256,
    ManualContractError,
    compare_dataset_trees,
    generate_manual,
    load_manual,
    load_queries,
    query_ground_truth,
    resolve_page_image,
    sha256_file,
)


def source_pdf() -> Path:
    manifest = load_manual(default_dataset_root())
    return default_dataset_root() / manifest.pdf_filename


def test_dataset_rebuild_is_byte_identical_to_the_tracked_revision(tmp_path: Path) -> None:
    tracked = load_manual(default_dataset_root())
    rebuilt_root = tmp_path / "rebuilt"
    rebuilt = generate_manual(rebuilt_root, source_pdf())

    compare_dataset_trees(default_dataset_root(), rebuilt_root)
    assert rebuilt.public_dict() == tracked.public_dict()
    assert rebuilt.page_count == PAGE_COUNT == 40
    assert rebuilt.revision == DATASET_REVISION == "NASA/SP-2016-6105 Rev 2"
    assert rebuilt.distribution == DISTRIBUTION == "PUBLIC"
    assert rebuilt.rights_determination == RIGHTS_DETERMINATION == "PUBLIC_USE_PERMITTED"
    assert rebuilt.pdf_sha256 == SOURCE_PDF_SHA256
    assert rebuilt.render_dpi == 144
    assert rebuilt.pdf_page_index_base == 1
    assert rebuilt.renderer == "PyMuPDF"
    assert rebuilt.renderer_version == "1.28.2"
    assert len(rebuilt.authors) == 3
    assert [author.name for author in rebuilt.authors] == [
        "Steven R. Hirshorn",
        "Linda D. Voss",
        "Linda K. Bromley",
    ]
    assert all(page.section and page.title for page in rebuilt.pages)
    assert [(page.pdf_page_index, page.printed_page) for page in rebuilt.pages] == [
        (64, 51),
        (66, 53),
        (69, 56),
        (73, 60),
        (77, 64),
        (79, 66),
        (81, 68),
        (86, 73),
        (91, 78),
        (92, 79),
        (96, 83),
        (102, 89),
        (104, 91),
        (112, 99),
        (116, 103),
        (118, 105),
        (125, 112),
        (130, 117),
        (138, 125),
        (146, 133),
        (164, 151),
        (170, 157),
        (174, 161),
        (176, 163),
        (178, 165),
        (181, 168),
        (182, 169),
        (184, 171),
        (185, 172),
        (189, 176),
        (197, 184),
        (198, 185),
        (206, 193),
        (208, 195),
        (249, 236),
        (253, 240),
        (256, 243),
        (261, 248),
        (269, 256),
        (272, 259),
    ]
    assert len({page.image_sha256 for page in rebuilt.pages}) == PAGE_COUNT


def test_natural_queries_and_internal_evaluation_mappings_are_fixed() -> None:
    manifest = load_manual(default_dataset_root())
    queries = load_queries(default_dataset_root(), manifest)

    assert queries == QUERY_SPECS
    assert [query.query_id for query in queries] == [
        "Q01",
        "Q02",
        "Q03",
        "Q04",
        "Q05",
        "Q06",
        "Q07",
        "Q08",
    ]
    assert [query.target_page_ids[0] for query in queries] == [
        "nasa-seh-printed-053",
        "nasa-seh-printed-068",
        "nasa-seh-printed-103",
        "nasa-seh-printed-161",
        "nasa-seh-printed-172",
        "nasa-seh-printed-195",
        "nasa-seh-printed-060",
        "nasa-seh-printed-083",
    ]
    assert all(len(query.hard_negatives) == 1 for query in queries)
    assert all(set(query.public_dict()) == {"query_id", "text"} for query in queries)
    assert query_ground_truth(queries, queries[0].text) == queries[0]
    assert query_ground_truth(queries, "an open-ended query") is None


def test_dataset_hash_tampering_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    shutil.copytree(default_dataset_root(), root)
    manifest = load_manual(root)
    page = resolve_page_image(root, manifest, "nasa-seh-printed-053")
    page.write_bytes(page.read_bytes() + b"tamper")

    with pytest.raises(ManualContractError, match="SHA-256 verification failed"):
        load_manual(root)


def test_dataset_refuses_overwrite_unknown_page_and_query_tampering(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    shutil.copytree(default_dataset_root(), root)
    manifest = load_manual(root)

    with pytest.raises(ManualContractError, match="Refusing to overwrite"):
        generate_manual(root, source_pdf())
    with pytest.raises(ManualContractError, match="Unknown NASA handbook page ID"):
        resolve_page_image(root, manifest, "missing")

    queries_path = root / manifest.queries_filename
    value = json.loads(queries_path.read_text(encoding="utf-8"))
    value["queries"][0]["target_page_ids"] = ["nasa-seh-printed-064"]
    queries_path.write_text(json.dumps(value), encoding="utf-8")
    assert sha256_file(queries_path) != manifest.queries_sha256
    with pytest.raises(ManualContractError, match="Query SHA-256"):
        load_queries(root, manifest)
