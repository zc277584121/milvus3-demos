from __future__ import annotations

from minhash_near_dup_demo.config import EXPECTED_DATASET_ID, EXPECTED_DOCUMENT_COUNT
from minhash_near_dup_demo.data import DatasetBundle


def test_dataset_identity(bundle: DatasetBundle) -> None:
    assert bundle.dataset_id == EXPECTED_DATASET_ID
    assert bundle.dataset_version == 1


def test_document_count(bundle: DatasetBundle) -> None:
    assert bundle.document_count == EXPECTED_DOCUMENT_COUNT


def test_every_variant_kind_is_covered(bundle: DatasetBundle) -> None:
    variants = {document.variant for document in bundle.documents}
    assert {
        "original",
        "exact-copy",
        "light-edits",
        "paragraph-removed",
        "paragraph-added",
        "same-theme",
        "unrelated",
    } <= variants


def test_exact_copies_match_their_originals(bundle: DatasetBundle) -> None:
    originals = {
        document.cluster: document.content
        for document in bundle.documents
        if document.variant == "original"
    }
    for document in bundle.documents:
        if document.variant == "exact-copy":
            assert document.content == originals[document.cluster]


def test_light_edits_differ_but_keep_most_words(bundle: DatasetBundle) -> None:
    originals = {
        document.cluster: document.content
        for document in bundle.documents
        if document.variant == "original"
    }
    for document in bundle.documents:
        if document.variant == "light-edits":
            original = originals[document.cluster]
            assert document.content != original
            # A lightly edited copy should still share most of its words.
            original_words = set(original.split())
            edited_words = set(document.content.split())
            overlap = len(original_words & edited_words)
            assert overlap / len(original_words) > 0.8


def test_unrelated_documents_do_not_copy_originals(bundle: DatasetBundle) -> None:
    originals = [
        document.content for document in bundle.documents if document.variant == "original"
    ]
    for document in bundle.documents:
        if document.variant == "unrelated":
            assert document.content not in originals


def test_presets_are_present(bundle: DatasetBundle) -> None:
    assert len(bundle.presets) == 3
    preset_ids = {preset.id for preset in bundle.presets}
    assert preset_ids == {"river-weekend", "night-market", "train-hills"}
