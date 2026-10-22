from __future__ import annotations

from minhash_near_dup_demo.diff import compute_diff


def test_light_edit_diff_marks_removed_and_added() -> None:
    incoming = (
        "On Sunday, we followed the river path to the old bridge. "
        "The morning market was already busy, so we bought bread and coffee before continuing."
    )
    existing = (
        "On Saturday, we followed the river path to the red bridge. "
        "The morning market was already busy, so we bought bread and tea before continuing."
    )
    view = compute_diff(
        incoming_text=incoming,
        document_id="river-weekend-001.txt",
        document_text=existing,
    )
    kinds = [segment.kind for segment in view.segments]
    assert "removed" in kinds
    assert "added" in kinds
    assert "equal" in kinds
    # The specific changed words should be flagged.
    removed = " ".join(segment.text for segment in view.segments if segment.kind == "removed")
    added = " ".join(segment.text for segment in view.segments if segment.kind == "added")
    assert "Saturday" in removed
    assert "red" in removed
    assert "tea" in removed
    assert "Sunday" in added
    assert "old" in added
    assert "coffee" in added


def test_identical_text_is_all_equal() -> None:
    text = "The night market opened under strings of yellow lanterns."
    view = compute_diff(incoming_text=text, document_id="x.txt", document_text=text)
    assert all(segment.kind == "equal" for segment in view.segments)
    assert view.removed_count == 0
    assert view.added_count == 0


def test_paragraph_added_diff_reports_added_words() -> None:
    existing = "The market opened under lanterns. By midnight the alleys smelled of smoke."
    incoming = (
        "The market opened under lanterns. A fortune teller sat in the corner. "
        "By midnight the alleys smelled of smoke."
    )
    view = compute_diff(incoming_text=incoming, document_id="x.txt", document_text=existing)
    assert view.added_count > 0
    assert "fortune" in " ".join(
        segment.text for segment in view.segments if segment.kind == "added"
    )
