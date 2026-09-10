from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from embedding_list_demo.config import default_dataset_root
from embedding_list_demo.manual import load_manual
from embedding_list_demo.repository import RepositoryContractError, _rows, normalize_sdk_names


@dataclass
class NamedItem:
    name: str


def test_normalize_sdk_names_handles_string_and_named_objects() -> None:
    assert normalize_sdk_names(
        [NamedItem("beta"), "alpha"],
        operation="list_collections()",
    ) == ("alpha", "beta")


def test_normalize_sdk_names_rejects_unknown_and_duplicate_shapes() -> None:
    with pytest.raises(TypeError, match="nonempty string name"):
        normalize_sdk_names([object()], operation="list_indexes()")
    with pytest.raises(RepositoryContractError, match="duplicate"):
        normalize_sdk_names(["same", NamedItem("same")], operation="list_collections()")


def test_rows_carry_structured_ntrs_and_page_metadata() -> None:
    manual = load_manual(default_dataset_root())
    rows = _rows(manual, [torch.ones((2, 128)) for _ in manual.pages])

    assert len(rows) == 40
    assert rows[1]["page_id"] == "nasa-seh-printed-053"
    assert rows[1]["pdf_page_index"] == 66
    assert rows[1]["printed_page"] == 53
    assert rows[1]["document_identifier"] == "NASA/SP-2016-6105 Rev 2"
    assert rows[1]["ntrs_id"] == 20170001761
    assert rows[1]["distribution"] == "PUBLIC"
    assert rows[1]["rights_determination"] == "PUBLIC_USE_PERMITTED"
    assert rows[1]["contains_third_party_material"] is False
    assert len(rows[1]["patches"]) == 2  # type: ignore[arg-type]
