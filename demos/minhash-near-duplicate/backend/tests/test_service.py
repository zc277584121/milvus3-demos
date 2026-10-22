from __future__ import annotations

from minhash_near_dup_demo.data import DatasetBundle
from minhash_near_dup_demo.repository import MilvusHit
from minhash_near_dup_demo.service import MinHashService, SearchContractError


class FakeRepository:
    def __init__(self) -> None:
        self.owns_collection = True
        self.prepared = False
        self.adopted = False
        self.last_search: tuple[str, int] | None = None

    def raw_audit(self):
        return {
            "milvus_uri": "http://127.0.0.1:49530",
            "server_version": "3.0.0",
            "expected_server_version": "3.0.0",
            "raw_collection_names": ["milvus3_demos_minhash_near_duplicate"],
            "target_collection": "milvus3_demos_minhash_near_duplicate",
            "target_exists": True,
            "index_names": ["milvus3_demos_minhash_signature"],
            "indexes": [],
            "schema": {},
            "collection_stats": {"row_count": 36},
        }

    def prepare(self, documents):
        self.prepared = True

        class Report:
            def public_dict(self):
                return {"status": "prepared"}

        return Report()

    def adopt(self, document_count: int) -> bool:
        self.adopted = True
        return True

    def search(self, query_text: str, *, limit: int) -> tuple[MilvusHit, ...]:
        self.last_search = (query_text, limit)
        light_edits = (
            "On Sunday, we followed the river path to the old bridge. The morning market was "
            "already busy, so we bought bread and coffee before continuing. Beyond the bridge, a "
            "narrow trail led to a quiet garden. We spent the afternoon there and returned to "
            "town before sunset."
        )
        original = (
            "On Saturday, we followed the river path to the red bridge. The morning market was "
            "already busy, so we bought bread and tea before continuing. Beyond the bridge, a "
            "narrow trail led to a quiet garden. We spent the afternoon there and returned to "
            "town before sunset."
        )
        return (
            MilvusHit(
                doc_id="river-weekend-003.txt", title="t", content=light_edits, distance=0.94
            ),
            MilvusHit(
                doc_id="river-weekend-001.txt", title="t", content=original, distance=0.87
            ),
        )

    def cleanup(self):
        class Report:
            def public_dict(self):
                return {"status": "clean"}

        return Report()


def test_search_reports_estimated_jaccard_directly(bundle: DatasetBundle) -> None:
    repository = FakeRepository()
    service = MinHashService(config=None, repository=repository)
    service._bundle = bundle  # noqa: SLF001 - test fixture wiring
    report = service.search(query="On Sunday, we followed the river path to the old bridge.")
    payload = report.public_dict()
    assert payload["status"] == "passed"
    # MHJACCARD returns the estimated Jaccard similarity directly, so the
    # estimated_jaccard equals the raw_distance rather than 1 - distance.
    assert payload["results"][0]["estimated_jaccard"] == 0.94
    assert payload["results"][0]["raw_distance"] == 0.94
    assert payload["results"][0]["above_threshold"] is True
    assert payload["results"][1]["estimated_jaccard"] == 0.87
    assert payload["results"][1]["above_threshold"] is True
    assert payload["above_threshold_count"] == 2


def test_search_rejects_empty_query(bundle: DatasetBundle) -> None:
    repository = FakeRepository()
    service = MinHashService(config=None, repository=repository)
    service._bundle = bundle  # noqa: SLF001 - test fixture wiring
    try:
        service.search(query="   ")
    except SearchContractError:
        return
    raise AssertionError("expected SearchContractError for empty query")


def test_search_rejects_out_of_range_threshold(bundle: DatasetBundle) -> None:
    repository = FakeRepository()
    service = MinHashService(config=None, repository=repository)
    service._bundle = bundle  # noqa: SLF001 - test fixture wiring
    try:
        service.search(query="hello", threshold=1.5)
    except SearchContractError:
        return
    raise AssertionError("expected SearchContractError for out-of-range threshold")
