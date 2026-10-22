from __future__ import annotations

import pytest

from movie_discovery_demo.data import DatasetBundle
from movie_discovery_demo.embedding import EmbeddingBatchReport, ModelIdentity
from movie_discovery_demo.repository import GroupedBucket, MovieHit
from movie_discovery_demo.service import (
    MovieDiscoveryService,
    SearchContractError,
)


class FakeEmbedder:
    def __init__(self) -> None:
        self.loaded = True
        self.cache_available = True
        self.identity = ModelIdentity(
            model_id="test/model",
            revision="abc",
            vector_dimension=4,
            dense_output_name="dense_vecs",
            device="cpu",
        )
        self.calls: list[list[str]] = []

    def load(self) -> None:
        self.loaded = True

    def unload(self) -> None:
        self.loaded = False

    def embed(self, texts: list[str]) -> tuple[list[list[float]], EmbeddingBatchReport]:
        self.calls.append(texts)
        vectors = [[1.0, 0.0, 0.0, 0.0] for _ in texts]
        report = EmbeddingBatchReport(
            vector_count=len(vectors), vector_dimension=4, norm_min=1.0, norm_max=1.0
        )
        return vectors, report


class FakeRepository:
    def __init__(self) -> None:
        self.status_result: dict[str, object] = {
            "milvus_uri": "http://127.0.0.1:49530",
            "server_version": "3.0.0",
            "collection_name": "c",
            "collection_exists": True,
            "raw_collection_names": ["c"],
            "row_count": 90,
            "index_names": ["i1"],
        }
        self.last_prepare_args: tuple | None = None

    def status(self) -> dict[str, object]:
        return self.status_result

    def prepare(self, movies, synopsis_vectors):
        self.last_prepare_args = (movies, synopsis_vectors)

        class Report:
            def public_dict(self) -> dict[str, object]:
                return {"status": "prepared"}

        return Report()

    def list_search(self, *, query_vector, movies_by_id, limit, filter=None):
        first = list(movies_by_id.values())[0]
        return (MovieHit(rank=1, score=0.9, movie=first),)

    def aggregated_search(
        self,
        *,
        query_vector,
        movies_by_id,
        facet,
        filter=None,
        top_hits_size=3,
        root_size=6,
    ):
        first = list(movies_by_id.values())[0]
        return (
            GroupedBucket(
                key_value=first.primary_genre,
                count=1,
                hits=(MovieHit(rank=1, score=0.9, movie=first),),
            ),
        )

    def cleanup(self) -> dict[str, object]:
        return {"status": "clean"}


def _service(bundle: DatasetBundle) -> MovieDiscoveryService:
    embedder = FakeEmbedder()
    repository = FakeRepository()
    svc = MovieDiscoveryService(embedder=embedder, repository=repository)
    svc._bundle = bundle
    svc._movies_by_id = {movie.movie_id: movie for movie in bundle.movies}
    return svc


def test_grouped_search_returns_buckets(bundle: DatasetBundle) -> None:
    svc = _service(bundle)
    report = svc.search(query="a lighthearted adventure")
    payload = report.public_dict()
    assert payload["status"] == "passed"
    assert payload["view"] == "grouped"
    assert payload["grouped"]["buckets"][0]["key"]
    assert payload["grouped"]["buckets"][0]["count"] == 1
    assert payload["list"]["results"] == []
    assert payload["retained_candidate_count"] == 1


def test_list_search_returns_plain_hits(bundle: DatasetBundle) -> None:
    svc = _service(bundle)
    report = svc.search(query="a cozy animated story", view="list", limit=4)
    payload = report.public_dict()
    assert payload["view"] == "list"
    assert len(payload["list"]["results"]) == 1
    assert payload["grouped"]["buckets"] == []
    assert payload["retained_candidate_count"] == 1


def test_filter_present_when_max_runtime_set(bundle: DatasetBundle) -> None:
    svc = _service(bundle)
    report = svc.search(query="a quiet drama", max_runtime=120)
    assert report.public_dict()["filter"] == "runtime_minutes <= 120"


def test_search_rejects_unknown_facet(bundle: DatasetBundle) -> None:
    svc = _service(bundle)
    with pytest.raises(SearchContractError):
        svc.search(query="x", facet="country")


def test_search_rejects_bad_view(bundle: DatasetBundle) -> None:
    svc = _service(bundle)
    with pytest.raises(SearchContractError):
        svc.search(query="x", view="grid")


def test_search_rejects_empty_query(bundle: DatasetBundle) -> None:
    svc = _service(bundle)
    with pytest.raises(SearchContractError):
        svc.search(query="   ")


def test_search_rejects_out_of_range_limit(bundle: DatasetBundle) -> None:
    svc = _service(bundle)
    with pytest.raises(SearchContractError):
        svc.search(query="x", limit=0)
    with pytest.raises(SearchContractError):
        svc.search(query="x", limit=21)
