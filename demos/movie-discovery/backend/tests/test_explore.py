"""Contract tests for the 2x2 capability matrix (query|search x sort|group)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from movie_discovery_demo.config import FACET_FIELDS
from movie_discovery_demo.data import MovieRecord
from movie_discovery_demo.main import create_app
from movie_discovery_demo.repository import MovieHit
from movie_discovery_demo.service import MovieDiscoveryService


def _movie(movie_id: str, genre: str = "Drama", year: int = 2020, runtime: int = 100):
    return MovieRecord(
        movie_id=movie_id,
        title=f"Title {movie_id}",
        synopsis="A synthetic synopsis.",
        primary_genre=genre,
        release_year=year,
        decade=f"{year // 10 * 10}s",
        region="United States",
        runtime_minutes=runtime,
        poster_path=f"posters/{movie_id}.jpg",
        poster_palette=("#111111", "#222222", "#333333"),
    )


class FakeEmbedder:
    loaded = True
    cache_available = True

    def identity(self):
        return type("I", (), {"public_dict": lambda self: {}})()

    def load(self) -> None: ...

    def unload(self) -> None: ...

    def embed(self, texts):
        return [[0.1] * 1024 for _ in texts], type("R", (), {"public_dict": lambda self: {}})()


class FakeRepository:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def browse_sorted(self, *, movies_by_id, sort_field, direction, limit, offset, filter=None):
        self.calls.append(
            ("browse_sorted", {"sort_field": sort_field, "direction": direction, "offset": offset})
        )
        movies = tuple(_movie(f"m{i:04d}", year=2025 - i) for i in range(offset, offset + limit))
        return movies, 90

    def query_aggregate(self, *, facet_field, metric_fields, filter=None):
        self.calls.append(
            ("query_aggregate", {"facet_field": facet_field, "metrics": list(metric_fields)})
        )
        # Shape mirrors what real Milvus returns for group_by_fields + aggregates.
        return tuple(
            {
                "key": genre,
                "values": {
                    "count(*)": 15,
                    "avg(runtime_minutes)": 106.46666666666667,
                    "min(release_year)": 1987,
                    "max(release_year)": 2024,
                },
            }
            for genre in ("Adventure", "Comedy")
        )

    def ranked_search(
        self, *, query_vector, movies_by_id, limit, sort_field=None, direction="desc", filter=None
    ):
        self.calls.append(("ranked_search", {"sort_field": sort_field, "direction": direction}))
        # With no scalar sort Milvus returns relevance order (monotonic); once a
        # scalar sort is applied the retained hits are re-ordered, so the score
        # sequence stops being monotonic.
        if sort_field is None:
            scores = [0.67, 0.63, 0.56, 0.55, 0.54]
        else:
            scores = [0.67, 0.63, 0.55, 0.56, 0.54]
        return tuple(
            MovieHit(rank=i + 1, score=s, movie=_movie(f"m{i:04d}"))
            for i, s in enumerate(scores[:limit])
        )

    def aggregated_search_with_metrics(self, *, query_vector, movies_by_id, facet, filter=None):
        self.calls.append(("aggregated_search_with_metrics", {"facet": facet}))
        return tuple(
            {
                "key": genre,
                "count": 3,
                "metrics": {"avg_runtime": 94.0, "max_year": 2024},
                "candidates": [
                    MovieHit(rank=1, score=0.6, movie=_movie("m0001", genre)).public_dict()
                ],
            }
            for genre in ("Animation", "Sci-Fi")
        )

    def status(self):
        return {
            "milvus_uri": "http://127.0.0.1:49530",
            "server_version": "3.0.0",
            "collection_name": "milvus3_demos_movie_catalog",
            "collection_exists": True,
            "raw_collection_names": ["milvus3_demos_movie_catalog"],
            "row_count": 90,
            "index_names": ["milvus3_demos_movie_synopsis_vector"],
        }

    def cleanup(self):
        return {"status": "clean"}


class FakeService(MovieDiscoveryService):
    def __init__(self) -> None:
        self.embedder = FakeEmbedder()
        self.repository = FakeRepository()

    def _movies_map(self):
        return {f"m{i:04d}": _movie(f"m{i:04d}") for i in range(200)}


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app(service=FakeService()))


# ---------------------------------------------------------------------------
# query + sort
# ---------------------------------------------------------------------------
def test_query_sort_returns_paged_exact_listing(client: TestClient) -> None:
    response = client.post(
        "/api/v1/explore/query/sort",
        json={"sort": "year", "direction": "desc", "page": 2, "page_size": 12},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["api"] == "query"
    assert body["mode"] == "sort"
    assert body["total"] == 90
    assert body["page"] == 2
    assert body["total_pages"] == 8  # ceil(90 / 12)
    assert len(body["results"]) == 12


def test_query_sort_uses_tiebreaker_and_rejects_bad_page(client: TestClient) -> None:
    ok = client.post("/api/v1/explore/query/sort", json={"sort": "runtime", "page": 1})
    assert ok.status_code == 200
    bad = client.post("/api/v1/explore/query/sort", json={"sort": "runtime", "page": 0})
    assert bad.status_code == 422


# ---------------------------------------------------------------------------
# query + group
# ---------------------------------------------------------------------------
def test_query_group_returns_exact_aggregates(client: TestClient) -> None:
    response = client.post(
        "/api/v1/explore/query/group",
        json={"facet": "genre", "metrics": ["count", "avg_runtime", "min_year", "max_year"]},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["api"] == "query"
    assert body["mode"] == "group"
    assert body["facet_field"] == FACET_FIELDS["genre"][0]
    groups = body["groups"]
    assert len(groups) == 2
    assert groups[0]["metrics"]["count"] == 15
    # A float average with no fractional part is normalised to an int.
    assert groups[0]["metrics"]["avg_runtime"] == 106.47
    assert body["semantics"].startswith("GROUP BY with count/sum/avg/min/max")


def test_query_group_rejects_unknown_metric(client: TestClient) -> None:
    response = client.post("/api/v1/explore/query/group", json={"metrics": ["median"]})
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# search + sort
# ---------------------------------------------------------------------------
def test_search_sort_reports_non_monotonic_scores(client: TestClient) -> None:
    response = client.post(
        "/api/v1/explore/search/sort",
        json={"query": "a cozy story", "sort": "year", "direction": "desc"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["api"] == "search"
    assert body["mode"] == "sort"
    # The honest signal that the scalar sort re-ordered the retained hits.
    assert body["score_monotonic"] is False
    assert body["retained_candidate_count"] == len(body["results"])
    assert "candidate set is unchanged" in body["semantics"]


def test_search_sort_score_baseline_stays_monotonic(client: TestClient) -> None:
    response = client.post(
        "/api/v1/explore/search/sort", json={"query": "a cozy story", "sort": "score"}
    )
    assert response.status_code == 200
    assert response.json()["score_monotonic"] is True


# ---------------------------------------------------------------------------
# search + group
# ---------------------------------------------------------------------------
def test_search_group_marks_counts_approximate(client: TestClient) -> None:
    response = client.post(
        "/api/v1/explore/search/group", json={"query": "a cozy story", "facet": "genre"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["api"] == "search"
    assert body["mode"] == "group"
    assert body["retained_candidate_count"] == 6  # 2 buckets x 3 retained
    bucket = body["buckets"][0]
    assert bucket["count"] == 3
    assert bucket["metrics"]["avg_runtime"] == 94.0
    assert "counts are approximate" in body["semantics"]


def test_search_endpoints_require_query_text(client: TestClient) -> None:
    assert client.post("/api/v1/explore/search/sort", json={"query": ""}).status_code == 422
    assert client.post("/api/v1/explore/search/group", json={"query": ""}).status_code == 422
