from __future__ import annotations

from movie_discovery_demo.config import RuntimeConfig
from movie_discovery_demo.data import build_dataset


def test_bundle_contract(config: RuntimeConfig) -> None:
    bundle = build_dataset(config)
    assert bundle.dataset_id == "movie-catalog-r1"
    assert bundle.dataset_version == 1
    assert bundle.movie_count == 90
    assert len(bundle.presets) == 5
    assert bundle.movies_sha256
    assert bundle.presets_sha256


def test_movies_are_ordered_and_valid(config: RuntimeConfig) -> None:
    bundle = build_dataset(config)
    genres = {movie.primary_genre for movie in bundle.movies}
    assert genres == {"Adventure", "Animation", "Comedy", "Drama", "Sci-Fi", "Thriller"}
    for index, movie in enumerate(bundle.movies, start=1):
        assert movie.movie_id == f"m{index:04d}"
        assert movie.decade == f"{movie.release_year // 10 * 10}s"
        assert movie.poster_path == f"posters/{movie.movie_id}.jpg"
        assert len(movie.poster_palette) == 3


def test_presets_have_unique_ids(config: RuntimeConfig) -> None:
    bundle = build_dataset(config)
    ids = [preset.id for preset in bundle.presets]
    assert len(ids) == len(set(ids))
    for preset in bundle.presets:
        assert preset.text.strip()
