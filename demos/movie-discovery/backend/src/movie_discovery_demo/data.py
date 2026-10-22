"""Deterministic loader for the checked-in movie catalog dataset."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from movie_discovery_demo.config import (
    EXPECTED_DATASET_ID,
    EXPECTED_DATASET_VERSION,
    EXPECTED_MOVIE_COUNT,
    GENRES,
    RuntimeConfig,
)


class DataContractError(ValueError):
    """Raised when the movie catalog violates the fixed contract."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _required_str(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DataContractError(f"{field} must be a non-empty string")
    return value.strip()


def _required_int(value: object, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise DataContractError(f"{field} must be an integer")
    return value


def _safe_poster_path(value: object, field: str) -> str:
    path = _required_str(value, field)
    if path != f"posters/{Path(path).name}":
        raise DataContractError(f"{field} must be a bare posters/*.jpg path")
    return path


@dataclass(frozen=True)
class MovieRecord:
    movie_id: str
    title: str
    synopsis: str
    primary_genre: str
    release_year: int
    decade: str
    region: str
    runtime_minutes: int
    poster_path: str
    poster_palette: tuple[str, str, str]


@dataclass(frozen=True)
class PresetRecord:
    id: str
    label: str
    text: str
    description: str

    def public_dict(self) -> dict[str, str]:
        return {
            "id": self.id,
            "label": self.label,
            "text": self.text,
            "description": self.description,
        }


@dataclass(frozen=True)
class DatasetBundle:
    movies: tuple[MovieRecord, ...]
    presets: tuple[PresetRecord, ...]
    dataset_id: str
    dataset_version: int
    movies_sha256: str
    presets_sha256: str

    @property
    def movie_count(self) -> int:
        return len(self.movies)


def _parse_movie(raw: object, index: int) -> MovieRecord:
    if not isinstance(raw, dict):
        raise DataContractError(f"movies[{index}] must be an object")
    movie_id = _required_str(raw.get("movie_id"), f"movies[{index}].movie_id")
    if movie_id != f"m{index + 1:04d}":
        raise DataContractError(f"Unexpected ordered movie_id: {movie_id}")
    title = _required_str(raw.get("title"), f"{movie_id}.title")
    synopsis = _required_str(raw.get("synopsis"), f"{movie_id}.synopsis")
    primary_genre = _required_str(raw.get("primary_genre"), f"{movie_id}.primary_genre")
    if primary_genre not in GENRES:
        raise DataContractError(f"{movie_id} has an unsupported primary_genre: {primary_genre}")
    release_year = _required_int(raw.get("release_year"), f"{movie_id}.release_year")
    if not 1900 <= release_year <= 2100:
        raise DataContractError(f"{movie_id} release_year is out of range")
    decade = _required_str(raw.get("decade"), f"{movie_id}.decade")
    expected_decade = f"{release_year // 10 * 10}s"
    if decade != expected_decade:
        raise DataContractError(f"{movie_id} decade {decade} != {expected_decade}")
    region = _required_str(raw.get("region"), f"{movie_id}.region")
    runtime_minutes = _required_int(raw.get("runtime_minutes"), f"{movie_id}.runtime_minutes")
    if not 1 <= runtime_minutes <= 500:
        raise DataContractError(f"{movie_id} runtime_minutes is out of range")
    poster_path = _safe_poster_path(raw.get("poster_path"), f"{movie_id}.poster_path")
    if poster_path != f"posters/{movie_id}.jpg":
        raise DataContractError(f"{movie_id} poster_path must be posters/{movie_id}.jpg")
    palette_raw = raw.get("poster_palette")
    if not isinstance(palette_raw, list) or len(palette_raw) != 3:
        raise DataContractError(f"{movie_id}.poster_palette must contain three colors")
    palette = tuple(_required_str(item, f"{movie_id}.poster_palette") for item in palette_raw)
    for color in palette:
        if len(color) != 7 or not color.startswith("#"):
            raise DataContractError(f"{movie_id} has an invalid poster palette color: {color}")
    return MovieRecord(
        movie_id=movie_id,
        title=title,
        synopsis=synopsis,
        primary_genre=primary_genre,
        release_year=release_year,
        decade=decade,
        region=region,
        runtime_minutes=runtime_minutes,
        poster_path=poster_path,
        poster_palette=palette,
    )


def _parse_preset(raw: object, index: int) -> PresetRecord:
    if not isinstance(raw, dict):
        raise DataContractError(f"presets[{index}] must be an object")
    preset_id = _required_str(raw.get("id"), f"presets[{index}].id")
    label = _required_str(raw.get("label"), f"{preset_id}.label")
    text = _required_str(raw.get("text"), f"{preset_id}.text")
    description = _required_str(raw.get("description"), f"{preset_id}.description")
    return PresetRecord(id=preset_id, label=label, text=text, description=description)


def build_dataset(config: RuntimeConfig) -> DatasetBundle:
    """Load and validate the checked-in movie catalog and preset queries."""
    if not config.movies_path.is_file():
        raise DataContractError(f"Missing movies file: {config.movies_path}")
    if not config.presets_path.is_file():
        raise DataContractError(f"Missing presets file: {config.presets_path}")

    try:
        movies_raw = json.loads(config.movies_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DataContractError(f"Invalid movies JSON: {exc}") from exc
    if not isinstance(movies_raw, dict):
        raise DataContractError("movies.json must be a JSON object")
    dataset_id = _required_str(movies_raw.get("dataset_id"), "dataset_id")
    if dataset_id != EXPECTED_DATASET_ID:
        raise DataContractError(f"Unexpected dataset identity: {dataset_id}")
    version = _required_int(movies_raw.get("version"), "version")
    if version != EXPECTED_DATASET_VERSION:
        raise DataContractError(f"Unexpected dataset version: {version}")
    raw_movies = movies_raw.get("movies")
    if not isinstance(raw_movies, list) or len(raw_movies) != EXPECTED_MOVIE_COUNT:
        raise DataContractError(f"movies.json must contain {EXPECTED_MOVIE_COUNT} movies")
    movies = tuple(_parse_movie(raw, index) for index, raw in enumerate(raw_movies))
    movie_ids = [movie.movie_id for movie in movies]
    if len(movie_ids) != len(set(movie_ids)):
        raise DataContractError("Movie ids must be unique")

    try:
        presets_raw = json.loads(config.presets_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DataContractError(f"Invalid presets JSON: {exc}") from exc
    if not isinstance(presets_raw, dict):
        raise DataContractError("presets.json must be a JSON object")
    raw_presets = presets_raw.get("presets")
    if not isinstance(raw_presets, list) or not raw_presets:
        raise DataContractError("presets.json must contain a non-empty preset list")
    presets = tuple(_parse_preset(raw, index) for index, raw in enumerate(raw_presets))
    preset_ids = [preset.id for preset in presets]
    if len(preset_ids) != len(set(preset_ids)):
        raise DataContractError("Preset ids must be unique")

    return DatasetBundle(
        movies=movies,
        presets=presets,
        dataset_id=dataset_id,
        dataset_version=EXPECTED_DATASET_VERSION,
        movies_sha256=_sha256(config.movies_path),
        presets_sha256=_sha256(config.presets_path),
    )
