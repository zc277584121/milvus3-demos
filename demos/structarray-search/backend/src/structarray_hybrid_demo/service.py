"""Orchestrate data, embedding, and the three Milvus search paths."""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass
from typing import Protocol

from structarray_hybrid_demo.config import (
    DEFAULT_COLLAPSE_STRATEGY,
    DEFAULT_COLLAPSE_TOPK,
    DEFAULT_TOP_K,
    MAX_TOP_K,
    QUERY_PRESETS,
    SAMPLE_SIZE,
    WEIGHT_MAX,
    WEIGHT_MIN,
    RuntimeConfig,
)
from structarray_hybrid_demo.data import DatasetBundle, build_dataset
from structarray_hybrid_demo.embedding import BgeM3OnnxEmbedder, EmbeddingBatchReport, ModelIdentity
from structarray_hybrid_demo.repository import (
    ChildHit,
    FusionHit,
    ParentHit,
    SearchContractError,
    StructArrayHybridRepository,
)


class ServiceContractError(RuntimeError):
    """Raised when the end-to-end backend state is not ready for an operation."""


@dataclass(frozen=True)
class QueryIntent:
    scene_terms: tuple[str, ...]
    object_terms: tuple[str, ...]
    color_terms: tuple[str, ...] = ()

    def public_dict(self) -> dict[str, object]:
        return {
            "scene_terms": list(self.scene_terms),
            "object_terms": list(self.object_terms),
            "color_terms": list(self.color_terms),
        }


QUERY_INTENTS: dict[str, QueryIntent] = {
    preset[0]: QueryIntent(
        scene_terms=preset[2],
        object_terms=preset[3],
        color_terms=preset[4] if len(preset) > 4 else (),
    )
    for preset in QUERY_PRESETS
}


def _ndcg(ranked_ids: list[str], ground_truth_ids: list[str]) -> float:
    """Binary-graded NDCG@k for one ranked path against a ground-truth set.

    Each ground-truth hit contributes a relevance of 1; the ideal order places
    every ground-truth hit ahead of the misses. Returns 0.0 when the query has
    no ground truth or the path returns nothing.
    """
    if not ground_truth_ids or not ranked_ids:
        return 0.0
    gt = set(ground_truth_ids)
    relevance = [1 if video_id in gt else 0 for video_id in ranked_ids]

    def dcg(values: list[int]) -> float:
        return sum((2**rel - 1) / math.log2(i + 2) for i, rel in enumerate(values))

    ideal = dcg(sorted(relevance, reverse=True))
    if ideal == 0.0:
        return 0.0
    return round(dcg(relevance) / ideal, 4)


_COLOR_WORD_SPLIT = re.compile(r"[^a-z]+")
_COLOR_NOISE = {"and", "dark", "empty", "unknown"}


def _color_words(color: str) -> list[str]:
    """Split a color string into meaningful colour words ("blue and white" → two)."""
    return [
        word for word in _COLOR_WORD_SPLIT.split(color.lower()) if word and word not in _COLOR_NOISE
    ]


def _detected_scene(environment: str) -> str:
    """Extract the actual road type from the summary's environment segment."""
    match = re.search(r"road type:\s*([^,]+)", environment)
    return match.group(1).strip() if match else ""


def _representative_object(counts: dict[str, int], object_types: set[str]) -> str:
    """The matched object term when present, otherwise the most frequent type."""
    if not counts:
        return ""
    for term in object_types:
        if term in counts:
            return term
    return max(counts, key=lambda key: counts[key])


def _representative_color(
    matched_colors: dict[str, int],
    wrong_colors: dict[str, int],
    color_terms: set[str],
    color_matched: bool,
) -> str:
    """The intent color when matched, otherwise the most frequent wrong color.

    Counts are only taken from observations whose object_type matched the
    intent, so the value describes the queried object itself (never an
    unrelated object's colour). When the intent color does not match, the value
    is a colour word that actually differs from it (e.g. "red" for a
    "white and red" truck), so the UI can strike out the contradicting word.
    Returns "" when the matched object carries no usable colour word.
    """
    if color_matched:
        for term in color_terms:
            if term in matched_colors:
                return term
        if matched_colors:
            return max(matched_colors, key=lambda key: matched_colors[key])
        return ""
    if wrong_colors:
        return max(wrong_colors, key=lambda key: wrong_colors[key])
    if matched_colors:
        return max(matched_colors, key=lambda key: matched_colors[key])
    return ""


class EmbedderLike(Protocol):
    @property
    def loaded(self) -> bool: ...

    @property
    def identity(self) -> ModelIdentity: ...

    def load(self) -> None: ...

    def embed(self, texts: list[str]) -> tuple[list[list[float]], EmbeddingBatchReport]: ...

    def unload(self) -> None: ...


class RepositoryLike(Protocol):
    def status(self) -> dict[str, object]: ...

    def prepare(
        self,
        bundle: DatasetBundle,
        parent_vectors: list[list[float]],
        child_vectors: list[list[float]],
    ): ...

    def parent_search(self, *, query_vector, videos_by_id, limit) -> tuple[ParentHit, ...]: ...

    def child_search(self, *, query_vector, videos_by_id, limit) -> tuple[ChildHit, ...]: ...

    def fusion_search(
        self,
        *,
        query_vector,
        videos_by_id,
        limit,
        parent_weight,
        child_weight,
        collapse_strategy,
        collapse_topk,
    ) -> tuple[FusionHit, ...]: ...

    def cleanup(self) -> dict[str, object]: ...


@dataclass(frozen=True)
class SearchReport:
    query: str
    query_vector_dimension: int
    limit: int
    parent_weight: float
    child_weight: float
    collapse_strategy: str
    collapse_topk: int
    latency_ms: float
    parent: tuple[ParentHit, ...]
    child: tuple[ChildHit, ...]
    fusion: tuple[FusionHit, ...]
    path_matches: dict[str, dict[str, dict[str, object]]]
    ground_truth: dict[str, object] | None = None
    path_recall: dict[str, dict[str, object]] | None = None

    def _annotated_results(
        self,
        path_key: str,
        hits: tuple,
    ) -> list[dict[str, object]]:
        if not self.path_matches:
            return [hit.public_dict() for hit in hits]
        matches = self.path_matches.get(path_key, {})
        annotated: list[dict[str, object]] = []
        for hit in hits:
            result = hit.public_dict()
            flags = matches.get(str(hit.video.video_id), {})
            result["scene_match"] = flags.get("scene", False)
            result["object_match"] = flags.get("object", False)
            result["color_match"] = flags.get("color", False)
            result["actual_scene"] = flags.get("scene_value", "")
            result["actual_object"] = flags.get("object_value", "")
            result["actual_color"] = flags.get("color_value", "")
            annotated.append(result)
        return annotated

    def public_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "status": "passed",
            "query": self.query,
            "query_vector_dimension": self.query_vector_dimension,
            "limit": self.limit,
            "parent_weight": self.parent_weight,
            "child_weight": self.child_weight,
            "collapse_strategy": self.collapse_strategy,
            "collapse_topk": self.collapse_topk,
            "latency_ms": round(self.latency_ms, 2),
            "score_semantics": "COSINE similarity, not a probability",
            "paths": {
                "parent": {
                    "anns_field": "summary_vector",
                    "label": "Parent-only semantic search",
                    "results": self._annotated_results("parent", self.parent),
                },
                "child": {
                    "anns_field": "observations[description_vector]",
                    "group_by_field": "video_id",
                    "label": "Child-only semantic search, grouped by parent",
                    "results": self._annotated_results("child", self.child),
                },
                "fusion": {
                    "label": "Parent + child hybrid with collapse and weighted rerank",
                    "ranker": "WeightedRanker",
                    "parent_anns_field": "summary_vector",
                    "child_anns_field": "observations[description_vector]",
                    "results": self._annotated_results("fusion", self.fusion),
                },
            },
        }
        if self.ground_truth is not None:
            payload["ground_truth"] = self.ground_truth
        if self.path_recall is not None:
            payload["path_recall"] = self.path_recall
        return payload


class StructArrayHybridService:
    """Own the complete backend lifecycle within one process."""

    def __init__(
        self,
        *,
        config: RuntimeConfig | None = None,
        embedder: EmbedderLike | None = None,
        repository: RepositoryLike | None = None,
    ) -> None:
        self.config = config or RuntimeConfig.from_environment()
        self.embedder = embedder or BgeM3OnnxEmbedder(self.config)
        self.repository = repository or StructArrayHybridRepository()
        self._bundle: DatasetBundle | None = None
        self._videos_by_id: dict[str, object] | None = None
        self._last_prepare: dict[str, object] | None = None

    def _dataset(self) -> DatasetBundle:
        if self._bundle is None:
            self._bundle = build_dataset(self.config)
        return self._bundle

    def _videos_map(self):
        if self._videos_by_id is None:
            self._videos_by_id = {video.video_id: video for video in self._dataset().videos}
        return self._videos_by_id

    def evidence_frame_names(self) -> list[str]:
        """Every raw + annotated frame file name allowlisted by the prepared slice."""
        names: set[str] = set()
        for video in self._dataset().videos:
            for observation in video.observations:
                if observation.raw_frame_file:
                    names.add(observation.raw_frame_file)
                if observation.annotated_frame_file:
                    names.add(observation.annotated_frame_file)
        return sorted(names)

    def model_identity(self) -> dict[str, object]:
        return self.embedder.identity.public_dict()

    def status(self) -> dict[str, object]:
        repository_status = self.repository.status()
        dataset = self._dataset()
        return {
            "status": "ready" if repository_status.get("collection_exists") else "not_prepared",
            "sample_size": SAMPLE_SIZE,
            "model": {
                **self.model_identity(),
                "loaded": self.embedder.loaded,
                "cache_available": getattr(self.embedder, "cache_available", False),
            },
            "dataset": {
                "dataset_id": dataset.dataset_id,
                "dataset_version": dataset.dataset_version,
                "video_count": dataset.video_count,
                "observation_count": dataset.observation_count,
                "evidence_frame_count": dataset.evidence_frame_count,
                "manifest_sha256": dataset.manifest_sha256,
            },
            "milvus": repository_status,
            "query_presets": [
                {
                    "id": preset[0],
                    "text": preset[1],
                    "scene_terms": list(preset[2]),
                    "object_terms": list(preset[3]),
                    "color_terms": list(preset[4]) if len(preset) > 4 else [],
                }
                for preset in QUERY_PRESETS
            ],
        }

    def prepare(self) -> dict[str, object]:
        dataset = self._dataset()
        self.embedder.load()

        parent_texts = [video.video_summary for video in dataset.videos]
        child_texts = [
            observation.description
            for video in dataset.videos
            for observation in video.observations
        ]

        started = time.perf_counter()
        parent_vectors, _parent_report = self.embedder.embed(parent_texts)
        child_vectors, child_report = self.embedder.embed(child_texts)
        embed_seconds = time.perf_counter() - started

        report = self.repository.prepare(dataset, parent_vectors, child_vectors)
        self._videos_by_id = {video.video_id: video for video in dataset.videos}
        self._last_prepare = {
            "status": "prepared",
            "embedding": {
                "parent_vectors": len(parent_vectors),
                "child_vectors": len(child_vectors),
                "child_vector_dimension": child_report.vector_dimension,
                "embedding_seconds": round(embed_seconds, 3),
            },
            "milvus": report.public_dict(),
        }
        return self._last_prepare

    def _resolve_query(self, query: str) -> str:
        query = query.strip()
        if not query:
            raise SearchContractError("Query text must not be empty")
        if len(query) > 2000:
            raise SearchContractError("Query text is too long")
        return query

    def _intent_for_query(self, query: str) -> QueryIntent | None:
        for preset in QUERY_PRESETS:
            if preset[1] == query:
                return QueryIntent(
                    scene_terms=preset[2],
                    object_terms=preset[3],
                    color_terms=preset[4] if len(preset) > 4 else (),
                )
        return None

    def _match_flags(
        self,
        videos_by_id: dict[str, object],
        intent: QueryIntent,
    ) -> dict[str, dict[str, object]]:
        """Per-video scene/object/color booleans plus the ACTUAL detected values.

        The booleans drive ground truth and recall exactly as before; the
        ``*_value`` fields are the concrete tokens actually present in each
        video (road type, representative object type, representative color) so
        the UI can strike out words that contradict the query intent.
        """
        flags: dict[str, dict[str, object]] = {}
        scene_terms = [term.lower() for term in intent.scene_terms]
        object_types = {term.lower() for term in intent.object_terms}
        color_terms = {term.lower() for term in intent.color_terms}
        for video in videos_by_id.values():
            summary = getattr(video, "video_summary", "")
            environment = summary.split("||")[0].lower() if summary else ""
            scene = any(term in environment for term in scene_terms)

            object_matched = False
            color_matched = False
            object_type_counts: dict[str, int] = {}
            matched_colors: dict[str, int] = {}
            wrong_colors: dict[str, int] = {}
            for observation in getattr(video, "observations", ()):
                object_type = getattr(observation, "object_type", "").lower()
                color = getattr(observation, "color", "").lower()
                if object_type:
                    object_type_counts[object_type] = object_type_counts.get(object_type, 0) + 1
                if object_type in object_types:
                    object_matched = True
                    if not color_terms:
                        color_matched = True
                    elif color in color_terms:
                        color_matched = True
                    for word in _color_words(color):
                        if word in color_terms:
                            matched_colors[word] = matched_colors.get(word, 0) + 1
                        else:
                            wrong_colors[word] = wrong_colors.get(word, 0) + 1

            object_value = _representative_object(object_type_counts, object_types)
            color_value = _representative_color(
                matched_colors, wrong_colors, color_terms, color_matched
            )
            flags[getattr(video, "video_id", "")] = {
                "scene": scene,
                "object": object_matched,
                "color": color_matched,
                "scene_value": _detected_scene(environment),
                "object_value": object_value,
                "color_value": color_value,
            }
        return flags

    def search(
        self,
        *,
        query: str,
        limit: int = DEFAULT_TOP_K,
        parent_weight: float = 0.5,
        collapse_strategy: str = DEFAULT_COLLAPSE_STRATEGY,
    ) -> SearchReport:
        if not 1 <= limit <= MAX_TOP_K:
            raise SearchContractError(f"Search limit must be between 1 and {MAX_TOP_K}")
        if not WEIGHT_MIN <= parent_weight <= WEIGHT_MAX:
            raise SearchContractError(
                f"parent_weight must be between {WEIGHT_MIN} and {WEIGHT_MAX}"
            )
        child_weight = round(1.0 - parent_weight, 6)

        resolved_query = self._resolve_query(query)
        videos_by_id = self._videos_map()

        started = time.perf_counter()
        vectors, report = self.embedder.embed([resolved_query])
        query_vector = vectors[0]
        embed_ms = (time.perf_counter() - started) * 1000.0

        search_started = time.perf_counter()
        parent = self.repository.parent_search(
            query_vector=query_vector, videos_by_id=videos_by_id, limit=limit
        )
        child = self.repository.child_search(
            query_vector=query_vector, videos_by_id=videos_by_id, limit=limit
        )
        fusion = self.repository.fusion_search(
            query_vector=query_vector,
            videos_by_id=videos_by_id,
            limit=limit,
            parent_weight=parent_weight,
            child_weight=child_weight,
            collapse_strategy=collapse_strategy,
            collapse_topk=DEFAULT_COLLAPSE_TOPK,
        )
        total_ms = (time.perf_counter() - search_started) * 1000.0 + embed_ms

        intent = self._intent_for_query(resolved_query)
        path_matches: dict[str, dict[str, dict[str, object]]] = {}
        ground_truth: dict[str, object] | None = None
        path_recall: dict[str, dict[str, object]] | None = None
        if intent is not None:
            all_flags = self._match_flags(videos_by_id, intent)
            matched_ids = [
                video_id
                for video_id, flags in all_flags.items()
                if flags["scene"] and flags["object"] and flags["color"]
            ]
            ground_truth = {
                "scene_terms": list(intent.scene_terms),
                "object_terms": list(intent.object_terms),
                "color_terms": list(intent.color_terms),
                "matched_video_ids": sorted(matched_ids),
                "matched_count": len(matched_ids),
            }
            path_recall = {}
            for path_key, hits in (
                ("parent", parent),
                ("child", child),
                ("fusion", fusion),
            ):
                hit_ids = [hit.video.video_id for hit in hits]
                hits_found = [video_id for video_id in matched_ids if video_id in hit_ids]
                path_matches[path_key] = {
                    hit.video.video_id: all_flags.get(
                        hit.video.video_id, {"scene": False, "object": False, "color": False}
                    )
                    for hit in hits
                }
                path_recall[path_key] = {
                    "matched": len(hits_found),
                    "gt_count": len(matched_ids),
                    "recall": round(len(hits_found) / len(matched_ids), 4) if matched_ids else 0.0,
                    "ndcg": _ndcg(hit_ids, matched_ids),
                }

        return SearchReport(
            query=resolved_query,
            query_vector_dimension=report.vector_dimension,
            limit=limit,
            parent_weight=parent_weight,
            child_weight=child_weight,
            collapse_strategy=collapse_strategy,
            collapse_topk=DEFAULT_COLLAPSE_TOPK,
            latency_ms=total_ms,
            parent=parent,
            child=child,
            fusion=fusion,
            path_matches=path_matches,
            ground_truth=ground_truth,
            path_recall=path_recall,
        )

    def cleanup(self) -> dict[str, object]:
        milvus = self.repository.cleanup()
        self.embedder.unload()
        self._bundle = None
        self._videos_by_id = None
        self._last_prepare = None
        return {
            "status": "clean",
            "milvus": milvus,
            "model_loaded": self.embedder.loaded,
        }
