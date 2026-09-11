"""Milvus 3.0 StructArray provisioning and the three semantic search paths."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from pymilvus import AnnSearchRequest, DataType, MilvusClient, WeightedRanker

from structarray_hybrid_demo.config import (
    CHILD_ANNS_FIELD,
    CHILD_ARRAY_FIELD,
    CHILD_INDEX_NAME,
    CHILD_TEXT_FIELD,
    CHILD_VECTOR_FIELD,
    COLLECTION_NAME,
    DEFAULT_COLLAPSE_STRATEGY,
    DEFAULT_COLLAPSE_TOPK,
    GROUP_BY_FIELD,
    INDEX_BUILD_POLL_SECONDS,
    INDEX_BUILD_TIMEOUT_SECONDS,
    INDEX_EF_CONSTRUCTION,
    INDEX_M,
    INDEX_METRIC,
    INDEX_TYPE,
    MILVUS_EXPECTED_VERSION,
    MILVUS_TIMEOUT_SECONDS,
    MILVUS_URI,
    OBSERVATIONS_MAX_CAPACITY,
    PARENT_ANNS_FIELD,
    PARENT_INDEX_NAME,
    PARENT_VECTOR_FIELD,
    SEARCH_EF,
    VECTOR_DIMENSION,
)
from structarray_hybrid_demo.data import DatasetBundle, ObservationRecord, VideoRecord

EXPECTED_INDEX_NAMES = (PARENT_INDEX_NAME, CHILD_INDEX_NAME)


class RepositoryContractError(RuntimeError):
    """Raised when the real Milvus state violates the fixed contract."""


class SearchContractError(ValueError):
    """Raised when a search cannot be executed honestly against the prepared collection."""


@dataclass(frozen=True)
class ParentHit:
    rank: int
    score: float
    video: VideoRecord

    def public_dict(self) -> dict[str, object]:
        return {
            "rank": self.rank,
            "score": round(self.score, 6),
            "video_id": self.video.video_id,
            "video_summary": self.video.video_summary,
            "source_ordinal": self.video.source_ordinal,
            "preview": preview_dict(self.video),
        }


@dataclass(frozen=True)
class ChildHit:
    rank: int
    score: float
    video: VideoRecord
    offset: int
    observation: ObservationRecord

    def public_dict(self) -> dict[str, object]:
        return {
            "rank": self.rank,
            "score": round(self.score, 6),
            "video_id": self.video.video_id,
            "video_summary": self.video.video_summary,
            "source_ordinal": self.video.source_ordinal,
            "offset": self.offset,
            "observation": observation_dict(self.observation),
        }


def observation_dict(observation: ObservationRecord) -> dict[str, object]:
    return {
        "description": observation.description,
        "object_type": observation.object_type,
        "frame_id": observation.frame_id,
        "image_id": observation.image_id,
        "bbox": list(observation.bbox),
        "vehicle_type": observation.vehicle_type,
        "color": observation.color,
        "orientation": observation.orientation,
        "lights_on": observation.lights_on,
        "v_ego": round(observation.v_ego, 4),
        "a_ego": round(observation.a_ego, 4),
        "clip_id": observation.clip_id,
        "raw_frame": observation.raw_frame_file,
        "annotated_frame": observation.annotated_frame_file,
    }


def preview_dict(video: VideoRecord) -> dict[str, object] | None:
    """First evidence frame of a video, for parent/fusion cards."""
    for observation in video.observations:
        if observation.annotated_frame_file is not None:
            return {
                "frame_id": observation.frame_id,
                "annotated_frame": observation.annotated_frame_file,
                "raw_frame": observation.raw_frame_file,
            }
    return None


@dataclass(frozen=True)
class FusionHit:
    rank: int
    score: float
    video: VideoRecord
    child_observation: ObservationRecord | None = None

    def public_dict(self) -> dict[str, object]:
        return {
            "rank": self.rank,
            "score": round(self.score, 6),
            "video_id": self.video.video_id,
            "video_summary": self.video.video_summary,
            "source_ordinal": self.video.source_ordinal,
            "preview": preview_dict(self.video),
            "child_observation": (
                observation_dict(self.child_observation)
                if self.child_observation is not None
                else None
            ),
        }


@dataclass(frozen=True)
class ProvisioningReport:
    server_version: str
    collection_name: str
    parent_video_count: int
    element_count: int
    index_names: tuple[str, ...]

    def public_dict(self) -> dict[str, object]:
        return {
            "status": "prepared",
            "milvus_uri": MILVUS_URI,
            "server_version": self.server_version,
            "collection_name": self.collection_name,
            "parent_video_count": self.parent_video_count,
            "element_count": self.element_count,
            "index_names": list(self.index_names),
            "schema": {
                "parent_fields": ["video_id", "video_summary", "source_ordinal", "summary_vector"],
                "child_array": CHILD_ARRAY_FIELD,
                "child_vector_field": CHILD_VECTOR_FIELD,
                "child_text_field": CHILD_TEXT_FIELD,
                "max_capacity": OBSERVATIONS_MAX_CAPACITY,
            },
        }


class StructArrayHybridRepository:
    """Own the fixed collection: schema, indexes, prepare, and three search paths."""

    def __init__(self, client_factory=None) -> None:
        self._client_factory = client_factory or default_client_factory

    def _client(self):
        return self._client_factory()

    def _checked_version(self, client) -> str:
        version = client.get_server_version()
        if version != MILVUS_EXPECTED_VERSION:
            raise RepositoryContractError(
                f"Expected Milvus {MILVUS_EXPECTED_VERSION}, got {version}"
            )
        return version

    def status(self) -> dict[str, object]:
        client = self._client()
        try:
            version = self._checked_version(client)
            exists = client.has_collection(COLLECTION_NAME)
            names = [str(name) for name in client.list_collections()]
            stats = client.get_collection_stats(COLLECTION_NAME) if exists else None
            indexes = client.list_indexes(COLLECTION_NAME) if exists else []
        finally:
            client.close()
        return {
            "milvus_uri": MILVUS_URI,
            "server_version": version,
            "collection_name": COLLECTION_NAME,
            "collection_exists": exists,
            "raw_collection_names": names,
            "row_count": int(stats.get("row_count", 0)) if isinstance(stats, Mapping) else None,
            "index_names": [str(index) for index in indexes],
        }

    def _observation_schema(self, client) -> Any:
        observation = client.create_struct_field_schema()
        observation.add_field("description", DataType.VARCHAR, max_length=512)
        observation.add_field("description_vector", DataType.FLOAT_VECTOR, dim=VECTOR_DIMENSION)
        observation.add_field("object_type", DataType.VARCHAR, max_length=32)
        observation.add_field("frame_id", DataType.INT64)
        observation.add_field("image_id", DataType.VARCHAR, max_length=64)
        observation.add_field("bbox_x1", DataType.INT64)
        observation.add_field("bbox_y1", DataType.INT64)
        observation.add_field("bbox_x2", DataType.INT64)
        observation.add_field("bbox_y2", DataType.INT64)
        return observation

    def _schema(self, client) -> Any:
        schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("video_id", DataType.VARCHAR, max_length=64, is_primary=True)
        schema.add_field("video_summary", DataType.VARCHAR, max_length=4096)
        schema.add_field("source_ordinal", DataType.INT64)
        schema.add_field(PARENT_VECTOR_FIELD, DataType.FLOAT_VECTOR, dim=VECTOR_DIMENSION)
        schema.add_field(
            CHILD_ARRAY_FIELD,
            DataType.ARRAY,
            element_type=DataType.STRUCT,
            struct_schema=self._observation_schema(client),
            max_capacity=OBSERVATIONS_MAX_CAPACITY,
        )
        return schema

    def _index_params(self, client) -> Any:
        params = client.prepare_index_params()
        params.add_index(
            field_name=PARENT_VECTOR_FIELD,
            index_name=PARENT_INDEX_NAME,
            index_type=INDEX_TYPE,
            metric_type=INDEX_METRIC,
            params={"M": INDEX_M, "efConstruction": INDEX_EF_CONSTRUCTION},
        )
        params.add_index(
            field_name=CHILD_ANNS_FIELD,
            index_name=CHILD_INDEX_NAME,
            index_type=INDEX_TYPE,
            metric_type=INDEX_METRIC,
            params={"M": INDEX_M, "efConstruction": INDEX_EF_CONSTRUCTION},
        )
        return params

    def _wait_for_finished_indexes(self, client) -> tuple[str, ...]:
        deadline = time.monotonic() + INDEX_BUILD_TIMEOUT_SECONDS
        while True:
            raw_names = client.list_indexes(COLLECTION_NAME)
            index_names = tuple(sorted(str(name) for name in raw_names))
            if set(index_names) != set(EXPECTED_INDEX_NAMES):
                raise RepositoryContractError(
                    f"Milvus index names differ: expected={sorted(EXPECTED_INDEX_NAMES)}, "
                    f"actual={sorted(index_names)}"
                )
            descriptions = []
            for name in index_names:
                descriptions.append(client.describe_index(COLLECTION_NAME, name))
            states = {str(item.get("index_name")): str(item.get("state")) for item in descriptions}
            if all(str(item.get("state")) == "Finished" for item in descriptions):
                return index_names
            if time.monotonic() >= deadline:
                raise RepositoryContractError(
                    f"Milvus indexes did not finish within {INDEX_BUILD_TIMEOUT_SECONDS}s: {states}"
                )
            time.sleep(INDEX_BUILD_POLL_SECONDS)

    def prepare(
        self,
        bundle: DatasetBundle,
        parent_vectors: Sequence[Sequence[float]],
        child_vectors: Sequence[Sequence[float]],
    ) -> ProvisioningReport:
        if len(parent_vectors) != bundle.video_count:
            raise RepositoryContractError("Parent vector count does not match video count")
        if len(child_vectors) != bundle.observation_count:
            raise RepositoryContractError("Child vector count does not match observation count")

        client = self._client()
        try:
            version = self._checked_version(client)
            if client.has_collection(COLLECTION_NAME):
                raise RepositoryContractError("Collection already exists; refuse to replace it")
            client.create_collection(
                collection_name=COLLECTION_NAME,
                schema=self._schema(client),
                index_params=self._index_params(client),
                consistency_level="Strong",
            )

            rows: list[dict[str, Any]] = []
            child_cursor = 0
            for video in bundle.videos:
                elements: list[dict[str, Any]] = []
                for observation in video.observations:
                    elements.append(
                        {
                            "description": observation.description,
                            CHILD_VECTOR_FIELD: list(child_vectors[child_cursor]),
                            "object_type": observation.object_type,
                            "frame_id": observation.frame_id,
                            "image_id": observation.image_id,
                            "bbox_x1": observation.bbox[0],
                            "bbox_y1": observation.bbox[1],
                            "bbox_x2": observation.bbox[2],
                            "bbox_y2": observation.bbox[3],
                        }
                    )
                    child_cursor += 1
                rows.append(
                    {
                        "video_id": video.video_id,
                        "video_summary": video.video_summary,
                        "source_ordinal": video.source_ordinal,
                        PARENT_VECTOR_FIELD: list(parent_vectors[video.source_ordinal]),
                        CHILD_ARRAY_FIELD: elements,
                    }
                )
            client.insert(collection_name=COLLECTION_NAME, data=rows)
            client.flush(COLLECTION_NAME)
            client.load_collection(COLLECTION_NAME)

            stats = client.get_collection_stats(COLLECTION_NAME)
            row_count = int(stats.get("row_count", 0)) if isinstance(stats, Mapping) else 0
            if row_count != bundle.video_count:
                raise RepositoryContractError(
                    f"Unexpected parent row count: {row_count} != {bundle.video_count}"
                )
            index_names = self._wait_for_finished_indexes(client)
        finally:
            client.close()

        return ProvisioningReport(
            server_version=version,
            collection_name=COLLECTION_NAME,
            parent_video_count=bundle.video_count,
            element_count=bundle.observation_count,
            index_names=tuple(index_names),
        )

    def _hydrate_parent(
        self, hit: dict[str, Any], videos_by_id: Mapping[str, VideoRecord]
    ) -> VideoRecord:
        primary_key = hit.get(GROUP_BY_FIELD, hit.get("id"))
        if not isinstance(primary_key, str):
            raise RepositoryContractError("Parent hit is missing the video_id primary key")
        video = videos_by_id.get(primary_key)
        if video is None:
            raise RepositoryContractError(f"Unknown video_id in search hit: {primary_key}")
        return video

    def parent_search(
        self,
        *,
        query_vector: Sequence[float],
        videos_by_id: Mapping[str, VideoRecord],
        limit: int,
    ) -> tuple[ParentHit, ...]:
        client = self._client()
        try:
            self._checked_version(client)
            if not client.has_collection(COLLECTION_NAME):
                raise RepositoryContractError("Collection is not prepared")
            raw = client.search(
                collection_name=COLLECTION_NAME,
                data=[list(query_vector)],
                anns_field=PARENT_ANNS_FIELD,
                search_params={"metric_type": "COSINE", "params": {"ef": SEARCH_EF}},
                limit=limit,
                output_fields=[GROUP_BY_FIELD, "video_summary", "source_ordinal"],
            )
        finally:
            client.close()
        hits: list[ParentHit] = []
        for rank, hit in enumerate(raw[0], start=1):
            video = self._hydrate_parent(hit, videos_by_id)
            hits.append(ParentHit(rank=rank, score=float(hit["distance"]), video=video))
        return tuple(hits)

    def child_search(
        self,
        *,
        query_vector: Sequence[float],
        videos_by_id: Mapping[str, VideoRecord],
        limit: int,
        filter: str | None = None,
    ) -> tuple[ChildHit, ...]:
        client = self._client()
        try:
            self._checked_version(client)
            if not client.has_collection(COLLECTION_NAME):
                raise RepositoryContractError("Collection is not prepared")
            kwargs: dict[str, Any] = {
                "collection_name": COLLECTION_NAME,
                "data": [list(query_vector)],
                "anns_field": CHILD_ANNS_FIELD,
                "search_params": {"metric_type": "COSINE", "params": {"ef": SEARCH_EF}},
                "limit": limit,
                "group_by_field": GROUP_BY_FIELD,
                "output_fields": [GROUP_BY_FIELD, "video_summary", "source_ordinal"],
            }
            if filter is not None:
                kwargs["filter"] = filter
            raw = client.search(**kwargs)
        finally:
            client.close()
        hits: list[ChildHit] = []
        for rank, hit in enumerate(raw[0], start=1):
            video = self._hydrate_parent(hit, videos_by_id)
            offset = hit.get("offset")
            if isinstance(offset, bool) or not isinstance(offset, int):
                raise RepositoryContractError("Child hit is missing an integer offset")
            if not 0 <= offset < len(video.observations):
                raise RepositoryContractError(f"Child offset out of range: {offset}")
            observation = video.observations[offset]
            hits.append(
                ChildHit(
                    rank=rank,
                    score=float(hit["distance"]),
                    video=video,
                    offset=offset,
                    observation=observation,
                )
            )
        return tuple(hits)

    def _top_observations(
        self,
        *,
        query_vector: Sequence[float],
        videos_by_id: Mapping[str, VideoRecord],
        video_ids: Sequence[str],
    ) -> Mapping[str, ObservationRecord]:
        """Map each fused video_id to its strongest single child observation.

        This re-runs the same group-by child route restricted to the fused video
        set, so the fusion card's child text is exactly what the child column
        would have surfaced for that video (group-by top-1), rather than the
        internal ``topk_sum(3)`` aggregate used for fusion scoring.
        """
        if not video_ids:
            return {}
        quoted = ", ".join(json.dumps(video_id) for video_id in video_ids)
        hits = self.child_search(
            query_vector=query_vector,
            videos_by_id=videos_by_id,
            limit=len(video_ids),
            filter=f"{GROUP_BY_FIELD} in [{quoted}]",
        )
        return {hit.video.video_id: hit.observation for hit in hits}

    def fusion_search(
        self,
        *,
        query_vector: Sequence[float],
        videos_by_id: Mapping[str, VideoRecord],
        limit: int,
        parent_weight: float,
        child_weight: float,
        collapse_strategy: str = DEFAULT_COLLAPSE_STRATEGY,
        collapse_topk: int = DEFAULT_COLLAPSE_TOPK,
    ) -> tuple[FusionHit, ...]:
        client = self._client()
        try:
            self._checked_version(client)
            if not client.has_collection(COLLECTION_NAME):
                raise RepositoryContractError("Collection is not prepared")

            parent_req = AnnSearchRequest(
                data=[list(query_vector)],
                anns_field=PARENT_ANNS_FIELD,
                param={"metric_type": "COSINE", "params": {"ef": SEARCH_EF}},
                limit=limit,
            )
            child_req = AnnSearchRequest(
                data=[list(query_vector)],
                anns_field=CHILD_ANNS_FIELD,
                param={
                    "metric_type": "COSINE",
                    "params": {
                        "ef": SEARCH_EF,
                        "element_scope": {
                            "collapse": {"strategy": collapse_strategy, "topk": collapse_topk}
                        },
                    },
                },
                limit=max(limit * 4, 20),
            )
            raw = client.hybrid_search(
                collection_name=COLLECTION_NAME,
                reqs=[parent_req, child_req],
                ranker=WeightedRanker(parent_weight, child_weight),
                limit=limit,
                output_fields=[GROUP_BY_FIELD, "video_summary", "source_ordinal"],
            )
        finally:
            client.close()
        hits: list[FusionHit] = []
        fused_videos = [self._hydrate_parent(hit, videos_by_id) for hit in raw[0]]
        top_by_id = self._top_observations(
            query_vector=query_vector,
            videos_by_id=videos_by_id,
            video_ids=[video.video_id for video in fused_videos],
        )
        for rank, video in enumerate(fused_videos, start=1):
            distance = raw[0][rank - 1]["distance"]
            hits.append(
                FusionHit(
                    rank=rank,
                    score=float(distance),
                    video=video,
                    child_observation=top_by_id.get(video.video_id),
                )
            )
        return tuple(hits)

    def cleanup(self) -> dict[str, object]:
        client = self._client()
        try:
            version = self._checked_version(client)
            existed = client.has_collection(COLLECTION_NAME)
            indexes = (
                [str(index) for index in client.list_indexes(COLLECTION_NAME)] if existed else []
            )
            if existed:
                client.drop_collection(COLLECTION_NAME)
            absent = not client.has_collection(COLLECTION_NAME)
        finally:
            client.close()
        return {
            "status": "clean",
            "server_version": version,
            "collection_name": COLLECTION_NAME,
            "dropped": existed,
            "indexes_deleted_with_collection": list(indexes),
            "collection_absent_after": absent,
        }


def default_client_factory():
    return MilvusClient(uri=MILVUS_URI, timeout=MILVUS_TIMEOUT_SECONDS)
