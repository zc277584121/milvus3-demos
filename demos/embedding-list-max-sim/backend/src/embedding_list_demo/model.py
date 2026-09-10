"""CPU-only ColSmol inference and local late-interaction scoring."""

from __future__ import annotations

import gc
import importlib.metadata
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from colpali_engine.models import ColIdefics3, ColIdefics3Processor
from PIL import Image

from embedding_list_demo.config import (
    MAX_PATCHES_PER_PAGE,
    MODEL_ADAPTER_REVISION,
    MODEL_BASE_ID,
    MODEL_BASE_REVISION,
    MODEL_DTYPE,
    MODEL_GIT_HASH,
    MODEL_ID,
    VECTOR_DIMENSION,
    RuntimeConfig,
)
from embedding_list_demo.explanation import (
    ExplanationContractError,
    build_heatmap_contract,
    query_concepts,
    query_token_spans,
    spatial_sequence_indices,
)

if TYPE_CHECKING:
    from embedding_list_demo.manual import ManualManifest


class ModelContractError(RuntimeError):
    """Raised when the installed model or CPU runtime violates the fixed contract."""


@dataclass(frozen=True)
class SnapshotRecord:
    model_id: str
    revision: str
    git_hash: str
    snapshot_path: str

    def public_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class ModelIdentity:
    model_id: str
    adapter: SnapshotRecord
    base: SnapshotRecord
    inference_dtype: str
    inference_device: str
    vector_dimension: int
    colpali_engine_version: str
    torch_version: str
    transformers_version: str

    def public_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "adapter": self.adapter.public_dict(),
            "base": self.base.public_dict(),
            "inference_dtype": self.inference_dtype,
            "inference_device": self.inference_device,
            "vector_dimension": self.vector_dimension,
            "packages": {
                "colpali-engine": self.colpali_engine_version,
                "torch": self.torch_version,
                "transformers": self.transformers_version,
            },
        }


@dataclass(frozen=True)
class EmbeddingBatchReport:
    vector_counts: tuple[int, ...]
    vector_dimension: int
    norm_min: float
    norm_max: float
    inference_device: str
    inference_dtype: str

    def public_dict(self) -> dict[str, object]:
        return {
            "vector_counts": list(self.vector_counts),
            "vector_dimension": self.vector_dimension,
            "norm_min": self.norm_min,
            "norm_max": self.norm_max,
            "inference_device": self.inference_device,
            "inference_dtype": self.inference_dtype,
        }


def _hub_directory_name(model_id: str) -> str:
    return "models--" + model_id.replace("/", "--")


def _snapshot_record(
    *,
    hub_cache: Path,
    model_id: str,
    expected_revision: str,
) -> SnapshotRecord:
    model_root = hub_cache / _hub_directory_name(model_id)
    snapshot = model_root / "snapshots" / expected_revision
    if not snapshot.is_dir():
        raise ModelContractError(f"Missing local model snapshot: {snapshot}")
    git_hash_path = snapshot / "git_hash.txt"
    if not git_hash_path.is_file():
        raise ModelContractError(f"Missing model git_hash.txt: {git_hash_path}")
    git_hash = git_hash_path.read_text(encoding="utf-8").strip()
    if git_hash != MODEL_GIT_HASH:
        raise ModelContractError(
            f"Model git hash differs for {model_id}: expected={MODEL_GIT_HASH}, actual={git_hash}"
        )
    return SnapshotRecord(
        model_id=model_id,
        revision=expected_revision,
        git_hash=git_hash,
        snapshot_path=str(snapshot.resolve()),
    )


def resolve_model_identity(config: RuntimeConfig) -> ModelIdentity:
    """Verify both offline snapshots and return their immutable identity."""

    adapter = _snapshot_record(
        hub_cache=config.hf_hub_cache,
        model_id=MODEL_ID,
        expected_revision=MODEL_ADAPTER_REVISION,
    )
    base = _snapshot_record(
        hub_cache=config.hf_hub_cache,
        model_id=MODEL_BASE_ID,
        expected_revision=MODEL_BASE_REVISION,
    )
    return ModelIdentity(
        model_id=MODEL_ID,
        adapter=adapter,
        base=base,
        inference_dtype=MODEL_DTYPE,
        inference_device=config.device,
        vector_dimension=VECTOR_DIMENSION,
        colpali_engine_version=importlib.metadata.version("colpali-engine"),
        torch_version=importlib.metadata.version("torch"),
        transformers_version=importlib.metadata.version("transformers"),
    )


def nonzero_vectors(embeddings: torch.Tensor) -> torch.Tensor:
    """Return finite nonzero token vectors as contiguous float32 CPU tensors."""

    if embeddings.ndim != 2 or embeddings.shape[1] != VECTOR_DIMENSION:
        raise ModelContractError(
            f"Unexpected embedding shape: expected=(*,{VECTOR_DIMENSION}), "
            f"actual={tuple(embeddings.shape)}"
        )
    vectors = embeddings.detach().to(device="cpu", dtype=torch.float32)
    finite = torch.isfinite(vectors).all(dim=1)
    nonzero = torch.linalg.vector_norm(vectors, dim=1) > 1e-6
    vectors = vectors[finite & nonzero].contiguous()
    if vectors.shape[0] == 0:
        raise ModelContractError("Model returned no finite nonzero vectors")
    if vectors.shape[0] > MAX_PATCHES_PER_PAGE:
        raise ModelContractError(f"Model vector count exceeds Milvus capacity: {vectors.shape[0]}")
    return vectors


def embedding_report(vectors: list[torch.Tensor]) -> EmbeddingBatchReport:
    """Summarize vector counts, dimension, norms, device, and dtype."""

    if not vectors:
        raise ModelContractError("Embedding report requires at least one tensor")
    norms = torch.cat([torch.linalg.vector_norm(item, dim=1) for item in vectors])
    return EmbeddingBatchReport(
        vector_counts=tuple(int(item.shape[0]) for item in vectors),
        vector_dimension=int(vectors[0].shape[1]),
        norm_min=round(float(norms.min()), 8),
        norm_max=round(float(norms.max()), 8),
        inference_device="cpu",
        inference_dtype=MODEL_DTYPE,
    )


class ColSmolModel:
    """Lazily own one CPU FP32 model and processor for a backend process."""

    def __init__(self, config: RuntimeConfig) -> None:
        self.config = config
        self._identity: ModelIdentity | None = None
        self._model: ColIdefics3 | None = None
        self._processor: ColIdefics3Processor | None = None

    @property
    def identity(self) -> ModelIdentity:
        """Resolve and memoize the exact offline model identity."""

        if self._identity is None:
            self._identity = resolve_model_identity(self.config)
        return self._identity

    @property
    def loaded(self) -> bool:
        return self._model is not None and self._processor is not None

    def _load(self) -> tuple[ColIdefics3, ColIdefics3Processor]:
        if self.loaded:
            assert self._model is not None and self._processor is not None
            return self._model, self._processor
        identity = self.identity
        if self.config.device != "cpu":
            raise ModelContractError("ColSmol runtime must use the explicit cpu device")
        model = ColIdefics3.from_pretrained(
            identity.adapter.snapshot_path,
            cache_dir=str(self.config.hf_hub_cache),
            local_files_only=True,
            force_download=False,
            torch_dtype=torch.float32,
            device_map="cpu",
        ).eval()
        processor = ColIdefics3Processor.from_pretrained(
            identity.adapter.snapshot_path,
            cache_dir=str(self.config.hf_hub_cache),
            local_files_only=True,
            force_download=False,
        )
        self._model = model
        self._processor = processor
        return model, processor

    def runtime_dict(self) -> dict[str, object]:
        """Return model identity plus the actual CPU-only execution facts."""

        value = self.identity.public_dict()
        value.update(
            {
                "loaded": self.loaded,
                "configured_device": self.config.device,
                "device_type": "cpu",
                "cpu_only": True,
                "torch_runtime_dtype": str(torch.float32),
            }
        )
        return value

    def embed_pages(
        self,
        *,
        dataset_root: Path,
        manifest: ManualManifest,
    ) -> tuple[list[torch.Tensor], EmbeddingBatchReport]:
        """Run real sequential CPU FP32 inference for every rendered page."""

        model, processor = self._load()
        vectors: list[torch.Tensor] = []
        for page in manifest.pages:
            with Image.open(dataset_root / page.image_filename) as source:
                image = source.convert("RGB")
                batch = processor.process_images([image]).to(model.device)
                with torch.inference_mode():
                    raw_embeddings = model(**batch)
                vectors.append(nonzero_vectors(raw_embeddings[0]))
                image.close()
        return vectors, embedding_report(vectors)

    def embed_query(self, query: str) -> tuple[torch.Tensor, EmbeddingBatchReport]:
        """Generate one real-time CPU FP32 query-token multi-vector."""

        if not query.strip():
            raise ModelContractError("Query text must not be empty")
        model, processor = self._load()
        batch = processor.process_queries([query]).to(model.device)
        with torch.inference_mode():
            raw_embeddings = model(**batch)
        vectors = nonzero_vectors(raw_embeddings[0])
        return vectors, embedding_report([vectors])

    def score_pages(
        self,
        query_vectors: torch.Tensor,
        page_vectors: list[torch.Tensor],
    ) -> np.ndarray:
        """Run the genuine ColPali local MaxSim scorer for comparison only."""

        _, processor = self._load()
        scores = processor.score_multi_vector(
            [query_vectors],
            page_vectors,
            device="cpu",
        )
        return scores[0].detach().cpu().numpy().astype(np.float64, copy=False)

    def explain_page(
        self,
        *,
        query: str,
        query_vectors: torch.Tensor,
        page_vectors: torch.Tensor,
        page_image: Path,
        page_id: str,
    ) -> dict[str, object]:
        """Build an application-side spatial explanation from real model vectors."""

        model, processor = self._load()
        query_batch = processor.process_queries([query]).to(model.device)
        query_input_ids = query_batch.input_ids[0]
        concepts = query_concepts(
            query=query,
            tokenizer=processor.tokenizer,
            processed_input_ids=query_input_ids,
            query_vector_count=int(query_vectors.shape[0]),
        )
        query_tokens = query_token_spans(
            query=query,
            tokenizer=processor.tokenizer,
        )
        visible_query = processor.tokenizer(
            query,
            add_special_tokens=False,
            return_tensors="pt",
        )
        visible_query_token_count = int(visible_query["input_ids"].shape[1])

        with Image.open(page_image) as source:
            image = source.convert("RGB")
            page_batch = processor.process_images([image])
            row_col = processor.image_processor(
                images=[image],
                return_row_col_info=True,
                return_tensors="pt",
            )
            image.close()
        if int(page_batch.input_ids.shape[1]) != int(page_vectors.shape[0]):
            raise ExplanationContractError(
                "Processed page input IDs and cached page vectors differ in length"
            )
        split_rows = int(row_col["rows"][0][0])
        split_columns = int(row_col["cols"][0][0])
        if split_rows <= 0 or split_columns <= 0:
            raise ExplanationContractError("NASA page must use local Idefics3 image splits")
        tokens_per_subpatch_side = math.isqrt(int(processor.image_seq_len))
        if tokens_per_subpatch_side**2 != int(processor.image_seq_len):
            raise ExplanationContractError("Idefics3 image token count is not a square")
        grid_columns = split_columns * tokens_per_subpatch_side
        grid_rows = split_rows * tokens_per_subpatch_side
        local_mask = processor.get_local_image_mask(page_batch)[0].detach().cpu()
        sequence_indices = spatial_sequence_indices(
            local_image_mask=local_mask,
            grid_columns=grid_columns,
            grid_rows=grid_rows,
            tokens_per_subpatch_side=tokens_per_subpatch_side,
        )
        similarity_map = processor.get_similarity_maps_from_embeddings(
            image_embeddings=page_vectors.unsqueeze(0),
            query_embeddings=query_vectors.unsqueeze(0),
            n_patches=(grid_columns, grid_rows),
            image_mask=local_mask.unsqueeze(0),
        )[0]
        return build_heatmap_contract(
            page_id=page_id,
            query_vector_count=int(query_vectors.shape[0]),
            page_vector_count=int(page_vectors.shape[0]),
            concepts=concepts,
            query_tokens=query_tokens,
            similarity_map=similarity_map,
            sequence_indices=sequence_indices,
            ignored_special_token_count=(int(query_vectors.shape[0]) - visible_query_token_count),
        )

    def unload(self) -> None:
        """Release this process's model references."""

        self._model = None
        self._processor = None
        gc.collect()
