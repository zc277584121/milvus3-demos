"""Deterministic query-token to local-image-patch explanation contracts."""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import torch


class ExplanationContractError(RuntimeError):
    """Raised when token or spatial metadata no longer matches model vectors."""


class TokenizerLike(Protocol):
    all_special_ids: list[int]

    def __call__(self, text: str, **kwargs: Any) -> Any: ...


STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "before",
        "both",
        "by",
        "can",
        "do",
        "for",
        "from",
        "how",
        "in",
        "into",
        "is",
        "it",
        "must",
        "of",
        "or",
        "should",
        "that",
        "the",
        "their",
        "them",
        "this",
        "to",
        "what",
        "when",
        "while",
        "with",
    }
)


@dataclass(frozen=True)
class QueryConcept:
    """One visible query word and the model token rows that represent it."""

    label: str
    token_indices: tuple[int, ...]


@dataclass(frozen=True)
class QueryToken:
    """One visible subword token row and its original text span."""

    index: int
    text: str
    char_start: int
    char_end: int
    is_stopword: bool


def query_concepts(
    *,
    query: str,
    tokenizer: TokenizerLike,
    processed_input_ids: torch.Tensor,
    query_vector_count: int,
) -> tuple[QueryConcept, ...]:
    """Map visible words to exact query-vector rows and exclude augmentation tokens."""

    encoded = tokenizer(
        query,
        add_special_tokens=False,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    token_ids = encoded["input_ids"][0].detach().cpu()
    offsets = encoded["offset_mapping"][0].detach().cpu().tolist()
    processed_ids = processed_input_ids.detach().cpu()
    if processed_ids.ndim != 1:
        raise ExplanationContractError("Processed query input IDs must be one-dimensional")
    if query_vector_count != int(processed_ids.shape[0]):
        raise ExplanationContractError("Query input IDs and query vectors differ in length")
    if token_ids.numel() == 0 or token_ids.numel() > processed_ids.numel():
        raise ExplanationContractError("Visible query tokens are missing from processed input")
    if not torch.equal(processed_ids[: token_ids.numel()], token_ids):
        raise ExplanationContractError("Visible query tokens do not prefix processed input IDs")

    special_ids = set(int(item) for item in tokenizer.all_special_ids)
    concepts: list[QueryConcept] = []
    for match in re.finditer(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*", query):
        label = match.group(0)
        normalized = label.casefold()
        if len(normalized) < 3 or normalized in STOP_WORDS:
            continue
        indices = tuple(
            index
            for index, (start, end) in enumerate(offsets)
            if end > match.start()
            and start < match.end()
            and int(token_ids[index]) not in special_ids
        )
        if indices:
            concepts.append(QueryConcept(label=label, token_indices=indices))
    if not concepts:
        raise ExplanationContractError("Query has no explainable visible concepts")
    return tuple(concepts)


def query_token_spans(
    *,
    query: str,
    tokenizer: TokenizerLike,
) -> tuple[QueryToken, ...]:
    """Return every visible subword token with its exact original text span.

    Unlike :func:`query_concepts`, which merges subword tokens into explainable
    words, this keeps the raw tokenizer segmentation so the frontend can render
    how the query was actually split. Each token carries its character span in
    the original query text (including any leading whitespace the tokenizer
    attached) and whether it belongs to a kept (non-stopword) concept.
    """

    encoded = tokenizer(
        query,
        add_special_tokens=False,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    token_ids = encoded["input_ids"][0].detach().cpu()
    offsets = encoded["offset_mapping"][0].detach().cpu().tolist()
    special_ids = set(int(item) for item in tokenizer.all_special_ids)

    kept_spans: list[tuple[int, int]] = []
    for match in re.finditer(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*", query):
        normalized = match.group(0).casefold()
        if len(normalized) < 3 or normalized in STOP_WORDS:
            continue
        kept_spans.append((match.start(), match.end()))

    def _belongs_to_kept(start: int, end: int) -> bool:
        return any(end > kept_start and start < kept_end for kept_start, kept_end in kept_spans)

    spans: list[QueryToken] = []
    for index, (start, end) in enumerate(offsets):
        if int(token_ids[index]) in special_ids:
            continue
        text = query[start:end]
        spans.append(
            QueryToken(
                index=index,
                text=text,
                char_start=int(start),
                char_end=int(end),
                is_stopword=not _belongs_to_kept(int(start), int(end)),
            )
        )
    return tuple(spans)


def spatial_sequence_indices(
    *,
    local_image_mask: torch.Tensor,
    grid_columns: int,
    grid_rows: int,
    tokens_per_subpatch_side: int,
) -> torch.Tensor:
    """Rearrange model sequence indices into the processor's normalized page grid."""

    if local_image_mask.ndim != 1:
        raise ExplanationContractError("Local image mask must be one-dimensional")
    if grid_columns <= 0 or grid_rows <= 0 or tokens_per_subpatch_side <= 0:
        raise ExplanationContractError("Explanation grid dimensions must be positive")
    if grid_columns % tokens_per_subpatch_side or grid_rows % tokens_per_subpatch_side:
        raise ExplanationContractError("Explanation grid must contain whole model subpatches")
    positions = local_image_mask.detach().cpu().nonzero(as_tuple=True)[0]
    expected = grid_columns * grid_rows
    if positions.numel() != expected:
        raise ExplanationContractError(
            f"Local image mask has {positions.numel()} tokens; expected {expected}"
        )
    subpatch_columns = grid_columns // tokens_per_subpatch_side
    subpatch_rows = grid_rows // tokens_per_subpatch_side
    return (
        positions.reshape(
            subpatch_rows,
            subpatch_columns,
            tokens_per_subpatch_side,
            tokens_per_subpatch_side,
        )
        .permute(0, 2, 1, 3)
        .reshape(grid_rows, grid_columns)
        .transpose(0, 1)
        .contiguous()
    )


def build_heatmap_contract(
    *,
    page_id: str,
    query_vector_count: int,
    page_vector_count: int,
    concepts: Sequence[QueryConcept],
    query_tokens: Sequence[QueryToken],
    similarity_map: torch.Tensor,
    sequence_indices: torch.Tensor,
    ignored_special_token_count: int,
) -> dict[str, object]:
    """Aggregate real cosine similarities into a complete normalized patch grid."""

    if similarity_map.ndim != 3:
        raise ExplanationContractError("Similarity map must have query, column, and row axes")
    _, grid_columns, grid_rows = similarity_map.shape
    if tuple(sequence_indices.shape) != (grid_columns, grid_rows):
        raise ExplanationContractError("Spatial sequence indices do not match similarity grid")
    if query_vector_count != int(similarity_map.shape[0]):
        raise ExplanationContractError("Similarity query axis differs from query vector count")

    token_records: list[dict[str, object]] = []
    token_scores: list[float] = []
    for token in query_tokens:
        if token.index >= query_vector_count:
            raise ExplanationContractError(f"Query token index is out of range: {token.text!r}")
        token_map = similarity_map[token.index]
        token_max = float(token_map.max())
        flat_peak = int(torch.argmax(token_map).item())
        peak_column = flat_peak // grid_rows
        peak_row = flat_peak % grid_rows
        token_scores.append(token_max)
        # Per-token intensity map for the hover heatmap. The spatial shape is
        # self-normalized (so a token's *where* stays visible), then scaled by
        # the token's max similarity (its *how much*) so a low-scoring or
        # stopword token renders faint overall instead of peaking at pure red.
        token_min = float(token_map.min())
        token_range = token_max - token_min
        token_intensities = [
            round(
                float(
                    ((token_map[column, row] - token_min) / token_range) * token_max
                    if token_range > 0
                    else 0.0
                ),
                6,
            )
            for row in range(grid_rows)
            for column in range(grid_columns)
        ]
        token_records.append(
            {
                "index": token.index,
                "text": token.text,
                "char_start": token.char_start,
                "char_end": token.char_end,
                "is_stopword": token.is_stopword,
                "max_similarity": round(token_max, 6),
                "peak_patch_index": peak_row * grid_columns + peak_column,
                "intensities": token_intensities,
            }
        )

    concept_records: list[dict[str, object]] = []
    concept_maps: list[torch.Tensor] = []
    for concept in concepts:
        if not concept.token_indices or max(concept.token_indices) >= query_vector_count:
            raise ExplanationContractError(f"Concept token index is out of range: {concept.label}")
        concept_map = similarity_map[list(concept.token_indices)].amax(dim=0)
        concept_maps.append(concept_map)
        flat_peak = int(torch.argmax(concept_map).item())
        peak_column = flat_peak // grid_rows
        peak_row = flat_peak % grid_rows
        concept_records.append(
            {
                "label": concept.label,
                "query_token_indices": list(concept.token_indices),
                "peak_similarity": round(float(concept_map.max()), 6),
                "peak_patch_index": peak_row * grid_columns + peak_column,
            }
        )
    if not concept_maps:
        raise ExplanationContractError("At least one query concept is required")

    aggregate = torch.stack(concept_maps).amax(dim=0).detach().cpu().to(torch.float32)
    flat = aggregate.numpy().reshape(-1)
    low = float(np.quantile(flat, 0.5))
    high = float(np.quantile(flat, 0.98))
    if not math.isfinite(low) or not math.isfinite(high) or high <= low:
        raise ExplanationContractError("Similarity map cannot be normalized deterministically")
    normalized = ((aggregate - low) / (high - low)).clamp(0.0, 1.0)

    patches: list[dict[str, object]] = []
    for row in range(grid_rows):
        for column in range(grid_columns):
            patches.append(
                {
                    "patch_index": row * grid_columns + column,
                    "model_sequence_index": int(sequence_indices[column, row]),
                    "grid_column": column,
                    "grid_row": row,
                    "x": round(column / grid_columns, 6),
                    "y": round(row / grid_rows, 6),
                    "width": round(1 / grid_columns, 6),
                    "height": round(1 / grid_rows, 6),
                    "raw_similarity": round(float(aggregate[column, row]), 6),
                    "intensity": round(float(normalized[column, row]), 6),
                }
            )

    concept_records.sort(key=lambda item: float(item["peak_similarity"]), reverse=True)
    scored = [score for score in token_scores if score > 0]
    token_score_min = round(min(scored), 6) if scored else 0.0
    token_score_max = round(max(scored), 6) if scored else 0.0
    return {
        "page_id": page_id,
        "source": "local_colsmol_query_page_multi_vector",
        "rank_source": "milvus_page_level_max_sim_cosine",
        "query_vector_count": query_vector_count,
        "page_vector_count": page_vector_count,
        "explained_query_token_count": len(
            {index for concept in concepts for index in concept.token_indices}
        ),
        "ignored_special_token_count": ignored_special_token_count,
        "grid": {
            "columns": grid_columns,
            "rows": grid_rows,
            "patch_count": grid_columns * grid_rows,
            "coordinate_space": "normalized_unmodified_page",
        },
        "aggregation": {
            "similarity": "cosine_dot_product_on_l2_normalized_vectors",
            "concept": "maximum_over_subword_query_tokens",
            "patch": "maximum_over_visible_non_stopword_query_concepts",
            "normalization": "clamp((value-page_median)/(page_p98-page_median),0,1)",
            "spatial_mapping": (
                "ColPali Engine local image-token mask with global image tokens excluded; "
                "Idefics3 subpatch order rearranged to page coordinates"
            ),
        },
        "normalization_bounds": {"median": round(low, 6), "p98": round(high, 6)},
        "token_score_bounds": {"min": token_score_min, "max": token_score_max},
        "concepts": concept_records,
        "tokens": token_records,
        "patches": patches,
    }
