from __future__ import annotations

import torch

from embedding_list_demo.explanation import (
    QueryConcept,
    QueryToken,
    build_heatmap_contract,
    query_concepts,
    query_token_spans,
    spatial_sequence_indices,
)


class FakeTokenizer:
    all_special_ids = [99]

    def __call__(self, text: str, **_: object) -> dict[str, torch.Tensor]:
        assert text == "How risk analysis works"
        return {
            "input_ids": torch.tensor([[10, 11, 12, 13]]),
            "offset_mapping": torch.tensor([[[0, 3], [4, 8], [9, 17], [18, 23]]]),
        }


def test_visible_query_concepts_map_to_real_vector_rows() -> None:
    concepts = query_concepts(
        query="How risk analysis works",
        tokenizer=FakeTokenizer(),
        processed_input_ids=torch.tensor([10, 11, 12, 13, 99, 99]),
        query_vector_count=6,
    )

    assert concepts == (
        QueryConcept("risk", (1,)),
        QueryConcept("analysis", (2,)),
        QueryConcept("works", (3,)),
    )


def test_query_token_spans_keep_subwords_and_mark_stopwords() -> None:
    tokens = query_token_spans(query="How risk analysis works", tokenizer=FakeTokenizer())

    assert tokens == (
        QueryToken(index=0, text="How", char_start=0, char_end=3, is_stopword=True),
        QueryToken(index=1, text="risk", char_start=4, char_end=8, is_stopword=False),
        QueryToken(index=2, text="analysis", char_start=9, char_end=17, is_stopword=False),
        QueryToken(index=3, text="works", char_start=18, char_end=23, is_stopword=False),
    )


def test_spatial_indices_and_heatmap_are_complete_nonuniform_and_deterministic() -> None:
    mask = torch.ones(8, dtype=torch.bool)
    indices = spatial_sequence_indices(
        local_image_mask=mask,
        grid_columns=4,
        grid_rows=2,
        tokens_per_subpatch_side=2,
    )
    assert indices.tolist() == [[0, 2], [1, 3], [4, 6], [5, 7]]

    similarities = torch.tensor(
        [
            [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]],
            [[0.8, 0.7], [0.6, 0.5], [0.4, 0.3], [0.2, 0.1]],
            [[0.0, 0.1], [0.2, 0.3], [0.4, 0.5], [0.6, 0.9]],
        ],
        dtype=torch.float32,
    )
    kwargs = {
        "page_id": "page-1",
        "query_vector_count": 3,
        "page_vector_count": 8,
        "concepts": (QueryConcept("risk", (0, 1)), QueryConcept("analysis", (2,))),
        "query_tokens": (
            QueryToken(index=0, text="risk", char_start=0, char_end=4, is_stopword=False),
            QueryToken(index=1, text="analysis", char_start=5, char_end=13, is_stopword=False),
            QueryToken(index=2, text="of", char_start=14, char_end=16, is_stopword=True),
        ),
        "similarity_map": similarities,
        "sequence_indices": indices,
        "ignored_special_token_count": 10,
    }
    first = build_heatmap_contract(**kwargs)
    second = build_heatmap_contract(**kwargs)

    assert first == second
    assert first["source"] == "local_colsmol_query_page_multi_vector"
    assert first["grid"] == {
        "columns": 4,
        "rows": 2,
        "patch_count": 8,
        "coordinate_space": "normalized_unmodified_page",
    }
    tokens = first["tokens"]
    assert isinstance(tokens, list)
    assert len(tokens) == 3
    assert tokens[0]["text"] == "risk"
    assert tokens[0]["max_similarity"] == 0.8
    assert tokens[0]["peak_patch_index"] == 7
    assert len(tokens[0]["intensities"]) == 8
    # Per-token intensities are self-normalized then scaled by the token's max
    # similarity, so the peak patch equals the token score (0.8), not 1.0.
    assert min(tokens[0]["intensities"]) == 0.0
    assert max(tokens[0]["intensities"]) == 0.8
    assert tokens[0]["intensities"][7] == 0.8
    assert tokens[1]["text"] == "analysis"
    assert tokens[1]["peak_patch_index"] == 0
    assert len(tokens[1]["intensities"]) == 8
    assert max(tokens[1]["intensities"]) == 0.8
    assert tokens[2]["text"] == "of"
    assert tokens[2]["is_stopword"] is True
    assert first["token_score_bounds"] == {"min": 0.8, "max": 0.9}
    patches = first["patches"]
    assert isinstance(patches, list)
    assert len(patches) == 8
    assert {item["model_sequence_index"] for item in patches} == set(range(8))
    assert min(item["intensity"] for item in patches) == 0.0
    assert max(item["intensity"] for item in patches) == 1.0
    assert all(0 <= item["x"] < 1 and 0 <= item["y"] < 1 for item in patches)
