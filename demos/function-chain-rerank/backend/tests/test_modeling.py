from function_chain_demo.business_profiles import ITEM_PROFILE_NAMES, operational_profile
from function_chain_demo.catalog import DATASET, FEATURE_NAMES
from function_chain_demo.dataset import SIMULATED_FIELD
from function_chain_demo.modeling import generate_training_data, load_model, train_model


def test_grouped_training_data_and_ubj_model_are_byte_reproducible(tmp_path) -> None:
    first = generate_training_data()
    second = generate_training_data()

    assert first.train_features.tolist() == second.train_features.tolist()
    assert first.train_labels.tolist() == second.train_labels.tolist()
    assert first.validation_features.tolist() == second.validation_features.tolist()
    assert first.validation_labels.tolist() == second.validation_labels.tolist()
    assert first.train_features.shape == (960, 5)
    assert first.validation_features.shape == (480, 5)
    assert len(first.train_query_ids) == 4
    assert len(first.validation_query_ids) == 2
    assert set(first.train_query_ids).isdisjoint(first.validation_query_ids)

    artifact = train_model(tmp_path / "reranker.ubj")
    second_artifact = train_model(tmp_path / "reranker-second.ubj")

    assert artifact.path.is_file()
    assert artifact.byte_size > 0
    assert len(artifact.sha256) == 64
    assert artifact.feature_names == FEATURE_NAMES
    assert artifact.model_version == "synthetic-commerce-catalog-r1-reranker-v1"
    assert artifact.dataset_revision == "synthetic-commerce-catalog-r1"
    assert artifact.training_record_count == 1440
    assert artifact.sha256 == second_artifact.sha256
    assert artifact.path.read_bytes() == second_artifact.path.read_bytes()
    assert artifact.metrics == second_artifact.metrics
    assert artifact.metrics["validation_ndcg_at_10"] > 0.98
    assert artifact.metrics["all_query_relevant_top1_rate"] == 1.0
    assert artifact.metrics["validation_relevant_top1_rate"] == 1.0
    assert artifact.metrics["all_query_changed_order_rate"] >= 5 / 6
    assert artifact.metrics["all_query_relevant_candidate_changed_order_rate"] >= round(5 / 6, 8)
    assert artifact.parameters["monotone_constraints"] == "(1,1,1,1,1)"
    assert artifact.rounds == 128
    assert artifact.parameters["max_depth"] == 2
    assert artifact.direction_gate == second_artifact.direction_gate
    assert all(result["passed"] for result in artifact.direction_gate.values())
    assert artifact.business_gate == second_artifact.business_gate
    assert artifact.business_gate["passed"] is True
    assert artifact.business_gate["semantic_jitter"] == 1e-6
    assert all(report["passed"] for report in artifact.business_gate["queries"])
    assert all(
        report["type_recall_held"] and report["type_recall_floor"]
        for report in artifact.business_gate["queries"]
    )
    assert all(report["healthy_upward_ids"] for report in artifact.business_gate["queries"])
    assert all(report["repeat_order_stable"] for report in artifact.business_gate["queries"])
    assert load_model(artifact.path).num_boosted_rounds() == artifact.rounds

    assert SIMULATED_FIELD == "deterministic_simulated_operational_signal"


def test_operational_profiles_are_stable_item_level_inputs_only() -> None:
    known_ids = {product.item_id for product in DATASET.products}

    assert set(ITEM_PROFILE_NAMES) <= known_ids
    assert len(ITEM_PROFILE_NAMES) == 36
    assert all(operational_profile(item_id) is not None for item_id in ITEM_PROFILE_NAMES)
    assert all(profile in {"H", "G", "S", "N", "R"} for profile in ITEM_PROFILE_NAMES.values())

    feature_rows_by_item: dict[str, set[tuple[float, float, float, float]]] = {}
    for record in DATASET.training_records:
        item_id = DATASET.product_by_id(record.product_id).item_id
        feature_rows_by_item.setdefault(item_id, set()).add(record.features[1:])
    assert all(len(feature_rows_by_item[item_id]) == 1 for item_id in ITEM_PROFILE_NAMES)
