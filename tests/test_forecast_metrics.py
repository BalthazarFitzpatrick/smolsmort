"""registry metadata and recipe scores do not require parquet support"""

import json

from smolsmort.forecast.metrics import error_metrics, metric_registry, read_test_metrics


def test_regression_registry_defines_one_consistent_metric_set():
    registry = metric_registry({"task": "regression", "mode": "timeseries"})
    assert [item["id"] for item in registry] == [
        "validation_error",
        "wape",
        "mae",
        "mape",
        "test_ratio",
    ]
    required = {"id", "label", "explanation", "direction", "unit", "scope", "computation"}
    assert all(required <= item.keys() and item["direction"] == "lower" for item in registry)
    assert "validation windows" in registry[0]["explanation"]
    assert "actual > 0" in registry[3]["explanation"]
    assert "excluded and counted" in registry[3]["explanation"]
    registry[0]["label"] = "changed"
    assert metric_registry({})[0]["label"] == "validation error"


def test_other_tasks_do_not_offer_regression_test_metrics():
    classification = metric_registry({"task": "classification"})
    survival = metric_registry({"task": "regression", "censor": {"unit": "day"}})
    for registry in (classification, survival):
        assert [item["id"] for item in registry] == ["validation_error", "test_ratio"]
        assert registry[0]["unit"] == "number"
    assert "log loss" in classification[0]["explanation"]
    assert "hazard uses log loss" in survival[0]["explanation"]
    assert "integrated brier" in survival[0]["computation"]


def test_candidates_keep_saved_validation_metrics_and_never_get_test_scores():
    values = error_metrics({}, fitness=0.15, candidate={"wape": 0.2, "mae": 4, "mape": 0.3})
    assert values["validation_error"]["value"] == 0.15
    assert values["wape"]["value"] == 0.2 and values["mae"]["value"] == 4
    for name in ("wape", "mae"):
        assert values[name]["scope"] == "validation"
        assert "saved validation metric" in values[name]["note"]
    for name in ("mape", "test_ratio"):
        assert values[name]["value"] is None and "candidate" in values[name]["reason"]


def test_unavailable_and_nonfinite_values_have_reasons_and_valid_json(tmp_path):
    values = error_metrics({}, fitness=float("nan"), verdict={"accuracy": {"ratio": float("inf")}})
    assert all(value["value"] is None and value["reason"] for value in values.values())
    assert error_metrics({}, fitness=True)["validation_error"]["value"] is None
    assert all(
        value["value"] is None and value["reason"]
        for value in read_test_metrics(tmp_path / "missing.parquet").values()
    )
    json.dumps(values, allow_nan=False)
