"""hand-computed errors preserve group, pooled and weekly total meanings"""

import csv
import io

import pytest
from test_forecast_export_dialog import input_run as input_run
from test_forecast_export_dialog import saved_run as saved_run

from smolsmort.forecast.export_bundle import export_bundle
from smolsmort.forecast.export_metrics import MEASURES, metric_columns, score_rows, table_rows
from smolsmort.forecast.export_wide import calibrated_totals, normalize_columns


@pytest.mark.parametrize("dims", [["group"], ["group", "channel"]])
def test_metrics_hand_computed(dims):
    request = {
        "spec": {
            "columns": [
                {"name": "date", "role": "time"},
                *[{"name": name, "role": "dimension"} for name in dims],
            ]
        }
    }
    raw = {
        "step": ["2026-01-12"] * 2 + ["2026-01-19"] * 2,
        "origin": ["2026-01-05"] * 4,
        "weeks_ahead": [1, 1, 2, 2],
        "group": ["a", "b", "a", "b"],
        **({"channel": ["web"] * 4} if len(dims) == 2 else {}),
        "actual": [10, 20, 10, 20],
        "prediction": [8, 24, 12, 18],
        "naive_prediction": [5, 15, 5, 15],
        "lower_50": [8, 20, 11, 17],
        "upper_50": [10, 25, 13, 19],
        "lower": [7, 19, 9, 17],
        "upper": [11, 26, 14, 21],
    }
    tables = dict.fromkeys(("validation", "test", "forecast"), raw)
    output = metric_columns(
        {phase: normalize_columns(request, table) for phase, table in tables.items()},
        dims,
        "date",
        calibrated_totals(tables),
    )
    assert list(output) == ["dataset_slice", *dims, "horizon_step", *MEASURES]
    rows = table_rows(output)
    assert len(rows) == 2 * (2 + 2) * (2 + 1)
    for phase in ("validation", "test"):
        selected = {
            (row["group"], row["horizon_step"]): row
            for row in rows
            if row["dataset_slice"] == phase
        }
        a = selected["a", "all"]
        assert [
            a[name]
            for name in (
                "wape",
                "naive_wape",
                "ratio",
                "mae",
                "bias",
                "actual_sum",
                "prediction_sum",
                "rows",
                "coverage_50",
                "coverage_80",
            )
        ] == pytest.approx([0.2, 0.5, 0.4, 2, 0, 20, 20, 2, 0.5, 1])
        pooled = selected["all groups pooled", "all"]
        assert [
            pooled[name]
            for name in (
                "wape",
                "naive_wape",
                "ratio",
                "mae",
                "bias",
                "actual_sum",
                "prediction_sum",
                "rows",
                "coverage_50",
                "coverage_80",
            )
        ] == pytest.approx([10 / 60, 20 / 60, 0.5, 2.5, 0.5, 60, 62, 4, 0.5, 1])
        total = selected["total weekly sum", "all"]
        assert [
            total[name]
            for name in (
                "wape",
                "naive_wape",
                "ratio",
                "mae",
                "bias",
                "actual_sum",
                "prediction_sum",
                "rows",
            )
        ] == pytest.approx([2 / 60, 20 / 60, 0.1, 1, 1, 60, 62, 2])
        assert selected["all groups pooled", 1]["wape"] == pytest.approx(6 / 30)
        assert selected["total weekly sum", 1]["wape"] == pytest.approx(2 / 30)
        assert selected["total weekly sum", 2]["wape"] == 0
        assert total["coverage_50"] is None
        assert total["coverage_80"] is None
        if len(dims) == 2:
            assert total["channel"] == "total weekly sum"
            assert pooled["channel"] == "all groups pooled"
    without_bands = metric_columns(
        {phase: normalize_columns(request, raw) for phase in tables}, dims, "date", {}
    )
    assert all(
        row["coverage_80"] is None
        for row in table_rows(without_bands)
        if row["group"] == "total weekly sum"
    )
    bands = {
        phase: {
            ("2026-01-12", "2026-01-05", 1): [32, 30, 29, 33, 28, 34],
            ("2026-01-19", "2026-01-05", 2): [30, 30, 31, 32, 29, 33],
        }
        for phase in tables
    }
    covered = metric_columns(
        {phase: normalize_columns(request, raw) for phase in tables}, dims, "date", bands
    )
    for row in table_rows(covered):
        if row["group"] == "total weekly sum" and row["horizon_step"] == "all":
            assert row["coverage_50"] == 0.5
            assert row["coverage_80"] == 1


@pytest.mark.parametrize("naive", [None, 10])
def test_unavailable_and_zero_naive(naive):
    scores = score_rows([{"actual": 10, "forecast": 8, "naive_prediction": naive}])
    assert scores["ratio"] is None
    assert scores["naive_wape"] == (None if naive is None else 0)
    assert scores["coverage_50"] is None
    assert scores["coverage_80"] is None


@pytest.mark.parametrize("include", ["details + predictions", "everything"])
def test_metrics_export_and_slice_scores(input_run, tmp_path, include):
    folder, _, models, _ = input_run
    target = export_bundle(
        folder,
        {"folder": str(tmp_path / "export"), "include": include, "model": models[0]["tile_id"]},
    )

    def read_rows(name):
        return list(csv.DictReader(io.StringIO((target / name).read_text())))

    metrics = read_rows("error_metrics.csv")
    assert list(metrics[0]) == ["dataset_slice", "region", "horizon_step", *MEASURES]
    assert len(metrics) == 2 * (1 + 2) * (1 + 1)
    if include == "everything":
        demo = read_rows("demo_dataset.csv")
        for phase in ("validation", "test"):
            saved = read_rows(f"{phase}.csv")
            assert list(saved[0])[-2:] == ["wape", "total_wape"]
            selected = [row for row in demo if row["dataset_slice"] == phase]
            assert [(row["wape"], row["total_wape"]) for row in saved] == [
                (row["wape"], row["total_wape"]) for row in selected
            ]
        assert "wape" not in read_rows("forecast.csv")[0]
        assert "total_wape" not in read_rows("forecast.csv")[0]
