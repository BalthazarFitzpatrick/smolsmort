"""stacked exports keep actuals once and preserve predictions, horizons and group labels"""

import csv
import io
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from test_forecast_export_dialog import input_run as input_run
from test_forecast_export_dialog import saved_run as saved_run

from smolsmort.forecast import export_bundle as bundle
from smolsmort.forecast.export_bundle import export_bundle

BANDS = ["lower_50", "upper_50", "lower_80", "upper_80"]
TOTALS = [f"total_{name}" for name in ["actual", "prediction", *BANDS, "wape"]]
MEASURES = ["actual", "prediction", *BANDS, "wape", *TOTALS, "horizon_step", "dataset_slice"]


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        return reader.fieldnames, list(reader)


@pytest.mark.parametrize("include", ["details + predictions", "everything"])
@pytest.mark.parametrize("family,prediction", [("ridge", 8), ("snaive", 6), ("ensemble", 7)])
def test_demo_stacks_selected_model_with_history(input_run, tmp_path, include, family, prediction):
    folder, _, models, ensemble = input_run
    model = ensemble if family == "ensemble" else next(m for m in models if m["family"] == family)
    target = export_bundle(
        folder,
        {"folder": str(tmp_path / "export"), "include": include, "model": model["tile_id"]},
    )
    header, rows = read_rows(target / "demo_dataset.csv")
    assert header == ["week", "region", *MEASURES]
    assert all(list(row) == header and None not in row.values() for row in rows)
    assert Counter(row["dataset_slice"] for row in rows) == {
        "input": 6,
        "validation": 2,
        "test": 2,
        "prediction": 2,
    }
    assert [row["dataset_slice"] for row in rows] == (
        ["input"] * 6 + ["validation"] * 2 + ["test"] * 2 + ["prediction"] * 2
    )
    history = [row for row in rows if row["dataset_slice"] == "input"]
    _, inputs = read_rows(target / "input.csv")
    assert [{key: row[key] for key in inputs[0]} for row in history] == inputs
    assert all(row["horizon_step"] == "0" and row["prediction"] == "" for row in history)
    assert all(row["horizon_step"] == "1" for row in rows if row not in history)
    assert all(float(row["prediction"]) == prediction for row in rows if row not in history)
    assert all(row["actual"] == "" for row in rows if row not in history)
    assert all(row[name] == "" for row in history for name in [*BANDS, "wape"])
    assert all(row["wape"] == "" for row in rows if row["dataset_slice"] == "prediction")
    _, wide = read_rows(target / "wide.csv")
    assert [[row[name] for name in BANDS] for row in rows if row not in history] == [
        [row[name] for name in BANDS] for row in wide
    ]
    assert {row["region"] for row in rows} <= {row["region"] for row in inputs}
    readme = (target / "README.md").read_text()
    assert (
        "actuals appear once in the input slice, predictions per horizon step in the other slices; "
        "join or blend on date and dims to compare"
    ) in readme
    assert "before the first validation date" not in readme
    assert "plain sum of actual over-counts" not in readme
    assert "within this group, slice and horizon_step" in readme
    assert (
        "total bands are the sum of the group bounds, wider than a band fitted on the total itself"
        in readme
    )
    assert "differ from wide.csv total bands" in readme
    assert (
        "wape is blank for input and prediction, no finite pairs or zero absolute actuals" in readme
    )


def test_demo_details_only_writes_none(input_run, tmp_path):
    folder, _, models, _ = input_run
    target = export_bundle(folder, {"folder": str(tmp_path), "model": models[0]["tile_id"]})
    assert not (target / "demo_dataset.csv").exists()


@pytest.mark.parametrize("empty_validation", [False, True])
def test_demo_keeps_dimensions_missing_groups_and_mixed_horizons(monkeypatch, empty_validation):
    request = {
        "spec": {
            "mode": "timeseries",
            "columns": [
                {"name": "date", "role": "time"},
                {"name": "region", "role": "dimension"},
                {"name": "channel", "role": "dimension"},
                {"name": "units", "role": "target"},
                {"name": "price", "role": "measure"},
            ],
        }
    }
    tables = {
        phase: {
            "step": ["2026-01-12"] * 3,
            "region": [None, np.nan, "none"],
            "channel": ["web"] * 3,
            "actual": [5, 5, 5],
            "prediction": [6, 7, 8],
            "weeks_ahead": [1, 2, 2],
            "price": [100, 100, 100],
            "lower": [0, 0, 0],
            "naive_prediction": [4, 4, 4],
        }
        for phase in ("validation", "test", "forecast")
    }
    if empty_validation:
        tables["validation"] = {}
    monkeypatch.setattr(bundle.views, "_require_done", lambda _: {})
    monkeypatch.setattr(bundle, "_select_predictions", lambda *args: ({}, None))
    monkeypatch.setattr(
        bundle, "_read_predictions", lambda run, selected, phase, tiles: tables[phase]
    )
    history = bundle._table_csv(
        {
            "date": ["2026-01-05", "2026-01-12", "2026-01-19"],
            "region": [None, np.nan, "none"],
            "channel": ["web"] * 3,
            "actual": [5, 5, 5],
        }
    )
    data = bundle._demo_csv(Path("unused"), {"request": request, "model": "winner"}, history)
    reader = csv.DictReader(io.StringIO(data.decode()))
    rows = list(reader)
    assert reader.fieldnames == [
        "date",
        "region",
        "channel",
        *MEASURES,
    ]
    assert len(rows) == (9 if empty_validation else 12)
    for phase in ("input", "validation", "test", "prediction"):
        selected = [row for row in rows if row["dataset_slice"] == phase]
        if phase == "validation" and empty_validation:
            assert selected == []
            continue
        assert [row["region"] for row in selected] == ["", "", "none"]
        assert [row["channel"] for row in selected] == ["web"] * 3
        assert [row["horizon_step"] for row in selected] == (
            ["0"] * 3 if phase == "input" else ["1", "2", "2"]
        )
        assert [row["actual"] for row in selected] == (["5"] * 3 if phase == "input" else [""] * 3)
        assert [row["prediction"] for row in selected] == (
            [""] * 3 if phase == "input" else ["6.0", "7.0", "8.0"]
        )
        if phase != "input":
            assert [float(row["total_prediction"]) for row in selected] == [6, 15, 15]
            assert [float(row["total_lower_80"]) for row in selected] == [0, 0, 0]
    assert [row["date"] for row in rows if row["dataset_slice"] == "input"] == [
        "2026-01-05",
        "2026-01-12",
        "2026-01-19",
    ]


@pytest.mark.parametrize("dimensions", [[], ["region", "channel"]])
def test_demo_wape_pools_dates_per_group_slice_and_step(monkeypatch, dimensions):
    request = {
        "spec": {
            "mode": "timeseries",
            "columns": [
                {"name": "date", "role": "time"},
                *[{"name": name, "role": "dimension"} for name in dimensions],
            ],
        }
    }
    tables = {}
    for phase, predictions in (
        ("validation", [8, 24, 3, 9, 9, 1, 2, 100, np.inf]),
        ("test", [5, 20, 2, 12, 3, 4, 5, 100, np.inf]),
        ("forecast", [8, 24, 0, 9, 9, 1, 2, 100, np.inf]),
    ):
        tables[phase] = {
            "step": [f"2026-01-{day:02d}" for day in range(1, 10)],
            **(
                {"region": ["a"] * 5 + ["zero"] * 2 + ["a"] * 2, "channel": ["web"] * 9}
                if dimensions
                else {}
            ),
            "actual": [10, 20, 0, -10, 30, 0, 0, np.nan, 5],
            "prediction": predictions,
            "weeks_ahead": [1, 1, 1, 12, 12, 1, 1, 1, 1],
            "lower_50": [value - 1 for value in predictions],
            "upper_50": [value + 1 for value in predictions],
            "lower": [value - 2 for value in predictions],
            "upper": [value + 2 for value in predictions],
        }
    monkeypatch.setattr(bundle.views, "_require_done", lambda _: {})
    monkeypatch.setattr(bundle, "_select_predictions", lambda *args: ({}, None))
    monkeypatch.setattr(
        bundle, "_read_predictions", lambda run, selected, phase, tiles: tables[phase]
    )
    history = bundle._table_csv({"date": ["2025-01-01"], "actual": [10]})
    data = bundle._demo_csv(Path("unused"), {"request": request, "model": "winner"}, history)
    reader = csv.DictReader(io.StringIO(data.decode()))
    rows = list(reader)
    assert reader.fieldnames == ["date", *dimensions, *MEASURES]
    assert all(list(row) == reader.fieldnames and None not in row.values() for row in rows)
    for row in rows:
        phase = row["dataset_slice"]
        if phase in ("input", "prediction"):
            assert row["wape"] == ""
            assert row["total_wape"] == ""
            if phase == "input":
                assert all(row[name] == "" for name in BANDS)
            continue
        step = int(row["horizon_step"])
        pooled = {"validation": {1: 12 / 30, 12: 40 / 40}, "test": {1: 16 / 30, 12: 49 / 40}}
        assert float(row["total_wape"]) == pytest.approx(pooled[phase][step])
        if dimensions and row["region"] == "zero":
            assert row["wape"] == ""
        else:
            # hand sums include zero-actual errors and exclude nonfinite pairs
            expected = {"validation": {1: 9 / 30, 12: 40 / 40}, "test": {1: 7 / 30, 12: 49 / 40}}[
                phase
            ][step]
            if not dimensions and step == 1:
                expected = (12 if phase == "validation" else 16) / 30
            assert float(row["wape"]) == pytest.approx(expected)
        index = int(row["date"][-2:]) - 1
        for output, saved in zip(BANDS, ["lower_50", "upper_50", "lower", "upper"], strict=True):
            value = tables[phase][saved][index]
            assert row[output] == (str(float(value)) if np.isfinite(value) else "")


def test_demo_totals_sum_group_values_and_pool_wape(monkeypatch):
    request = {
        "spec": {
            "mode": "timeseries",
            "columns": [
                {"name": "date", "role": "time"},
                {"name": "group", "role": "dimension"},
            ],
        }
    }
    table = {
        "step": ["2026-01-01"] * 2 + ["2026-01-02"] * 2 + ["2026-01-03"] * 2 + ["2026-01-04"],
        "group": ["a", "b", "a", "b", "a", "b", "a"],
        "actual": [10, 20, 0, 0, 10, 30, 0],
        "prediction": [8, 24, 1, 2, 5, 20, 9],
        "weeks_ahead": [1, 1, 1, 1, 12, 12, 2],
        "lower_50": [1, 2, np.nan, np.nan, 5, np.nan, 0],
        "upper_50": [9, 25, np.nan, np.nan, 6, 21, 10],
        "lower": [-1, -2, np.nan, np.nan, 3, 4, -1],
        "upper": [10, 30, np.nan, np.nan, 7, 22, 12],
    }
    monkeypatch.setattr(bundle.views, "_require_done", lambda _: {})
    monkeypatch.setattr(bundle, "_select_predictions", lambda *args: ({}, None))
    monkeypatch.setattr(bundle, "_read_predictions", lambda *args: table)
    history = bundle._table_csv(
        {
            "date": ["2026-01-01"] * 2 + ["2026-01-02"],
            "group": ["a", "b", "a"],
            "actual": [10, 20, 0],
        }
    )
    data = bundle._demo_csv(Path("unused"), {"request": request, "model": "winner"}, history)
    reader = csv.DictReader(io.StringIO(data.decode()))
    rows = list(reader)
    assert reader.fieldnames == ["date", "group", *MEASURES]
    for row in rows:
        assert list(row) == reader.fieldnames and None not in row.values()
        day = int(row["date"][-2:])
        if row["dataset_slice"] == "input":
            assert float(row["total_actual"]) == (30 if day == 1 else 0)
            assert all(row[name] == "" for name in TOTALS if name != "total_actual")
            continue
        assert row["total_actual"] == ""
        assert float(row["total_prediction"]) == {1: 32, 2: 3, 3: 25, 4: 9}[day]
        expected_bands = {1: [3, 34, -3, 40], 2: None, 3: [5, 27, 7, 29], 4: [0, 10, -1, 12]}[day]
        if expected_bands is None:
            assert all(row[f"total_{name}"] == "" for name in BANDS)
        else:
            assert [float(row[f"total_{name}"]) for name in BANDS] == expected_bands
        if row["dataset_slice"] == "prediction" or day == 4:
            assert row["total_wape"] == ""
        else:
            assert float(row["total_wape"]) == pytest.approx(9 / 30 if day in (1, 2) else 15 / 40)


def test_real_run_demo_smoke(tmp_path):
    folder = (
        Path.home()
        / "Documents"
        / "dev"
        / "ai_lab"
        / "forecast_workspace"
        / ".forecast-runs"
        / "20261115-155627-6f006a"
    )
    if not (folder / "request.json").is_file():
        pytest.skip("real run absent")
    request = json.loads((folder / "request.json").read_text())
    if (
        not Path(request["spec"]["source"]).is_file()
        or not (Path(request["prepared"]) / "panel.parquet").is_file()
    ):
        pytest.skip("real dataset or panel absent")
    target = export_bundle(
        folder, {"folder": str(tmp_path), "include": "everything", "model": "winner"}
    )
    header, rows = read_rows(target / "demo_dataset.csv")
    assert header == [
        "swo_date",
        "client_group",
        *MEASURES,
    ]
    _, inputs = read_rows(target / "input.csv")
    tables = {
        phase: read_rows(target / f"{phase}.csv")[1] for phase in ("validation", "test", "forecast")
    }
    assert len(rows) == len(inputs) + sum(len(table) for table in tables.values())
    assert Counter(row["dataset_slice"] for row in rows) == {
        "input": len(inputs),
        "validation": len(tables["validation"]),
        "test": len(tables["test"]),
        "prediction": len(tables["forecast"]),
    }
    history = [row for row in rows if row["dataset_slice"] == "input"]
    assert [{key: row[key] for key in inputs[0]} for row in history] == inputs
    assert all(row["prediction"] == "" and row["horizon_step"] == "0" for row in history)
    assert {row["client_group"] for row in rows} <= {row["client_group"] for row in inputs}
    for phase, table in tables.items():
        selected = [
            row
            for row in rows
            if row["dataset_slice"] == ("prediction" if phase == "forecast" else phase)
        ]
        assert all(row["actual"] == "" for row in selected)
        assert [[row[name] for name in BANDS] for row in selected] == [
            [row.get(name, "") for name in BANDS] for row in table
        ]
        assert [
            (row["swo_date"], row["client_group"], row["horizon_step"], row["prediction"])
            for row in selected
        ] == [
            (row["swo_date"], row["client_group"], row["horizon"], row["forecast"]) for row in table
        ]
