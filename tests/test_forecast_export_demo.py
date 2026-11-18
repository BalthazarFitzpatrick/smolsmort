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
    assert header == ["week", "region", "actual", "prediction", "horizon_step", "dataset_slice"]
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
    assert {row["region"] for row in rows} <= {row["region"] for row in inputs}
    readme = (target / "README.md").read_text()
    assert (
        "actuals appear once in the input slice, predictions per horizon step in the other slices; "
        "join or blend on date and dims to compare"
    ) in readme
    assert "before the first validation date" not in readme
    assert "plain sum of actual over-counts" not in readme


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
        "actual",
        "prediction",
        "horizon_step",
        "dataset_slice",
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
    assert [row["date"] for row in rows if row["dataset_slice"] == "input"] == [
        "2026-01-05",
        "2026-01-12",
        "2026-01-19",
    ]


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
        "actual",
        "prediction",
        "horizon_step",
        "dataset_slice",
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
        assert [
            (row["swo_date"], row["client_group"], row["horizon_step"], row["prediction"])
            for row in selected
        ] == [
            (row["swo_date"], row["client_group"], row["horizon"], row["forecast"]) for row in table
        ]
