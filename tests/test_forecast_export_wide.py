"""wide exports preserve all origins and calibrate the total independently"""

import csv
import io
import json
from pathlib import Path

import numpy as np
import pytest

from smolsmort.forecast import evaluate as ev
from smolsmort.forecast import views
from smolsmort.forecast.export_wide import export_wide_csv, normalize_columns
from smolsmort.forecast.tables import read_table, write_table

HEADER = [
    "swo_date",
    "client_group",
    "region",
    "forecast_made_week",
    "horizon",
    "forecast",
    "actual",
    "lower_50",
    "upper_50",
    "lower_80",
    "upper_80",
    "total_forecast",
    "total_actual",
    "total_lower_50",
    "total_upper_50",
    "total_lower_80",
    "total_upper_80",
]
REAL_RUN = (
    Path.home()
    / "Documents/dev/ai_lab/forecast_workspace/.forecast-runs"
    / "20261115-155627-6f006a"
)


def read_csv(data):
    return list(csv.DictReader(io.StringIO(data.decode())))


@pytest.fixture
def saved_run(tmp_path):
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    request = {
        "prepared": str(prepared),
        "spec": {
            "mode": "timeseries",
            "columns": [
                {"name": "swo_date", "role": "time"},
                {"name": "client_group", "role": "dimension"},
                {"name": "region", "role": "dimension"},
                {"name": "units", "role": "target"},
            ],
        },
    }
    (tmp_path / "request.json").write_text(json.dumps(request))
    (tmp_path / "result.json").write_text(
        json.dumps(
            {
                "winner": {"family": "ridge", "fitness": 0.2},
                "ensemble": {"fitness": 0.1, "members": ["ridge", "snaive"]},
            }
        )
    )
    (tmp_path / "recipe.json").write_text(json.dumps({"family": "ridge"}))
    keys = ["a|special|north", "b|south"]
    write_table(
        prepared / "panel.parquet",
        {
            "__series": keys,
            "client_group": ["a|special", "b"],
            "region": ["north", "south"],
        },
    )
    tables = {}
    for phase, count in (("validation", 4), ("test", 2), ("forecast", 1)):
        records = [
            (origin, lead, group)
            for origin in range(count)
            for lead in (1, 2)
            for group in range(2)
        ]
        # one sparse horizon cannot fit a band; future coverage loses one group
        records += [(0, 3, 0)]
        if phase == "forecast":
            records.remove((0, 2, 1))
        origins = np.array(
            [np.datetime64("2026-01-05") + np.timedelta64(7 * o, "D") for o, _, _ in records]
        )
        leads = np.array([lead for _, lead, _ in records])
        predictions = np.array([10.0 + group * 10 + origin for origin, _, group in records])
        actuals = predictions + np.array([2 + origin + group for origin, _, group in records])
        if phase == "forecast":
            actuals[:] = np.nan
        if phase == "test":
            actuals[0] = np.nan
        columns = {
            "series": [keys[group] for _, _, group in records],
            "origin": origins,
            "step": origins + (leads * 7).astype("timedelta64[D]"),
            "weeks_ahead": leads,
            "prediction": predictions,
            "actual": actuals,
            "lower_50": predictions - 1,
            "upper_50": predictions + 1,
            "lower": predictions - 2,
            "upper": predictions + 2,
        }
        tables[phase] = columns
        write_table(tmp_path / f"{phase}.parquet", columns)
        write_table(tmp_path / f"ensemble_{phase}.parquet", columns)
    return tmp_path, request, tables


def test_wide_rows_dimensions_bands_and_totals(saved_run):
    folder, request, tables = saved_run
    data = export_wide_csv(folder, request, {})
    assert next(csv.reader(io.StringIO(data.decode()))) == HEADER
    rows = read_csv(data)
    assert len(rows) == 30
    assert {row["client_group"] for row in rows} == {"a|special", "b"}
    validation = tables["validation"]
    total_predictions = np.array([30 + 2 * origin for origin in range(4)])
    total_actuals = total_predictions + np.array([5 + 2 * origin for origin in range(4)])
    offset = 0
    for phase in tables:
        count = len(tables[phase]["prediction"])
        phase_rows = rows[offset : offset + count]
        offset += count
        for i, row in enumerate(phase_rows):
            assert row["forecast_made_week"] == str(tables[phase]["origin"][i])
            assert row["horizon"] == str(tables[phase]["weeks_ahead"][i])
            assert row["swo_date"] == str(tables[phase]["step"][i])
            for output, saved in (
                ("lower_50", "lower_50"),
                ("upper_50", "upper_50"),
                ("lower_80", "lower"),
                ("upper_80", "upper"),
            ):
                assert float(row[output]) == tables[phase][saved][i]
            peers = [
                peer
                for peer in phase_rows
                if all(
                    peer[key] == row[key] for key in ("swo_date", "forecast_made_week", "horizon")
                )
            ]
            assert float(row["total_forecast"]) == sum(float(peer["forecast"]) for peer in peers)
            if any(not peer["actual"] for peer in peers):
                assert row["total_actual"] == ""
            else:
                assert float(row["total_actual"]) == sum(float(peer["actual"]) for peer in peers)
            for level, suffix in ((0.5, "50"), (0.8, "80")):
                if row["horizon"] == "3":
                    assert row[f"total_lower_{suffix}"] == row[f"total_upper_{suffix}"] == ""
                else:
                    bands = ev.fit_lead_bands(
                        total_actuals,
                        total_predictions,
                        np.full(4, int(row["horizon"])),
                        level=level,
                    )
                    lo, hi = ev.apply_lead_bands(
                        bands, [float(row["total_forecast"])], [int(row["horizon"])]
                    )
                    assert float(row[f"total_lower_{suffix}"]) == pytest.approx(lo[0])
                    assert float(row[f"total_upper_{suffix}"]) == pytest.approx(hi[0])
    assert len(validation["prediction"]) == 17


def test_normalized_saved_tables_keep_exact_dimensions_and_numeric_measures(saved_run):
    _, request, tables = saved_run
    source = tables["test"]
    count = len(source["prediction"])
    normalized = normalize_columns(
        request,
        {
            **source,
            "bucket": np.ones(count),
            "family": ["ridge"] * count,
            "phase": ["test"] * count,
            "future": [False] * count,
            "other_label": ["label"] * count,
            "ratio": ["unknown"] * count,
            "naive_prediction": np.ones(count),
            "band_width_pct": np.full(count, np.nan),
        },
    )
    assert list(normalized) == [
        "swo_date",
        "client_group",
        "region",
        "forecast_made_week",
        "horizon",
        "forecast",
        "actual",
        "lower_50",
        "upper_50",
        "lower_80",
        "upper_80",
        "naive_prediction",
    ]
    assert normalized["client_group"][:2] == ["a|special", "b"]
    assert normalized["region"][:2] == ["north", "south"]
    assert all(type(value) is int for value in normalized["horizon"])
    assert normalized["swo_date"][0] == "2026-01-12"
    assert normalized["actual"][0] == ""
    assert normalize_columns(request, {"ratio": ["0.5", "1"]}) == {"ratio": [0.5, 1.0]}


@pytest.mark.parametrize("validation", ["missing", "short", "unknown"])
def test_insufficient_validation_and_absent_group_bands(saved_run, validation):
    folder, request, tables = saved_run
    path = folder / "ensemble_validation.parquet"
    if validation == "missing":
        path.unlink()
    else:
        columns = tables["validation"]
        if validation == "short":
            columns = {key: np.asarray(values)[:2] for key, values in columns.items()}
        else:
            columns["actual"][:] = np.nan
        write_table(path, columns)
    write_table(
        folder / "ensemble_forecast.parquet",
        {
            key: value
            for key, value in tables["forecast"].items()
            if key not in ("lower", "upper", "lower_50", "upper_50")
        },
    )
    rows = read_csv(export_wide_csv(folder, request, {}))
    assert all(
        row[key] == ""
        for row in rows
        for key in HEADER
        if key.startswith("total_lower") or key.startswith("total_upper")
    )
    assert all(
        row[key] == ""
        for row in rows[-4:]
        for key in ("lower_50", "upper_50", "lower_80", "upper_80")
    )


@pytest.mark.parametrize("model", [None, "winner", "ensemble"])
def test_selected_model(saved_run, model):
    folder, request, tables = saved_run
    columns = {**tables["forecast"], "prediction": np.full(4, 999.0)}
    write_table(folder / "ensemble_forecast.parquet", columns)
    rows = read_csv(export_wide_csv(folder, request, {"model": model}))
    expected = 10 if model == "winner" else 999
    assert float(rows[-4]["forecast"]) == expected


def test_invalid_run_and_entity(saved_run):
    folder, request, _ = saved_run
    request["spec"]["mode"] = "entity"
    with pytest.raises(views.ViewError, match="entity runs have no horizon or origin"):
        export_wide_csv(folder, request, {})
    (folder / "result.json").unlink()
    with pytest.raises(views.ViewError, match="has no result yet"):
        export_wide_csv(folder, request, {})


@pytest.mark.parametrize("cached", [False, True])
def test_tile_selection_does_not_write_caches(saved_run, cached):
    from smolsmort.forecast import tiles

    folder, request, tables = saved_run
    request.update(version=2, families=[{"name": "ridge"}, {"name": "snaive"}])
    (folder / "request.json").write_text(json.dumps(request))
    for family, offset in (("ridge", 0), ("snaive", 4)):
        target = folder / "families" / family
        target.mkdir(parents=True)
        for phase, columns in tables.items():
            write_table(
                target / f"{phase}.parquet",
                {
                    **columns,
                    "prediction": columns["prediction"] + offset,
                },
            )
    models = tiles.read_tiles(folder, include_ensembles=False)
    ensemble = tiles.ensemble_rows(folder)[0]
    (folder / "ensembles.json").write_text(json.dumps([ensemble]))
    if cached:
        target = folder / "tile_views" / ensemble["recipe_key"]
        target.mkdir(parents=True)
        for phase, columns in tables.items():
            write_table(
                target / f"{phase}.parquet",
                {
                    **columns,
                    "prediction": columns["prediction"] + 2,
                },
            )
    if not cached:
        (folder / "ensembles.json").unlink()
    before = {path: path.read_bytes() for path in folder.rglob("*") if path.is_file()}
    for model in models:
        rows = read_csv(export_wide_csv(folder, request, {"model": model["tile_id"]}))
        assert float(rows[0]["forecast"]) == (10 if model["family"] == "ridge" else 14)
    rows = read_csv(export_wide_csv(folder, request, {"model": ensemble["tile_id"]}))
    assert float(rows[0]["forecast"]) == 12
    assert bool(rows[0]["lower_50"]) == cached
    assert before == {path: path.read_bytes() for path in folder.rglob("*") if path.is_file()}


@pytest.mark.skipif(not REAL_RUN.exists(), reason="real run fixture is local")
def test_real_run_row_count_and_required_cells():
    request = views.read_request(REAL_RUN)
    selected = views._select_model(REAL_RUN, views._require_done(REAL_RUN), {})
    paths = {
        phase: views._optional_prediction_path(REAL_RUN, {"model": selected["model"]}, phase)
        for phase in ("validation", "test", "forecast")
    }
    rows = read_csv(export_wide_csv(REAL_RUN, request, {}))
    assert len(rows) == sum(len(read_table(path)["prediction"]) for path in paths.values())
    assert "phase" not in rows[0]
    assert "swo_date" in rows[0]
    assert all(row["forecast"] and row["horizon"] for row in rows)
