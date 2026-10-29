"""the input search ranks candidate columns on training rows, keeps pinned columns in every set,
and spends the deadline on explore, halving and a final run"""

from __future__ import annotations

import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from smolsmort.forecast import input_search, pipeline, prep
from smolsmort.forecast.features import split_family_name
from smolsmort.forecast.input_search import (
    InputSearch,
    draw_set,
    input_odds,
    screen_inputs,
    search_inputs,
    with_inputs,
)
from smolsmort.forecast.pipeline import load_workspace
from smolsmort.forecast.screen import screen_families
from smolsmort.forecast.spec import SERIES, STEP, Column, PrepSpec, SpecError

WEEKS, SERIES_COUNT, HORIZON = 156, 6, 8


def make_tables(seed=7):
    """seasonal series; `orders` leads the year-on-year change by one week, `site` share by four"""
    rng = np.random.default_rng(seed)
    steps = np.datetime64("2023-01-02") + np.arange(WEEKS) * np.timedelta64(7, "D")
    keys = [f"s{index}" for index in range(SERIES_COUNT)]
    target, orders, noise, inputs = [], [], [], []
    for key in keys:
        shock = rng.normal(0, 0.3, WEEKS + 4)
        level = 3 + 0.5 * np.sin(2 * np.pi * np.arange(WEEKS) / 52)
        drift = np.cumsum(shock[:WEEKS])
        y = np.expm1(level + 0.2 * drift)
        target.append(np.round(np.clip(y, 0, None)))
        orders.append(np.r_[shock[1 : WEEKS + 1]])
        noise.append(rng.normal(size=WEEKS))
        for index, step in enumerate(steps):
            share = 1 / (1 + np.exp(-3 * shock[min(index + 4, WEEKS + 3)]))
            north = int(round(100 * share))
            inputs.append((key, step, "site", "north", north))
            inputs.append((key, step, "site", "south", 100 - north))
            inputs.append((key, step, "dull", "x", int(rng.integers(40, 60))))
            inputs.append((key, step, "dull", "y", int(rng.integers(40, 60))))
    panel = {
        SERIES: np.repeat(keys, WEEKS),
        STEP: np.tile(steps, SERIES_COUNT),
        "units": np.concatenate(target),
        "orders": np.concatenate(orders),
        "noise": np.concatenate(noise),
    }
    table = {
        SERIES: np.array([row[0] for row in inputs]),
        STEP: np.array([row[1] for row in inputs]),
        "__column": np.array([row[2] for row in inputs]),
        "__level": np.array([row[3] for row in inputs]),
        "rows": np.array([row[4] for row in inputs]),
    }
    return {"panel.parquet": panel, "inputs.parquet": table}


def make_spec(**pins):
    return PrepSpec(
        "synthetic.csv",
        "timeseries",
        "regression",
        (
            Column("day", "time"),
            Column("units", "target", "sum"),
            Column("group", "dimension"),
            Column("orders", "measure", "mean", pin=pins.get("orders", False)),
            Column("noise", "measure", "mean", pin=pins.get("noise", False)),
            Column("site", "input"),
            Column("dull", "input"),
        ),
        step="week",
        horizon=HORIZON,
    )


@pytest.fixture
def tables(monkeypatch):
    data = make_tables()
    monkeypatch.setattr(input_search, "read_table", lambda path, **kwargs: data[Path(path).name])
    return data


def test_leading_columns_outrank_noise(tables):
    ranking = screen_inputs(make_spec(), Path("synthetic"))
    order = [entry["column"] for entry in ranking]
    assert set(order) == {"orders", "noise", "site", "dull"}
    assert order[0] == "orders"
    assert order.index("site") < order.index("dull")
    assert order.index("site") < order.index("noise")
    # one shock out of the 52 in a year-on-year change: about 1 / sqrt(52)
    assert ranking[0]["rho"][1] > 0.1
    assert [entry["score"] for entry in ranking] == sorted(
        (entry["score"] for entry in ranking), reverse=True
    )


def test_screen_reads_training_rows_only(tables):
    expected = screen_inputs(make_spec(), Path("synthetic"))
    panel = tables["panel.parquet"]
    # validation starts 26 + 13 steps before the end for 156 weekly steps and horizon 8
    held_out = panel[STEP] >= np.unique(panel[STEP])[WEEKS - 39]
    panel["units"][held_out] = 1e9
    panel["orders"][held_out] = -1e9
    assert screen_inputs(make_spec(), Path("synthetic")) == expected


def test_pinned_column_is_flagged_and_gets_odds_one(tables):
    ranking = screen_inputs(make_spec(noise=True), Path("synthetic"))
    odds = input_odds(ranking)
    assert next(e for e in ranking if e["column"] == "noise")["pinned"]
    assert odds["noise"] == 1.0
    assert odds["orders"] == pytest.approx(0.85)
    assert all(0.15 <= p <= 0.85 for name, p in odds.items() if name != "noise")


def test_draws_keep_pins_and_never_repeat():
    odds = {"a": 1.0, "b": 0.85, "c": 0.5, "d": 0.15}
    rng, seen = random.Random(1), set()
    while (genes := draw_set(odds, rng, seen, parents=list(seen)[:2])) is not None:
        assert "a" in genes
        assert genes == tuple(sorted(genes))
        seen.add(genes)
    # every subset of the three free columns, each with the pin
    assert len(seen) == 8


def test_with_inputs_keeps_time_target_and_dimensions():
    spec = with_inputs(make_spec(), ("site",))
    assert [c.name for c in spec.columns] == ["day", "units", "group", "site"]


def test_search_spends_the_deadline_and_writes_every_trial(tmp_path, monkeypatch):
    now = [0.0]
    calls = []
    ranking = [
        {"column": "orders", "score": 0.3, "rho": {}, "pinned": True},
        {"column": "site", "score": 0.2, "rho": {}, "pinned": False},
        {"column": "noise", "score": 0.05, "rho": {}, "pinned": False},
        {"column": "dull", "score": 0.01, "rho": {}, "pinned": False},
    ]
    monkeypatch.setattr(prep, "prepare", lambda spec, root: SimpleNamespace(folder=tmp_path))
    monkeypatch.setattr(input_search, "screen_inputs", lambda spec, folder: ranking)

    def trial(spec, families, budget_s, until_plateau):
        names = {c.name for c in spec.columns}
        calls.append((names, tuple(families or ()), budget_s, until_plateau))
        now[0] += budget_s if not until_plateau else budget_s / 2
        # site helps, noise hurts
        val = 0.2 - 0.03 * ("site" in names) + 0.02 * ("noise" in names) + budget_s * 1e-7
        return {"run": f"r{len(calls)}", "val": val}

    plan = InputSearch(deadline=5 * 3600, explore_s=360, final_families=("ridge", "ets"))
    folder = search_inputs(
        tmp_path, tmp_path, make_spec(orders=True), plan, trial=trial, clock=lambda: now[0]
    )
    state = json.loads((folder / "search.json").read_text())
    stages = [t["stage"] for t in state["trials"]]
    assert stages[0] == "control" and state["trials"][0]["inputs"] == []
    assert stages[1] == "explore1" and state["trials"][1]["inputs"] == ["orders"]
    assert any(s.startswith("halve") for s in stages)
    assert any(s.startswith("final") for s in stages)
    assert all("orders" in t["inputs"] for t in state["trials"][1:])
    assert now[0] <= plan.deadline
    explored = [tuple(t["inputs"]) for t in state["trials"] if t["stage"].startswith("explore")]
    assert len(explored) == len(set(explored))
    final = next(c for c in calls if c[3])
    assert final[1] == ("ridge", "ets")
    assert "site" in final[0] and "noise" not in final[0]
    assert "site" in state["best"]["inputs"]


def test_pin_is_left_out_of_the_digest_unless_set():
    plain = make_spec()
    unpinned = PrepSpec(
        "synthetic.csv",
        "timeseries",
        "regression",
        tuple(Column(c.name, c.role, c.aggregation) for c in plain.columns),
        step="week",
        horizon=HORIZON,
    )
    assert plain.digest() == unpinned.digest()
    assert make_spec(orders=True).digest() != plain.digest()


def test_pin_on_a_target_is_refused():
    with pytest.raises(SpecError):
        Column("units", "target", "sum", pin=True)
    with pytest.raises(SpecError):
        Column("orders", "measure", "sum", pin="yes")


def test_pinned_families_are_always_on_in_the_model_search(monkeypatch):
    from test_forecast_screen import make_panel

    spec, data = make_panel()
    monkeypatch.setattr(pipeline, "read_table", lambda path, **kwargs: data[path.name])
    ws = load_workspace(spec, Path("synthetic"), {"step": "week"})
    ranking = screen_families(ws, pinned=frozenset({"noise_07"}))
    pinned = [r for r in ranking if split_family_name(r["family"])[0] == "noise_07"]
    assert pinned and all(r["p_on"] == 1.0 and r["pinned"] for r in pinned)
    others = [r for r in ranking if split_family_name(r["family"])[0] != "noise_07"]
    assert all("pinned" not in r and r["p_on"] <= 0.9 for r in others)


def test_trial_runs_forward_the_holdout(tmp_path, monkeypatch):
    from smolsmort.forecast import runs

    seen = {}
    monkeypatch.setattr(prep, "prepare", lambda spec, root: SimpleNamespace(folder=tmp_path))

    def start(root, spec, prepared, **kwargs):
        seen.update(kwargs)
        return "run1"

    monkeypatch.setattr(runs, "start_run", start)
    monkeypatch.setattr(runs, "wait_run", lambda root, run_id, timeout: {"state": "done"})
    trial = input_search._run_trial(tmp_path, tmp_path, {"holdout": "seasonal"})
    assert trial(make_spec(), ["ridge"], 60, False) == {"run": "run1", "state": "done"}
    assert seen["holdout"] == "seasonal"
