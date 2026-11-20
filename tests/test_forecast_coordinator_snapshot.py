"""the coordinator parses each family checkpoint once per revision"""

import json
import os

import pytest

from smolsmort.forecast import coordinator, family_worker


@pytest.fixture(autouse=True)
def clear_snapshots():
    coordinator._SNAPSHOTS.clear()
    yield
    coordinator._SNAPSHOTS.clear()


def write_checkpoint(folder, name, fitness):
    path = folder / "families" / name / "checkpoint.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "fitness": fitness,
        "families": 1,
        "elapsed_s": 2.0,
        "phase": "broad",
        "cost_hint": 4.0,
        "genome": {},
    }
    staged = path.with_suffix(".tmp")
    staged.write_text(json.dumps({"entries": [entry], "generation": 1}))
    os.replace(staged, path)


def count_loads(monkeypatch):
    counter = {"loads": 0}
    original = json.loads

    def counting(*args, **kwargs):
        counter["loads"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(family_worker.json, "loads", counting)
    return counter


def test_one_load_per_revision_across_callers(tmp_path, monkeypatch):
    names = ["ridge", "xgboost"]
    for name in names:
        write_checkpoint(tmp_path, name, 1.5)
    counter = count_loads(monkeypatch)
    states = {name: {"state": "done"} for name in names}
    calibration = {}
    for _ in range(3):
        for name in names:
            coordinator.best_entry(tmp_path, name)
        coordinator.calibrate_families(tmp_path, states, calibration, 100)
        coordinator.estimate_finish_seconds(tmp_path, names, calibration, {})
    assert counter["loads"] == 2


def test_rewrite_invalidates_snapshot(tmp_path):
    write_checkpoint(tmp_path, "ridge", 1.5)
    assert coordinator.best_entry(tmp_path, "ridge")["fitness"] == 1.5
    write_checkpoint(tmp_path, "ridge", 0.5)
    assert coordinator.best_entry(tmp_path, "ridge")["fitness"] == 0.5


def test_missing_checkpoint_is_empty_and_not_cached(tmp_path):
    assert coordinator.best_entry(tmp_path, "ridge") is None
    write_checkpoint(tmp_path, "ridge", 1.0)
    assert coordinator.best_entry(tmp_path, "ridge")["fitness"] == 1.0
