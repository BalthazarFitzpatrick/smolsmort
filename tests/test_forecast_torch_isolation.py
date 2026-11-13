"""torch work stays in family children, including finish cache misses"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict

import numpy as np
import pytest
from test_forecast_coordinator import make_trending_panel

from smolsmort.forecast.pipeline import load_workspace, make_genome
from smolsmort.forecast.tables import read_table


def run_script(source, *args, timeout=120):
    env = dict(os.environ)
    env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    completed = subprocess.run(
        [sys.executable, "-c", source, *map(str, args)],
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return completed


@pytest.mark.parametrize(
    ("name", "conflict"),
    [
        ("mlp", "xgboost"),
        ("mlp", "lightgbm"),
        ("lstm", "xgboost"),
        ("lstm", "lightgbm"),
        ("xgboost", "torch"),
        ("lightgbm", "torch"),
        ("model", "torch"),
    ],
)
@pytest.mark.parametrize("operation", ["fit", "predict"])
def test_model_entrypoints_reject_mixed_runtimes(name, conflict, operation):
    run_script(
        """
import importlib
import sys
from types import ModuleType, SimpleNamespace
from smolsmort.forecast import model
from smolsmort.forecast.families import get_family

name, conflict, operation = sys.argv[1:]
assert not any(key in sys.modules for key in ('torch', 'xgboost', 'lightgbm'))
sys.modules[conflict] = ModuleType(conflict)
tree = SimpleNamespace(booster=SimpleNamespace(predict=lambda x: None), feature_types=None)
if name == 'model':
    call = (lambda: model.fit_model(None, objective='squared')) if operation == 'fit' else (
        lambda: model.predict(tree, None)
    )
elif operation == 'fit':
    call = lambda: get_family(name).fit(None, objective='squared')
else:
    module = importlib.import_module('smolsmort.forecast.families.' + name)
    fitted_type = getattr(module, name.capitalize() + 'Fitted')
    fitted = fitted_type.__new__(fitted_type)
    if name == 'xgboost':
        fitted.fitted = tree
    call = lambda: fitted.predict(None)
try:
    call()
except model.ModelError as error:
    assert 'torch families must run in their own process' in str(error), str(error)
else:
    raise AssertionError('mixed runtime work was allowed')
assert conflict == 'torch' or 'torch' not in sys.modules
""",
        name,
        conflict,
        operation,
    )


def test_neural_worker_imports_leave_tree_libraries_unloaded():
    run_script(
        """
import importlib
import sys
for name in ('pipeline', 'model', 'features', 'candidate_worker', 'family_worker', 'exit_test'):
    importlib.import_module('smolsmort.forecast.' + name)
assert not any(name in sys.modules for name in ('torch', 'xgboost', 'lightgbm'))
"""
    )


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.mark.skipif(
    any(importlib.util.find_spec(name) is None for name in ("torch", "xgboost")),
    reason="needs torch and xgboost",
)
@pytest.mark.parametrize(
    ("winner", "cache_damage"), [("xgboost", "deleted"), ("mlp", "mismatched")]
)
def test_finish_refits_isolated_member_in_child(tmp_path, winner, cache_damage):
    spec, prepared = make_trending_panel(tmp_path)
    ws = load_workspace(spec, prepared.folder, prepared.summary)
    features = [name for name in ws.families if "lag" in name][:2]
    assert features
    truth = np.concatenate(
        [frame.y[ws.series_masks[bucket]["val"]] for bucket, frame in ws.frames.items()]
    )
    run = tmp_path / "run"
    names = [winner, "mlp" if winner == "xgboost" else "xgboost"]
    for name in names:
        (run / "families" / name).mkdir(parents=True)
    params = {
        "mlp": {"n_layers": 1, "width": 8, "batch_size": 128, "learning_rate": 0.02},
        "xgboost": {"max_depth": 2, "eta": 0.2},
    }
    entries = [
        {
            "family": name,
            "genome": make_genome(features, "squared", params[name], name).to_dict(),
            "fitness": 0.1 + index * 0.1,
            "prediction": (truth * (0.98 + index * 0.04)).tolist(),
        }
        for index, name in enumerate(names)
    ]
    save_json(
        run / "request.json",
        {
            "version": 2,
            "families": [{"name": name} for name in names],
            "spec": asdict(spec),
            "prepared": str(prepared.folder),
            "summary": prepared.summary,
            "nthread": 2,
            "ensemble": {"enabled": True, "top": 2},
        },
    )
    save_json(
        run / "coordinator.json",
        {
            "deadline": time.time() + 180,
            "nthread": 2,
            "families": {name: {"state": "done", "best_error": 0.1} for name in names},
        },
    )
    save_json(run / "finalists.json", entries)
    run_script(
        """
import json
import sys
from pathlib import Path
from smolsmort.forecast.exit_test import score_exit
folder = Path(sys.argv[1])
entry = next(item for item in json.loads((folder / 'finalists.json').read_text())
             if item['family'] == 'mlp')
score_exit(folder, 'mlp', entry)
assert 'xgboost' not in sys.modules and 'lightgbm' not in sys.modules
""",
        run,
    )
    target = run / "families" / "mlp"
    marker_path = target / "exit_test.json"
    previous = json.loads(marker_path.read_text())
    previous_prediction = read_table(target / "test.parquet")["prediction"].copy()
    if cache_damage == "deleted":
        (target / previous["tables_path"] / "cache.json").unlink()
    else:
        save_json(marker_path, {**previous, "recipe_key": "different recipe"})
    finished = run_script(
        """
import json
import os
import sys
from pathlib import Path
from smolsmort.forecast.ensemble import run
run(Path(sys.argv[1]), os.getppid())
assert 'torch' not in sys.modules
print(json.dumps({'torch_imported': 'torch' in sys.modules}))
""",
        run,
    )
    completed = json.loads((run / "finish_complete.json").read_text())
    assert completed["family"] in {*names, "ensemble"}
    assert json.loads(finished.stdout)["torch_imported"] is False
    result = json.loads((run / "result.json").read_text())
    assert result["winner"]["family"] == winner
    assert set(result["ensemble"]["members"]) == set(names)
    current = json.loads(marker_path.read_text())
    assert current["recipe_key"] == make_genome(features, "squared", params["mlp"], "mlp").key()
    assert current["tables_path"] != previous["tables_path"]
    for name in names:
        family = run / "families" / name
        for artifact in ("forecast.parquet", "test.parquet", "validation.parquet", "band.parquet"):
            assert (family / artifact).is_file(), (name, artifact)
        if name == "mlp":
            assert result["families"][name]["importance"] is None
        else:
            assert (family / "importance.json").is_file(), name
        assert result["families"][name]["verdict"]
    for artifact in ("forecast.parquet", "test.parquet", "band.parquet", "winner_forecast.parquet"):
        assert (run / artifact).is_file(), artifact
    np.testing.assert_allclose(
        read_table(target / "test.parquet")["prediction"], previous_prediction
    )


@pytest.mark.parametrize("failure", ["exit", "timeout", "deadline", "launch"])
def test_isolated_child_failure_is_clean_and_bounded(tmp_path, monkeypatch, failure):
    from smolsmort.forecast import coordinator, exit_test
    from smolsmort.forecast.pipeline import PipelineError

    stopped, timeouts = [], []

    class Process:
        def wait(self, timeout):
            timeouts.append(timeout)
            if failure == "timeout":
                raise subprocess.TimeoutExpired("exit scorer", timeout)
            return -6

    process = Process()

    def launch(*args):
        if failure == "launch":
            raise OSError("launch refused")
        assert failure != "deadline"
        return process

    monkeypatch.setattr(exit_test, "launch_exit", launch)
    monkeypatch.setattr(coordinator, "stop_process", stopped.append)
    monkeypatch.setattr(exit_test.time, "time", lambda: 100)
    with pytest.raises(PipelineError, match="family 'mlp':"):
        exit_test.score_isolated(tmp_path, "mlp", {}, deadline=99 if failure == "deadline" else 107)
    assert stopped == ([process] if failure in {"exit", "timeout"} else [])
    assert timeouts == ([7] if failure in {"exit", "timeout"} else [])
