"""long allocations retain candidate scores across an actual worker kill"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from collections import Counter
from textwrap import dedent


def test_killed_long_run_worker_preserves_checkpointed_leaderboard(tmp_path):
    target = tmp_path / "families" / "ridge"
    target.mkdir(parents=True)
    request = {
        "version": 2,
        "spec": {
            "source": "synthetic.csv",
            "mode": "series",
            "task": "regression",
            "columns": [
                {"name": "day", "role": "time"},
                {"name": "value", "role": "target", "aggregation": "sum"},
            ],
        },
        "prepared": str(tmp_path),
        "summary": {},
        "families": [{"name": "ridge", "method": "grid", "space": {"alpha": [1, 2, 3, 4]}}],
        "time_budget_s": 86400,
        "nthread": 1,
    }
    started_at = time.time()
    allocation = {
        "parent_pid": os.getpid(),
        "started_at": started_at,
        "deadline": started_at + request["time_budget_s"],
        "slice_s": request["time_budget_s"],
        "phase": "deep",
        "nthread": 1,
    }
    (tmp_path / "request.json").write_text(json.dumps(request))
    (target / "slice.json").write_text(json.dumps(allocation))
    pause = tmp_path / "pause"
    pause.touch()
    runner = tmp_path / "synthetic_worker.py"
    runner.write_text(
        dedent(
            """\
            import importlib
            import json
            import sys
            import time
            from pathlib import Path
            from types import SimpleNamespace

            import numpy as np

            from smolsmort.forecast import family_worker

            folder = Path(sys.argv[1])
            target = folder / "families" / "ridge"
            allocation = json.loads((target / "slice.json").read_text())
            saved = family_worker.read_json(target / "checkpoint.json", {})
            search_module = importlib.import_module("smolsmort.forecast.search")

            class Clock:
                elapsed = saved.get("elapsed", 0)

                def time(self):
                    return allocation["started_at"] + self.elapsed

                def monotonic(self):
                    return self.elapsed

            clock = Clock()
            search_module.time = clock
            family_worker.time = clock
            workspace = SimpleNamespace(
                mode="series",
                spec=SimpleNamespace(task="regression"),
                series_masks={"one": {"train": np.ones(20, dtype=bool)}},
                families=["lag"],
                profile={"genes": {"objectives": ["squared"]}},
            )
            family_worker.load_workspace = lambda *args, **kwargs: workspace
            family_worker.estimate_cost = lambda *args: 1.0

            class SyntheticScorer:
                def __init__(self, *args, **kwargs):
                    pass

                def __enter__(self):
                    return self

                def __exit__(self, *args):
                    pass

                def score(self, genome, **kwargs):
                    calls = folder / "scored.jsonl"
                    completed = calls.read_text().splitlines() if calls.exists() else []
                    if len(completed) == 2 and (folder / "pause").exists():
                        (folder / "waiting").touch()
                        while True:
                            time.sleep(0.01)
                    clock.elapsed += 10800
                    with calls.open("a") as stream:
                        stream.write(json.dumps(genome.key()) + "\\n")
                    fitness = float(dict(genome.params)["alpha"])
                    return SimpleNamespace(
                        fitness=fitness,
                        contrib=np.array([fitness, fitness + 0.5]),
                        metrics={"synthetic": fitness},
                        eval_history=None,
                        prediction=np.array([fitness, fitness + 1]),
                        elapsed_s=10800,
                    )

            family_worker.CandidateScorer = SyntheticScorer
            real_search = family_worker.search

            def record_search(workspace, budget, **kwargs):
                with (folder / "budgets.jsonl").open("a") as stream:
                    stream.write(json.dumps({
                        "time_cap": budget.time_cap,
                        "deadline": kwargs["deadline"],
                        "prior_elapsed": (kwargs.get("resume") or {}).get("elapsed", 0),
                    }) + "\\n")
                return real_search(workspace, budget, **kwargs)

            family_worker.search = record_search
            raise SystemExit(family_worker.run(folder, "ridge"))
            """
        )
    )
    command = [sys.executable, str(runner), str(tmp_path)]
    checkpoint = target / "checkpoint.json"
    with (tmp_path / "worker.log").open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        try:
            until = time.monotonic() + 30
            while (
                not (tmp_path / "waiting").exists()
                and process.poll() is None
                and time.monotonic() < until
            ):
                time.sleep(0.02)
            assert (tmp_path / "waiting").exists(), (tmp_path / "worker.log").read_text()
            before = json.loads(checkpoint.read_text())
            assert len(before["entries"]) == 2
            assert before["elapsed"] == 6 * 3600
            process.kill()
            process.wait(timeout=5)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)
        assert json.loads(checkpoint.read_text()) == before
        pause.unlink()
        resumed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=30)
        assert resumed.returncode == 0, (tmp_path / "worker.log").read_text()

    after = json.loads(checkpoint.read_text())
    prior = {entry["key"]: entry for entry in before["entries"]}
    current = {entry["key"]: entry for entry in after["entries"]}
    assert len(current) == 4
    assert all(current[key] == entry for key, entry in prior.items())
    assert after["elapsed"] == 12 * 3600
    assert after["candidate_times"][:2] == before["candidate_times"]
    board = json.loads((target / "leaderboard.json").read_text())
    assert {entry["key"] for entry in board} == current.keys()
    for entry in board:
        if entry["key"] in prior:
            assert entry == {key: prior[entry["key"]][key] for key in entry}
    scored = [json.loads(line) for line in (tmp_path / "scored.jsonl").read_text().splitlines()]
    assert Counter(scored) == Counter(dict.fromkeys(current, 1))
    budgets = [json.loads(line) for line in (tmp_path / "budgets.jsonl").read_text().splitlines()]
    assert [budget["prior_elapsed"] for budget in budgets] == [0, 6 * 3600]
    assert all(budget["time_cap"] == 86400 for budget in budgets)
    assert all(budget["deadline"] == allocation["deadline"] for budget in budgets)
    assert json.loads((target / "slice.json").read_text()) == allocation
    status = json.loads((target / "status.json").read_text())
    assert status["state"] == "done"
    assert status["reason"] == "grid complete"
