"""isolated exit fits; only finishing and display code consume their artifacts"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np

from smolsmort.forecast.family_worker import read_json
from smolsmort.forecast.pipeline import finish, genome_from_dict, load_workspace
from smolsmort.forecast.processes import watch_parent
from smolsmort.forecast.spec import spec_from_dict
from smolsmort.forecast.worker import _Run, _write_tables

EXIT_THREADS = 1
EXIT_TIMEOUT_S = 300


def restore_exit_cache(target, genome, prediction_cache, importance_cache):
    """hydrate the same mask-keyed cache finish uses, only for the saved recipe"""
    marker = read_json(target / "exit_test.json", {})
    if marker.get("recipe_key") != genome.key():
        return False
    relative = marker.get("tables_path", f"exit_scores/{genome.key()}")
    cache = read_json(target / relative / "cache.json", {})
    if cache.get("recipe_key") != genome.key():
        return False
    restored = {}
    try:
        for item in cache["predictions"]:
            key = (
                genome.key(),
                tuple(item["bucket"]),
                bytes.fromhex(item["fit_rows"]),
                bytes.fromhex(item["eval_rows"]),
            )
            restored[key] = np.asarray(item["prediction"], dtype=float)
    except (KeyError, TypeError, ValueError):
        return False
    prediction_cache.update(restored)
    importance_cache[genome.key()] = cache.get("importance", [])
    return True


def score_exit(folder, name, entry, *, nthread=EXIT_THREADS):
    request = read_json(folder / "request.json")
    spec = spec_from_dict(request["spec"])
    if spec.mode != "timeseries":
        return
    ws = load_workspace(
        spec,
        Path(request["prepared"]),
        request["summary"],
        transform=request.get("transform", "none"),
        holdout=request.get("holdout", "block"),
        horizon_mode=request.get("horizon_mode", "mixed"),
    )
    genome = genome_from_dict(entry["genome"], spec=spec)
    predictions, importance = {}, {}
    final = finish(
        ws,
        genome,
        nthread=nthread,
        validation=entry.get("prediction"),
        prediction_cache=predictions,
        importance_cache=importance,
    )
    target = folder / "families" / name
    cache_root = target / "exit_scores" / genome.key()
    cache_root.mkdir(parents=True, exist_ok=True)
    tables = Path(tempfile.mkdtemp(prefix="score-", dir=cache_root))
    relative = tables.relative_to(target)
    _write_tables(tables, spec, final)
    _Run(tables).save(
        "cache.json",
        {
            "recipe_key": genome.key(),
            "predictions": [
                {
                    "bucket": bucket,
                    "fit_rows": fit_rows.hex(),
                    "eval_rows": eval_rows.hex(),
                    "prediction": prediction,
                }
                for (_, bucket, fit_rows, eval_rows), prediction in predictions.items()
            ],
            "importance": importance.get(genome.key(), []),
        },
    )
    # publish tables before the marker so polling never observes a partial exit
    for source in tables.glob("*.parquet"):
        temporary = target / f"{source.name}.exit.tmp"
        temporary.write_bytes(source.read_bytes())
        temporary.replace(target / source.name)
    _Run(target).save(
        "exit_test.json",
        {
            "recipe_key": genome.key(),
            "tables_path": relative.as_posix(),
            "scored_at": time.time(),
            "model_error": final["model_error"],
            "baseline_error": final["baseline_error"],
            "verdict": final["verdict"],
        },
    )


class ExitScorer:
    """one low-thread worker alongside search; polls never wait for fitting"""

    def __init__(self, folder, *, enabled):
        self.folder = folder
        self.enabled = enabled
        self.deadline = float("inf")
        self.pending = OrderedDict()
        self.seen = set()
        self.completed = set()
        self.process = None
        self.process_token = None
        self.started = None

    def enqueue(self, name, entry, event):
        if not self.enabled or entry is None:
            return False
        token = (name, genome_from_dict(entry["genome"]).key(), event)
        if token in self.seen:
            return False
        self.seen.add(token)
        self.pending[name] = (entry, token)
        return True

    def poll(self):
        if not self.enabled:
            return
        if self.process is not None:
            expired = time.time() >= min(self.deadline, self.started + EXIT_TIMEOUT_S)
            result = self.process.poll()
            if result is None and not expired:
                return
            if result == 0:
                self.completed.add(self.process_token)
            self.stop()
        if not self.pending or time.time() >= self.deadline:
            return
        name, (entry, token) = self.pending.popitem(last=False)
        target = self.folder / "families" / name
        target.mkdir(parents=True, exist_ok=True)
        _Run(target).save("exit_request.json", entry)
        env = dict(os.environ)
        for key in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            env[key] = str(EXIT_THREADS)
        with (target / "exit_test.log").open("a") as log:
            self.process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "smolsmort.forecast.exit_test",
                    str(self.folder),
                    name,
                    str(os.getpid()),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
            )
        self.started = time.time()
        self.process_token = token

    def stop(self):
        if self.process is not None:
            from smolsmort.forecast.coordinator import stop_process

            stop_process(self.process)
            self.process = None

    def close(self):
        self.pending.clear()
        self.stop()


if __name__ == "__main__":
    folder, name, parent = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
    watch_parent(parent, stop_path=folder / "cancel")
    score_exit(folder, name, read_json(folder / "families" / name / "exit_request.json"))
