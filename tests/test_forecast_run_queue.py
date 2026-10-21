"""real worker processes serialize before loading data or starting their budget"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from smolsmort.forecast import runs
from smolsmort.forecast.prep import Prepared
from smolsmort.forecast.processes import is_process_alive
from smolsmort.forecast.run_lock import RunLock
from smolsmort.forecast.spec import Column, PrepSpec

_WORKER = """
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from smolsmort.forecast import launcher, worker

folder = Path(sys.argv[1])

def load_workspace(spec, prepared, summary, **kwargs):
    (folder / "load.started").write_text(str(time.time()))
    return SimpleNamespace(mode="series", profile={}, excluded_series=[], hold=summary["hold"])

def screen_families(*args, **kwargs):
    (folder / "screen.started").write_text(str(time.time()))
    return {}

def search(*args, **kwargs):
    return {"best": {"family": "snaive"}, "leaderboard": [], "population": [], "stopped": "done"}

def finish(ws, genome, **kwargs):
    (folder / "work.started").write_text(str(time.time()))
    deadline = time.monotonic() + 60
    while ws.hold and not (folder / "release").exists():
        if time.monotonic() > deadline:
            raise TimeoutError("test did not release the worker")
        time.sleep(0.02)
    (folder / "work.ended").write_text(str(time.time()))
    return {"verdict": {}}

def write_result(folder, spec, result, final):
    (folder / "recipe.json").write_text(json.dumps(result["best"]))
    (folder / "result.json").write_text(json.dumps(final))

worker.load_workspace = load_workspace
worker.screen_families = screen_families
worker.read_priors = lambda *args: None
worker.read_top_families = lambda *args: None
worker.search = search
launcher.launch_families = search
worker.genome_from_dict = lambda data: SimpleNamespace(to_dict=lambda: data)
worker.finish = finish
worker._write_result = write_result
raise SystemExit(worker.run(folder))
"""

_LOCK_CHILD = """
import sys
import time
from pathlib import Path

folder = Path(sys.argv[1])
(folder / "child.ready").touch()
deadline = time.monotonic() + 60
while not (folder / "child.stop").exists() and time.monotonic() < deadline:
    time.sleep(0.02)
"""

_LOCK_HOLDER = """
import os
import subprocess
import sys
import time
from pathlib import Path

from smolsmort.forecast.run_lock import RunLock

folder = Path(sys.argv[1])
mode = sys.argv[2]
lock = RunLock()
assert lock.try_acquire()
if mode == "fork":
    child_pid = os.fork()
    if child_pid == 0:
        try:
            exec(sys.argv[3])
        finally:
            os._exit(0)
else:
    child = subprocess.Popen(
        [sys.executable, "-c", sys.argv[3], str(folder)], close_fds=False
    )
    child_pid = child.pid
(folder / "child.pid").write_text(str(child_pid))
try:
    deadline = time.monotonic() + 30
    while not (folder / "child.ready").exists():
        if time.monotonic() > deadline:
            raise TimeoutError("child did not start")
        time.sleep(0.02)
    lock.close()
    (folder / "holder.closed").touch()
    while not (folder / "holder.stop").exists() and time.monotonic() < deadline:
        time.sleep(0.02)
finally:
    (folder / "child.stop").touch()
    if mode == "fork":
        os.waitpid(child_pid, 0)
    else:
        child.wait(timeout=5)
    lock.close()
"""


@pytest.fixture
def start_worker(tmp_path, monkeypatch):
    processes = []
    roots = tmp_path / "runs"
    spec = PrepSpec(
        str(tmp_path / "source.csv"),
        "series",
        "regression",
        (Column("day", "time"), Column("value", "target", aggregation="sum")),
    )

    def spawn_worker(command, **kwargs):
        assert command[1:3] == ["-m", "smolsmort.forecast.worker"]
        process = subprocess.Popen([command[0], "-c", _WORKER, command[-1]], **kwargs)
        processes.append((Path(command[-1]), process))
        return process

    monkeypatch.setattr(
        runs, "subprocess", SimpleNamespace(Popen=spawn_worker, STDOUT=subprocess.STDOUT)
    )

    def start(*, kind="v2", hold=False):
        prepared = Prepared(tmp_path / "prepared", "", {"hold": hold}, False)
        options = {}
        if kind != "v1":
            options.update(families=[{"name": "snaive"}], time_budget_s=10)
        if kind == "refit":
            options["recipe"] = {"family": "snaive", "transform": "none"}
        run_id = runs.start_run(roots, spec, prepared, nthread=1, **options)
        return roots / run_id, processes[-1][1]

    yield start
    for folder, process in processes:
        (folder / "release").touch()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        runs._PROCESSES.pop(process.pid, None)


def read_state(folder):
    return runs.run_state(folder.parent, folder.name)


def wait_for_state(folder, expected, timeout=30):
    deadline = time.monotonic() + timeout
    state = {}
    while time.monotonic() < deadline:
        state = read_state(folder)
        if state["state"] == expected:
            return state
        if state["state"] in ("done", "failed", "cancelled"):
            break
        time.sleep(0.02)
    pytest.fail(f"expected {expected}, got {state}; log: {(folder / 'worker.log').read_text()}")


def wait_for_file(folder, filename, process, timeout=30):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if (folder / filename).exists():
            return
        if process.poll() is not None:
            break
        time.sleep(0.02)
    pytest.fail(f"missing {filename}; log: {(folder / 'worker.log').read_text()}")


def assert_not_started(folder):
    assert "started_at" not in json.loads((folder / "status.json").read_text())
    assert not (folder / "coordinator.json").exists()
    for filename in ("load.started", "screen.started", "work.started", "work.ended"):
        assert not (folder / filename).exists(), filename


@pytest.mark.parametrize("kind", ["v2", "v1", "refit"])
def test_runs_queue_before_loading_or_starting_budget(start_worker, kind):
    first, first_process = start_worker(hold=True)
    wait_for_file(first, "work.started", first_process)
    second, second_process = start_worker(kind=kind)
    queued = wait_for_state(second, "queued")
    assert queued["alive"]
    assert queued["queued_at"] <= time.time()
    if kind != "v1":
        assert queued["families"]["snaive"]["state"] == "pending"
    assert_not_started(second)
    assert read_state(first)["state"] == "running"

    (first / "release").touch()
    assert first_process.wait(timeout=30) == 0
    assert second_process.wait(timeout=30) == 0
    assert read_state(first)["state"] == "done"
    assert read_state(second)["state"] == "done"
    first_end = float((first / "work.ended").read_text())
    assert float((second / "load.started").read_text()) >= first_end
    if kind != "v1":
        coordinator = json.loads((second / "coordinator.json").read_text())
        assert coordinator["started_at"] >= first_end
        assert coordinator["deadline"] - coordinator["started_at"] == pytest.approx(10)
        assert coordinator["queued_s"] > 0
        assert coordinator["queued_s"] == pytest.approx(
            coordinator["started_at"] - queued["queued_at"], abs=0.1
        )


@pytest.mark.parametrize("cancel_method", ["file", "api"])
def test_cancelled_queued_run_never_starts(start_worker, cancel_method):
    first, first_process = start_worker(hold=True)
    wait_for_file(first, "work.started", first_process)
    second, second_process = start_worker()
    wait_for_state(second, "queued")
    if cancel_method == "file":
        (second / "cancel").touch()
    else:
        runs.cancel_run(second.parent, second.name)
    cancelled = wait_for_state(second, "cancelled")
    assert second_process.wait(timeout=5) == 0
    assert cancelled["families"]["snaive"]["state"] == "cancelled"
    assert_not_started(second)
    assert read_state(first)["state"] == "running"
    (first / "release").touch()
    assert first_process.wait(timeout=30) == 0
    assert read_state(first)["state"] == "done"
    assert_not_started(second)


@pytest.mark.skipif(sys.platform == "win32", reason="sigkill is a posix signal")
def test_sigkill_holder_releases_lock_for_next_run(start_worker):
    first, first_process = start_worker(hold=True)
    wait_for_file(first, "work.started", first_process)
    second, second_process = start_worker()
    wait_for_state(second, "queued")
    assert_not_started(second)
    first_process.send_signal(signal.SIGKILL)
    assert first_process.wait(timeout=5) == -signal.SIGKILL
    assert read_state(first)["state"] == "failed"
    assert second_process.wait(timeout=30) == 0
    assert read_state(second)["state"] == "done"
    assert (second / "work.ended").exists()


@pytest.mark.parametrize(
    "mode",
    [
        "exec",
        pytest.param(
            "fork",
            marks=pytest.mark.skipif(sys.platform == "win32", reason="fork is posix only"),
        ),
    ],
)
def test_live_child_does_not_retain_run_lock(tmp_path, mode):
    with (tmp_path / "worker.log").open("w") as log:
        process = subprocess.Popen(
            [sys.executable, "-c", _LOCK_HOLDER, str(tmp_path), mode, _LOCK_CHILD],
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            wait_for_file(tmp_path, "holder.closed", process)
            child_pid = int((tmp_path / "child.pid").read_text())
            assert process.poll() is None
            assert is_process_alive(child_pid)
            with RunLock() as lock:
                assert lock.try_acquire()
            assert is_process_alive(child_pid)
        finally:
            (tmp_path / "holder.stop").touch()
            (tmp_path / "child.stop").touch()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                if sys.platform == "win32":
                    process.kill()
                else:
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
        assert process.returncode == 0, (tmp_path / "worker.log").read_text()
