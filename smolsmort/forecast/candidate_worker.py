"""killable candidate scoring; only the family search writes checkpoints"""

from __future__ import annotations

import multiprocessing
import os
import time
from contextlib import suppress
from pathlib import Path

import numpy as np

from smolsmort.forecast.families import get_family
from smolsmort.forecast.model import ModelError
from smolsmort.forecast.pipeline import PipelineError, score
from smolsmort.forecast.processes import watch_parent

# spawn imports this module in statsforecast's children, which also follow their parent
if parent_pid := os.environ.get("SMOLSMORT_CANDIDATE_PARENT"):
    watch_parent(int(parent_pid))


class CandidateTimeout(ModelError):
    too_slow = True

    def __init__(self, elapsed_s):
        super().__init__("too slow")
        self.elapsed_s = elapsed_s


def score_candidates(connection, ws, parent_pid, score_fn, target):
    multiprocessing.set_start_method("spawn", force=True)
    os.environ["SMOLSMORT_CANDIDATE_PARENT"] = str(os.getpid())
    stop_path = target / f"candidate-stop-{os.getpid()}"
    watch_parent(parent_pid, stop_path=stop_path)
    connection.send("ready")
    try:
        while (task := connection.recv()) is not None:
            genome, nthread, sample = task
            started = time.monotonic()
            try:
                result = score_fn(ws, genome, nthread=nthread, sample=sample)
            except (ModelError, PipelineError) as exc:
                connection.send((None, str(exc), time.monotonic() - started))
            else:
                connection.send((result, None, time.monotonic() - started))
    except (EOFError, BrokenPipeError):
        pass
    finally:
        connection.close()


def estimate_cost(ws, genome):
    family = get_family(genome.family)
    params = dict(genome.params)
    if ws.mode == "row":
        columns = sum(name in genome.families for name in ws.features.families)
        return max(float(family.cost((int(ws.masks["train"].sum()), columns), params)), 1e-9)
    hints = []
    for bucket, frame in ws.frames.items():
        if family.needs == "raw_series":
            shape = (len(frame.y), len(np.unique(frame.series)))
        else:
            columns = sum(name in genome.families for name in frame.features.families)
            shape = (int(ws.series_masks[bucket]["train"].sum()), columns)
        hints.append(family.cost(shape, params))
    return max(float(sum(hints)), 1e-9)


class CandidateScorer:
    def __init__(
        self, ws, target, *, deadline, slice_s, fraction=0.25, stop=lambda: False, score_fn=score
    ):
        self.ws = ws
        self.target = target
        self.deadline = deadline
        self.limit_s = slice_s * fraction
        self.stop = stop
        self.score_fn = score_fn
        self.process = None
        self.connection = None
        self.busy = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        if self.process is not None:
            if self.process.is_alive() and not self.busy and self.connection is not None:
                with suppress(BrokenPipeError, EOFError, OSError):
                    self.connection.send(None)
                self.process.join(timeout=0.1)
            if self.process.is_alive():
                (self.target / f"candidate-stop-{self.process.pid}").touch()
                self.process.join(timeout=0.5)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=0.2)
            if self.process.is_alive():
                self.process.kill()
                self.process.join(timeout=0.2)
            self.process.join(timeout=0)
            self.process.close()
            self.process = None
            self.busy = False
        if self.connection is not None:
            self.connection.close()
            self.connection = None

    def receive(self, until):
        while time.time() < until and not self.stop():
            if self.connection.poll(min(0.02, max(0, until - time.time()))):
                try:
                    return self.connection.recv()
                except (EOFError, OSError) as exc:
                    raise ModelError("candidate process exited without a score") from exc
            if not self.process.is_alive():
                raise ModelError("candidate process exited without a score")
        return None

    def start(self):
        from smolsmort.forecast.worker import _Run

        context = multiprocessing.get_context("spawn")
        self.connection, child_connection = context.Pipe()
        self.process = context.Process(
            target=score_candidates,
            args=(child_connection, self.ws, os.getpid(), self.score_fn, Path(self.target)),
        )
        self.process.start()
        child_connection.close()
        _Run(self.target).save("candidate_pid.json", {"pid": self.process.pid})
        if self.receive(self.deadline) != "ready":
            self.close()
            raise CandidateTimeout(0)

    def score(self, genome, *, nthread=1, sample=None):
        if time.time() >= self.deadline or self.stop():
            raise CandidateTimeout(0)
        started = time.monotonic()
        try:
            if self.process is None:
                self.start()
            started = time.monotonic()
            self.busy = True
            self.connection.send((genome, nthread, sample))
            until = min(self.deadline, time.time() + self.limit_s)
            message = self.receive(until)
        except (OSError, ModelError) as exc:
            self.close()
            if isinstance(exc, ModelError):
                raise
            raise ModelError("candidate process exited without a score") from exc
        if message is None:
            elapsed_s = time.monotonic() - started
            self.close()
            raise CandidateTimeout(elapsed_s)
        result, error, elapsed_s = message
        self.busy = False
        if error:
            raise ModelError(error)
        result.elapsed_s = elapsed_s
        return result
