"""a process pool whose width follows the machine's available memory, not a fixed count.

cpu-bound tasks want every core busy, but memory has a cliff: past it the os pages to disk and
everything slows, or a supervisor stops the job. a fixed worker count either leaves cores idle or
runs into that cliff when tasks grow. `paced_map` starts one worker, measures how much memory a
worker really takes over a task, and widens only while the available memory stays above
`reserve_gb` after every running worker reaches that size; under pressure it starts nothing new,
and running tasks finishing shrink the pool. a worker that outgrows the others is replaced after
its task, which returns its memory.

workers are spawned on every os, so a run behaves the same on mac, linux and windows. the memory
source is injected (`memory`), so tests drive the width without touching the machine
"""

from __future__ import annotations

import contextlib
import multiprocessing
import statistics
import traceback
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from multiprocessing.connection import wait
from typing import Any

GB = 2**30

# a reading of the machine: available bytes, and each worker's resident bytes by pid
MemorySource = Callable[[list[int]], tuple[int, dict[int, int]]]


class PacedError(RuntimeError):
    """a task raised in its worker; carries the worker's traceback text"""


def psutil_memory(pids: list[int]) -> tuple[int, dict[int, int]]:
    """available memory (not free: on macos free sits near zero under file cache) and each
    worker's resident size, from psutil (the `parallel` extra)"""
    import psutil

    rss = {}
    for pid in pids:
        try:
            rss[pid] = psutil.Process(pid).memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return psutil.virtual_memory().available, rss


def default_workers() -> int:
    """the widest the pool goes: the machine's physical cores, else its logical ones"""
    from smolsmort.forecast.machine import describe_machine

    machine = describe_machine(nthread=None, max_concurrent_families=None)
    return max(1, machine.get("physical_cores") or machine.get("logical_cores") or 1)


def room_for_another(
    available: int, reserve: int, running_rss: list[int], worker_peak: int
) -> bool:
    """whether one more worker fits: the memory left after every running worker grows to the
    measured peak, and the new one does too, stays above the reserve"""
    growth = sum(max(0, worker_peak - rss) for rss in running_rss)
    return available - growth - worker_peak >= reserve


def _work(conn, fn) -> None:
    """a worker: run tasks sent down `conn` until told to stop"""
    while True:
        message = conn.recv()
        if message is None:
            return
        index, task = message
        try:
            conn.send((index, True, fn(task)))
        # any task failure goes back to the caller as text, the worker stays up
        except Exception:
            conn.send((index, False, traceback.format_exc()))


@dataclass
class _Worker:
    process: Any
    conn: Any
    done: int = 0
    task: int | None = None
    peak: int = 0
    peaks: list[int] = field(default_factory=list)


def paced_map(
    fn: Callable[[Any], Any],
    tasks: Iterable[Any],
    *,
    reserve_gb: float,
    max_workers: int | None = None,
    worker_gb: float = 1.0,
    recycle_after: int | None = None,
    grow_share: float = 1.5,
    poll_s: float = 2.0,
    memory: MemorySource | None = None,
) -> Iterator[tuple[Any, Any]]:
    """(task, fn(task)) as each finishes, in finishing order. `fn` must be picklable (a module
    level function). `worker_gb` is the guess for a worker's size until one is measured;
    `recycle_after` caps tasks per worker; a worker whose peak passes `grow_share` x the median
    measured peak is replaced after its task. a task's exception stops the pool and raises
    PacedError with the worker's traceback"""
    pending = list(enumerate(tasks))
    pending.reverse()
    inputs = dict(pending)
    widest = max_workers or default_workers()
    read = memory or psutil_memory
    reserve = int(reserve_gb * GB)
    context = multiprocessing.get_context("spawn")
    workers: list[_Worker] = []
    measured: list[int] = []

    def start() -> _Worker:
        parent, child = context.Pipe()
        process = context.Process(target=_work, args=(child, fn), daemon=True)
        process.start()
        child.close()
        worker = _Worker(process, parent)
        workers.append(worker)
        return worker

    def retire(worker: _Worker) -> None:
        with contextlib.suppress(BrokenPipeError, OSError):
            worker.conn.send(None)
        worker.process.join(timeout=10)
        if worker.process.is_alive():
            worker.process.terminate()
        worker.conn.close()
        workers.remove(worker)

    def assign(worker: _Worker) -> None:
        index, task = pending.pop()
        worker.task, worker.peak = index, 0
        worker.conn.send((index, task))

    try:
        while pending or any(w.task is not None for w in workers):
            available, rss = read([w.process.pid for w in workers])
            for worker in workers:
                worker.peak = max(worker.peak, rss.get(worker.process.pid, 0))
            peak = int(statistics.median(measured)) if measured else int(worker_gb * GB)
            # idle workers take work while memory allows; a new one starts while there is room
            for worker in [w for w in workers if w.task is None]:
                if not pending:
                    break
                busy = [rss.get(w.process.pid, 0) for w in workers if w.task is not None]
                # one task always runs, so a tight machine slows down instead of stalling
                if not busy or room_for_another(available, reserve, busy, peak):
                    assign(worker)
            while pending and len(workers) < widest:
                busy = [rss.get(w.process.pid, 0) for w in workers if w.task is not None]
                if workers and not room_for_another(available, reserve, busy, peak):
                    break
                assign(start())
                available -= peak
            for worker in [w for w in workers if w.task is None]:
                if not pending:
                    retire(worker)
            ready = wait([w.conn for w in workers if w.task is not None], timeout=poll_s)
            for worker in [w for w in workers if w.conn in ready]:
                index, ok, value = worker.conn.recv()
                if not ok:
                    raise PacedError(f"task {index} failed in its worker:\n{value}")
                worker.task, worker.done = None, worker.done + 1
                if worker.peak:
                    measured.append(worker.peak)
                    worker.peaks.append(worker.peak)
                yield inputs.pop(index), value
                grown = measured and worker.peak > grow_share * statistics.median(measured)
                if (recycle_after and worker.done >= recycle_after) or grown:
                    retire(worker)
    finally:
        for worker in list(workers):
            worker.process.terminate()
            worker.conn.close()
            workers.remove(worker)
