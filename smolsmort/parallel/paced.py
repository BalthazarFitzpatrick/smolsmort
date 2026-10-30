"""a spawned process pool that runs as wide as the machine allows and no wider.

cpu-bound tasks want every core busy, but a machine has two memory cliffs: the os pages to disk and
everything slows, and a supervisor watching free memory (claude code stops background jobs near
6.6 gb free) may stop the job. `paced_map` starts one worker and adds one at a time, each only after
the last has settled, while the memory left once every running worker reaches its measured peak
stays above a floor, the pagefile is not growing and the cores are not yet full. under pressure the
width narrows by one as running tasks finish. below an emergency line it stops the newest worker
and puts its task back, the one case where a running task is lost and redone.

workers are spawned on every os, so a run behaves the same on mac, linux and windows. the memory
source is injected (`memory`), so tests drive the width without touching the machine. a json
`control` file is read every poll, so the limits can be moved while a run is going.
"""

from __future__ import annotations

import contextlib
import json
import logging
import math
import multiprocessing
import statistics
import time
import traceback
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from multiprocessing.connection import wait
from pathlib import Path
from typing import Any

GB = 2**30
MB = 2**20


@dataclass
class MemoryReading:
    """available bytes, each worker's resident bytes by pid, cpu percent over all cores since the
    last reading and the pagefile's growth in mb a second"""

    available: int
    rss: dict[int, int]
    cpu: float = 0.0
    paging_mb_s: float = 0.0


MemorySource = Callable[[list[int]], MemoryReading | tuple[int, dict[int, int]]]


@dataclass(frozen=True)
class Limits:
    floor_gb: float
    emergency_gb: float
    max_workers: int
    cpu_full: float = 92.0
    page_high: float = 30.0
    page_hold_s: float = 8.0
    settle_s: float = 15.0
    step_s: float = 20.0


@dataclass(frozen=True)
class Gauge:
    """what the controller sees on one poll"""

    width: int
    limit: int
    available_gb: float
    growth_gb: float
    peak_gb: float
    cpu: float
    page_hot_s: float
    unsettled: bool
    since_change_s: float


def decide(gauge: Gauge, limits: Limits) -> tuple[int, bool]:
    """(the new width limit, whether to stop the newest worker now). widen by one only when the
    pool sits at its limit, every worker has settled, the memory still fits one more peak above
    the floor and the cores have room; narrow by one, not faster than step_s, when memory is under
    the floor or the pagefile keeps growing; below the emergency line shed one worker"""
    if gauge.available_gb < limits.emergency_gb and gauge.width > 1:
        return max(1, gauge.width - 1), True
    squeezed = gauge.available_gb < limits.floor_gb or gauge.page_hot_s >= limits.page_hold_s
    if squeezed:
        if gauge.since_change_s < limits.step_s:
            return gauge.limit, False
        return max(1, min(gauge.limit, gauge.width) - 1), False
    room = gauge.available_gb - gauge.growth_gb - gauge.peak_gb >= limits.floor_gb
    if (
        room
        and not gauge.unsettled
        and gauge.cpu < limits.cpu_full
        and gauge.width >= gauge.limit
        and gauge.since_change_s >= limits.settle_s
    ):
        return min(gauge.limit + 1, limits.max_workers), False
    return min(gauge.limit, limits.max_workers), False


class PacedError(RuntimeError):
    """a task raised in its worker; carries the worker's traceback text"""


class _Probe:
    """psutil readings: available memory (not free: on macos free sits near zero under file
    cache), workers' resident sizes, cpu since the last call and the pagefile's growth"""

    def __init__(self) -> None:
        self.swap_used: int | None = None
        self.at = time.monotonic()

    def __call__(self, pids: list[int]) -> MemoryReading:
        import psutil

        rss = {}
        for pid in pids:
            try:
                rss[pid] = psutil.Process(pid).memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        now, used = time.monotonic(), psutil.swap_memory().used
        grown = 0.0 if self.swap_used is None else max(0, used - self.swap_used) / MB
        elapsed = max(now - self.at, 1e-3)
        self.swap_used, self.at = used, now
        return MemoryReading(
            psutil.virtual_memory().available,
            rss,
            cpu=psutil.cpu_percent(interval=None),
            paging_mb_s=grown / elapsed,
        )


_PROBE = _Probe()


def psutil_memory(pids: list[int]) -> MemoryReading:
    """one reading of the machine from psutil (the `parallel` extra)"""
    return _PROBE(pids)


def default_workers() -> int:
    """the widest the pool goes: the machine's logical cores; the cpu gate stops it sooner"""
    from smolsmort.forecast.machine import describe_machine

    machine = describe_machine(nthread=None, max_concurrent_families=None)
    return max(1, machine.get("logical_cores") or machine.get("physical_cores") or 1)


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
    started: float = 0.0
    done: int = 0
    task: int | None = None
    peak: int = 0
    peaks: list[int] = field(default_factory=list)


CONTROL_KEYS = ("max_workers", "floor_gb", "emergency_gb", "cpu_full", "page_high")


def read_control(path: Path | None, limits: Limits) -> Limits:
    """the limits with any valid keys of the control file applied; a bad file changes nothing"""
    if path is None:
        return limits
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("control must be an object")
        changes = {}
        for key in CONTROL_KEYS:
            if key not in raw:
                continue
            value = raw[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{key} must be a number")
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{key} must be finite and not negative")
            if key == "max_workers" and (not isinstance(value, int) or value < 1):
                raise ValueError("max_workers must be a whole number from 1")
            changes[key] = value
    except (OSError, ValueError) as error:
        logging.getLogger(__name__).warning("ignored control file: %s", error)
        return limits
    return Limits(**{**limits.__dict__, **changes})


def paced_map(
    fn: Callable[[Any], Any],
    tasks: Iterable[Any],
    *,
    reserve_gb: float = 7.6,
    emergency_gb: float | None = None,
    max_workers: int | None = None,
    worker_gb: float = 1.0,
    recycle_after: int | None = None,
    grow_share: float = 1.5,
    poll_s: float = 2.0,
    memory: MemorySource | None = None,
    cpu_full: float = 92.0,
    page_high: float = 30.0,
    page_hold_s: float = 8.0,
    settle_s: float = 15.0,
    step_s: float = 20.0,
    control: Path | None = None,
    on_status: Callable[[dict], None] | None = None,
) -> Iterator[tuple[Any, Any]]:
    """(task, fn(task)) as each finishes, in finishing order. `fn` must be picklable (a module
    level function). `reserve_gb` is the floor of free memory the pool keeps; `emergency_gb`
    (default 0.6 gb under it) is where the newest worker is stopped and its task redone.
    `worker_gb` guesses a worker's size until one is measured; `recycle_after` caps tasks per
    worker; a worker whose peak passes `grow_share` x the median is replaced after its task. a
    task's exception stops the pool and raises PacedError with the worker's traceback"""
    pending = list(enumerate(tasks))
    pending.reverse()
    inputs = dict(pending)
    floor = reserve_gb
    limits = Limits(
        floor_gb=floor,
        emergency_gb=floor - 0.6 if emergency_gb is None else emergency_gb,
        max_workers=max_workers or default_workers(),
        cpu_full=cpu_full,
        page_high=page_high,
        page_hold_s=page_hold_s,
        settle_s=settle_s,
        step_s=step_s,
    )
    read = memory or psutil_memory
    context = multiprocessing.get_context("spawn")
    workers: list[_Worker] = []
    measured: list[int] = []
    limit, shed, last_change, last_shed = 1, 0, time.monotonic() - settle_s, 0.0
    hot_since: float | None = None

    def start() -> _Worker:
        parent, child = context.Pipe()
        process = context.Process(target=_work, args=(child, fn), daemon=True)
        process.start()
        child.close()
        worker = _Worker(process, parent, started=time.monotonic())
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

    def stop_newest() -> None:
        """the newest running worker is stopped and its task goes back to the front"""
        running = [w for w in workers if w.task is not None]
        victim = max(running, key=lambda w: w.started)
        pending.append((victim.task, inputs[victim.task]))
        victim.process.terminate()
        victim.process.join(timeout=10)
        victim.conn.close()
        workers.remove(victim)

    try:
        while pending or any(w.task is not None for w in workers):
            now = time.monotonic()
            live = read_control(control, limits)
            reading = read([w.process.pid for w in workers])
            if not isinstance(reading, MemoryReading):
                reading = MemoryReading(*reading)
            available, rss = reading.available, reading.rss
            for worker in workers:
                worker.peak = max(worker.peak, rss.get(worker.process.pid, 0))
            if reading.paging_mb_s > live.page_high:
                hot_since = now if hot_since is None else hot_since
            else:
                hot_since = None
            peak = max(measured[-8:]) if measured else int(worker_gb * GB)
            growth = sum(
                max(0, peak - rss.get(w.process.pid, 0)) for w in workers if w.task is not None
            )
            gauge = Gauge(
                width=len(workers),
                limit=limit,
                available_gb=available / GB,
                growth_gb=growth / GB,
                peak_gb=peak / GB,
                cpu=reading.cpu,
                page_hot_s=0.0 if hot_since is None else now - hot_since,
                unsettled=any(now - w.started < live.settle_s for w in workers),
                since_change_s=now - last_change,
            )
            new_limit, shed_now = decide(gauge, live)
            if shed_now and now - last_shed >= live.page_hold_s and len(workers) > 1:
                stop_newest()
                shed, last_shed = shed + 1, now
                new_limit = max(1, len(workers))
            elif shed_now:
                new_limit = limit
            if new_limit != limit:
                limit, last_change = new_limit, now
            if on_status:
                on_status(
                    {
                        "width": len(workers),
                        "limit": limit,
                        "running": sum(w.task is not None for w in workers),
                        "available_gb": available / GB,
                        "cpu": reading.cpu,
                        "paging_mb_s": reading.paging_mb_s,
                        "shed": shed,
                    }
                )
            # idle workers take work while the pool is inside its limit; the rest retire
            for worker in [w for w in workers if w.task is None]:
                if len(workers) > limit or not pending:
                    retire(worker)
                else:
                    assign(worker)
            while pending and len(workers) < limit:
                assign(start())
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
