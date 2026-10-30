"""paced_map: a pool as wide as the machine allows, every task done once, workers recycled"""

from __future__ import annotations

import json
import os
import time

import pytest

from smolsmort.parallel.paced import (
    GB,
    Gauge,
    Limits,
    PacedError,
    decide,
    paced_map,
    read_control,
    room_for_another,
)


def square(value):
    return value * value


def pid_of(_):
    return os.getpid()


def slow_square(value):
    time.sleep(0.3)
    return value * value


def fail_on_three(value):
    if value == 3:
        raise ValueError("three")
    return value


class FakeMemory:
    """a machine with `total` bytes, each worker taking `worker` bytes; records the widest pool"""

    def __init__(self, total, worker):
        self.total, self.worker, self.widest = total, worker, 0

    def __call__(self, pids):
        self.widest = max(self.widest, len(pids))
        return self.total - self.worker * len(pids), dict.fromkeys(pids, self.worker)


# a pool that widens on every poll, for tests that do not wait on a settle time
QUICK = {"settle_s": 0.0, "step_s": 0.0}
LIMITS = Limits(floor_gb=7.6, emergency_gb=7.0, max_workers=12)


def gauge(**changes):
    base = {
        "width": 4,
        "limit": 4,
        "available_gb": 12.0,
        "growth_gb": 0.0,
        "peak_gb": 1.2,
        "cpu": 60.0,
        "page_hot_s": 0.0,
        "unsettled": False,
        "since_change_s": 60.0,
    }
    return Gauge(**{**base, **changes})


def test_room_counts_running_workers_growing_to_the_measured_peak():
    assert room_for_another(10 * GB, 2 * GB, [], 3 * GB)
    # two running workers still grow 2 gb each: 10 - 4 - 3 < 4
    assert not room_for_another(10 * GB, 4 * GB, [1 * GB, 1 * GB], 3 * GB)
    assert room_for_another(10 * GB, 4 * GB, [3 * GB, 3 * GB], 3 * GB)


def test_it_widens_by_one_while_memory_and_cores_have_room():
    assert decide(gauge(), LIMITS) == (5, False)
    # 12 free - 1.2 for the new worker = 10.8, but 4 gb of growth still to come leaves 6.8 < 7.6
    assert decide(gauge(growth_gb=4.0), LIMITS) == (4, False)


def test_it_does_not_widen_into_full_cores_a_young_worker_or_the_cap():
    assert decide(gauge(cpu=95.0), LIMITS) == (4, False)
    assert decide(gauge(unsettled=True), LIMITS) == (4, False)
    assert decide(gauge(width=3, limit=4), LIMITS) == (4, False)
    assert decide(gauge(since_change_s=5.0), LIMITS) == (4, False)
    assert decide(gauge(width=12, limit=12), LIMITS) == (12, False)


def test_it_narrows_by_one_under_the_floor_but_not_faster_than_the_step():
    assert decide(gauge(available_gb=7.3), LIMITS) == (3, False)
    assert decide(gauge(available_gb=7.3, since_change_s=5.0), LIMITS) == (4, False)
    assert decide(gauge(available_gb=7.3, width=1, limit=1), LIMITS) == (1, False)


def test_a_pagefile_that_keeps_growing_narrows_the_pool():
    assert decide(gauge(page_hot_s=9.0), LIMITS) == (3, False)
    assert decide(gauge(page_hot_s=2.0), LIMITS) == (5, False)


def test_below_the_emergency_line_it_sheds_a_worker_but_never_the_last():
    assert decide(gauge(available_gb=6.5), LIMITS) == (3, True)
    assert decide(gauge(available_gb=6.5, width=1, limit=1), LIMITS) == (1, False)


def test_a_lowered_cap_pulls_the_limit_down():
    assert decide(gauge(width=8, limit=8), Limits(7.6, 7.0, max_workers=5)) == (5, False)


def test_every_task_finishes_once_whatever_the_order():
    memory = FakeMemory(total=64 * GB, worker=1 * GB)
    results = dict(paced_map(square, range(8), reserve_gb=4, max_workers=3, memory=memory, **QUICK))
    assert results == {value: value * value for value in range(8)}
    assert 1 <= memory.widest <= 3


def test_the_pool_stays_as_narrow_as_the_memory_allows():
    # 4 gb over the floor and 2 gb a worker: never more than two at once
    memory = FakeMemory(total=8 * GB, worker=2 * GB)
    results = dict(
        paced_map(
            slow_square,
            range(6),
            reserve_gb=4,
            max_workers=8,
            worker_gb=2,
            memory=memory,
            **QUICK,
        )
    )
    assert len(results) == 6
    assert memory.widest <= 2


def test_one_task_always_runs_when_memory_is_short():
    memory = FakeMemory(total=1 * GB, worker=2 * GB)
    results = dict(paced_map(square, range(3), reserve_gb=4, max_workers=4, memory=memory, **QUICK))
    assert results == {0: 0, 1: 1, 2: 4}
    assert memory.widest == 1


def test_a_worker_is_replaced_after_its_task_cap():
    memory = FakeMemory(total=64 * GB, worker=1 * GB)
    pids = [
        pid
        for _, pid in paced_map(
            pid_of,
            range(4),
            reserve_gb=1,
            max_workers=1,
            recycle_after=1,
            memory=memory,
            **QUICK,
        )
    ]
    assert len(set(pids)) == 4


def test_a_failing_task_stops_the_pool_with_its_traceback():
    memory = FakeMemory(total=64 * GB, worker=1 * GB)
    with pytest.raises(PacedError, match="three"):
        list(
            paced_map(fail_on_three, range(6), reserve_gb=1, max_workers=2, memory=memory, **QUICK)
        )


def test_an_emergency_stops_the_newest_worker_and_its_task_is_done_again():
    class Squeeze:
        """healthy until two workers run, then one reading far under the emergency line"""

        def __init__(self):
            self.squeezed = False

        def __call__(self, pids):
            if len(pids) >= 2 and not self.squeezed:
                self.squeezed = True
                return 1 * GB, dict.fromkeys(pids, 1 * GB)
            return 64 * GB, dict.fromkeys(pids, 1 * GB)

    seen = []
    results = dict(
        paced_map(
            slow_square,
            range(6),
            reserve_gb=4,
            max_workers=4,
            memory=Squeeze(),
            on_status=seen.append,
            **QUICK,
        )
    )
    assert results == {value: value * value for value in range(6)}
    assert max(status["shed"] for status in seen) == 1


def test_the_control_file_moves_the_cap_while_the_pool_runs(tmp_path):
    control = tmp_path / "control.json"
    control.write_text(json.dumps({"max_workers": 1}))
    memory = FakeMemory(total=64 * GB, worker=1 * GB)
    results = dict(
        paced_map(
            slow_square,
            range(4),
            reserve_gb=4,
            max_workers=4,
            memory=memory,
            control=control,
            **QUICK,
        )
    )
    assert len(results) == 4
    assert memory.widest == 1


def test_a_bad_control_file_changes_nothing(tmp_path):
    control = tmp_path / "control.json"
    control.write_text('{"max_workers": "many", "floor_gb": 9}')
    assert read_control(control, LIMITS) == LIMITS
    control.write_text("not json")
    assert read_control(control, LIMITS) == LIMITS
    assert read_control(tmp_path / "missing.json", LIMITS) == LIMITS
    control.write_text('{"floor_gb": 9, "max_workers": 3}')
    changed = read_control(control, LIMITS)
    assert (changed.floor_gb, changed.max_workers) == (9, 3)
