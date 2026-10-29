"""paced_map: a pool as wide as the memory allows, every task done once, workers recycled"""

from __future__ import annotations

import os

import pytest

from smolsmort.parallel.paced import GB, PacedError, paced_map, room_for_another


def square(value):
    return value * value


def pid_of(_):
    return os.getpid()


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


def test_room_counts_running_workers_growing_to_the_measured_peak():
    assert room_for_another(10 * GB, 2 * GB, [], 3 * GB)
    # two running workers still grow 2 gb each: 10 - 4 - 3 < 4
    assert not room_for_another(10 * GB, 4 * GB, [1 * GB, 1 * GB], 3 * GB)
    assert room_for_another(10 * GB, 4 * GB, [3 * GB, 3 * GB], 3 * GB)


def test_every_task_finishes_once_whatever_the_order():
    memory = FakeMemory(total=64 * GB, worker=1 * GB)
    results = dict(paced_map(square, range(8), reserve_gb=4, max_workers=3, memory=memory))
    assert results == {value: value * value for value in range(8)}
    assert 1 <= memory.widest <= 3


def test_the_pool_stays_as_narrow_as_the_memory_allows():
    # 4 gb over the reserve and 2 gb a worker: never more than two at once
    memory = FakeMemory(total=8 * GB, worker=2 * GB)
    results = dict(
        paced_map(square, range(6), reserve_gb=4, max_workers=8, worker_gb=2, memory=memory)
    )
    assert len(results) == 6
    assert memory.widest <= 2


def test_one_task_always_runs_when_memory_is_short():
    memory = FakeMemory(total=1 * GB, worker=2 * GB)
    results = dict(paced_map(square, range(3), reserve_gb=4, max_workers=4, memory=memory))
    assert results == {0: 0, 1: 1, 2: 4}
    assert memory.widest == 1


def test_a_worker_is_replaced_after_its_task_cap():
    memory = FakeMemory(total=64 * GB, worker=1 * GB)
    pids = [
        pid
        for _, pid in paced_map(
            pid_of, range(4), reserve_gb=1, max_workers=1, recycle_after=1, memory=memory
        )
    ]
    assert len(set(pids)) == 4


def test_a_failing_task_stops_the_pool_with_its_traceback():
    memory = FakeMemory(total=64 * GB, worker=1 * GB)
    with pytest.raises(PacedError, match="three"):
        list(paced_map(fail_on_three, range(6), reserve_gb=1, max_workers=2, memory=memory))
