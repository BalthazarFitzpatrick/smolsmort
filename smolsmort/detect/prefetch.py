"""decode frames ahead of the model on worker threads, in order.

measured per frame at 3420x2224: 51 ms to decode and downscale the jpeg against 39 ms for the
forward pass, so more than half a sweep was the cpu waiting on pillow with the gpu idle. keeping a
few frames decoded ahead fills that gap; the window is bounded so a long sweep never holds every
decoded frame in memory at once.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import TypeVar

T = TypeVar("T")


def decode_ahead(
    items: Sequence, load: Callable[..., T], *, ahead: int = 3, workers: int = 2
) -> Iterator[tuple[object, T]]:
    """yield (item, load(item)) in order while up to `ahead` later items decode on `workers`"""
    pool = ThreadPoolExecutor(max_workers=workers)
    pending = {i: pool.submit(load, item) for i, item in enumerate(items[:ahead])}
    try:
        for index, item in enumerate(items):
            following = index + ahead
            if following < len(items):
                pending[following] = pool.submit(load, items[following])
            yield item, pending.pop(index).result()
    finally:
        pool.shutdown(wait=False)
