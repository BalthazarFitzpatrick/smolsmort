"""portable parent supervision and bounded cleanup of owned processes"""

from __future__ import annotations

import multiprocessing
import os
import sys
import threading
import time
from pathlib import Path


def stop_children():
    children = multiprocessing.active_children()
    for child in children:
        child.terminate()
    until = time.monotonic() + 0.2
    for child in children:
        child.join(timeout=max(0, until - time.monotonic()))
        if child.is_alive():
            child.kill()
    until = time.monotonic() + 0.2
    for child in children:
        child.join(timeout=max(0, until - time.monotonic()))


def watch_parent(parent_pid, *, stop_path=None):
    """the watchdog runs while a native fit blocks the scoring thread"""

    def watch():
        from smolsmort.forecast.runs import _windows_alive

        while True:
            alive = os.getppid() == parent_pid
            if alive and sys.platform == "win32":
                alive = _windows_alive(parent_pid)
            if not alive or (stop_path and Path(stop_path).exists()):
                stop_children()
                os._exit(1)
            time.sleep(0.02)

    threading.Thread(target=watch, daemon=True).start()
