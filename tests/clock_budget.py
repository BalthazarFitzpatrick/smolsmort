"""wall-clock budgets for tests that run real worker processes, stretched on small hosts"""

import os

REFERENCE_CORES = 8
MAX_STRETCH = 3.0


def scaled(seconds: float) -> int:
    """a run budget sized on an idle 8 core host, stretched up to 3x when the host has fewer
    cores, since the worker processes then share them (ci runners have 2 to 4)"""
    stretch = REFERENCE_CORES / (os.cpu_count() or REFERENCE_CORES)
    return int(seconds * min(max(stretch, 1.0), MAX_STRETCH))
