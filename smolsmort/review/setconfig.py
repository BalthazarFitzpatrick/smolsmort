"""per-set and per-recording settings that live in small json files beside the data.

a training set names its backend and box size mode in `<set>._backend.json` next to the set file.
a recording keeps its per-frame labelling mode in `<recording>._frames.json` next to its labels.
both are optional: an absent file means the safe default (heatmap + uniform, every frame explicit),
so every set and recording written before these files existed reads exactly as it did.

PATHS ARE READ THROUGH `paths`, never imported by value - see review/paths.py.
"""

from __future__ import annotations

import json
from pathlib import Path

from smolsmort import backends
from smolsmort.review import paths
from smolsmort.review.find_state import write_json_atomic
from smolsmort.review.naming import _flat, set_filename

SIZE_MODES = ("uniform", "native")
DEFAULT_BACKEND = backends.LEGACY
DEFAULT_SIZE_MODE = "uniform"


class SetConfigError(Exception):
    pass


def set_path(name: str) -> Path:
    """a set's file: a plain name, or a path relative to the sets folder. a path that would
    leave the folder raises ValueError, so a picker cannot bind something outside it"""
    if "/" not in name:
        return paths.DATASETS_DIR / f"{set_filename(name)}.jsonl"
    root = paths.DATASETS_DIR.resolve()
    target = (root / name).resolve()
    if root not in target.parents:
        raise ValueError(f"{name!r} is outside the sets folder")
    return target if target.suffix == ".jsonl" else target.with_name(target.name + ".jsonl")


def backend_file(name: str) -> Path:
    path = set_path(name)
    return path.with_name(f"{path.stem}._backend.json")


def read_set_config(name: str | None) -> dict | None:
    """the set's stored {backend, size_mode}, or None when it has no file (or no name)"""
    if not name:
        return None
    try:
        found = json.loads(backend_file(name).read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(found, dict):
        return None
    size_mode = found.get("size_mode")
    return {
        "backend": str(found.get("backend") or DEFAULT_BACKEND),
        "size_mode": size_mode if size_mode in SIZE_MODES else DEFAULT_SIZE_MODE,
    }


def set_config(name: str | None) -> dict:
    """what a set trains with: its file, else the legacy defaults (heatmap, uniform)"""
    return read_set_config(name) or {
        "backend": DEFAULT_BACKEND,
        "size_mode": DEFAULT_SIZE_MODE,
    }


def default_size_mode(backend: str) -> str:
    return "native" if backend == "box" else "uniform"


def write_set_config(name: str, backend: str, size_mode: str | None = None) -> dict:
    """persist a set's backend. an unknown backend or size mode raises SetConfigError"""
    known = backends.names()
    if backend not in known:
        raise SetConfigError(f"no backend called {backend!r} - known: {', '.join(known)}")
    mode = size_mode or default_size_mode(backend)
    if mode not in SIZE_MODES:
        raise SetConfigError(f"size_mode must be one of {', '.join(SIZE_MODES)}, got {mode!r}")
    value = {"backend": backend, "size_mode": mode}
    write_json_atomic(backend_file(name), value, indent=1)
    return value


# ---------------------------------------------------------------- per-frame labelling mode


def frames_file(recording: str) -> Path:
    return paths.LABELS_DIR / f"{_flat(recording)}._frames.json"


def read_frame_modes(recording: str) -> dict[str, dict]:
    """frame -> {"exhaustive": bool}, only for frames that have a stored mode"""
    try:
        found = json.loads(frames_file(recording).read_text())
    except (OSError, ValueError):
        return {}
    if not isinstance(found, dict):
        return {}
    return {
        str(frame): {"exhaustive": bool(value.get("exhaustive"))}
        for frame, value in found.items()
        if isinstance(value, dict)
    }


def exhaustive_frames(recording: str) -> set[str]:
    return {f for f, v in read_frame_modes(recording).items() if v["exhaustive"]}


def write_frame_mode(recording: str, frame: str, exhaustive: bool) -> None:
    """explicit is the default, so turning exhaustive off removes the frame's entry"""
    modes = read_frame_modes(recording)
    if exhaustive:
        modes[frame] = {"exhaustive": True}
    else:
        modes.pop(frame, None)
    write_json_atomic(frames_file(recording), modes, indent=1, sort_keys=True)
