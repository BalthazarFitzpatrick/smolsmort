"""decoded training frames kept on disk, so a frame is decoded once and not once per epoch, run and
setting.

a stored frame is the net's input as uint8, channels-first, one `.npy` per frame per downscale and
capture width. training reads a batch's frames from disk as the batch is built and lets them go
after it (`fill_paths`, `read_stored`): the os file cache keeps hot ones, and a run's resident
memory stays a few batches, not the whole store; `stored_frame` still maps one frame. its name
hashes the source's path, size and mtime with the settings, so a changed frame or setting never
reads an old one. storing a frame also writes `<source>.stored.json` beside it (its size and store
keys), so a source deleted after storing still trains: only the stored frame is needed then
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

# bumped when a stored frame's meaning changes, so old files are never read as new ones
STORE_VERSION = 1


# beside each stored source: its size and store keys, read once the source itself is gone
SIDECAR_SUFFIX = ".stored.json"


def sidecar_of(path: Path) -> Path:
    """the file beside a source that outlives it once its frame is stored"""
    path = Path(path)
    return path.with_name(path.name + SIDECAR_SUFFIX)


def source_present(path: Path) -> bool:
    """the source is there, or was stored before it was deleted"""
    return Path(path).is_file() or sidecar_of(path).is_file()


def source_size(path: Path) -> tuple[int, int]:
    """(width, height) from the source's header, else from its sidecar"""
    path = Path(path)
    if path.is_file():
        from PIL import Image

        with Image.open(path) as handle:
            return handle.size
    width, height = _read_sidecar(path)["size"]
    return int(width), int(height)


def _settings(downscale: int, capture_width: int | None) -> str:
    return f"{downscale}|{capture_width}"


def _read_sidecar(path: Path) -> dict:
    sidecar = sidecar_of(path)
    if not sidecar.is_file():
        raise FileNotFoundError(f"{path}: no source and no stored frame record")
    return json.loads(sidecar.read_text(encoding="utf-8"))


def _note_stored(path: Path, key: str, downscale: int, capture_width: int | None) -> None:
    """record the source's size and this setting's key beside it, written aside then moved in"""
    sidecar = sidecar_of(path)
    record = json.loads(sidecar.read_text(encoding="utf-8")) if sidecar.is_file() else {}
    keys = record.setdefault("keys", {})
    if keys.get(_settings(downscale, capture_width)) == key and "size" in record:
        return
    keys[_settings(downscale, capture_width)] = key
    record["size"] = list(source_size(path))
    partial = sidecar.with_name(f"{sidecar.name}.{os.getpid()}.partial")
    partial.write_text(json.dumps(record), encoding="utf-8")
    os.replace(partial, sidecar)


def store_key(path: Path, downscale: int, capture_width: int | None) -> str:
    """the stored frame's name: the source file's identity and the settings it was decoded with;
    a deleted source answers from its sidecar"""
    source = Path(path).resolve()
    if not source.is_file():
        keys = _read_sidecar(source).get("keys", {})
        if _settings(downscale, capture_width) not in keys:
            raise FileNotFoundError(f"{path}: no source and no stored frame at these settings")
        return keys[_settings(downscale, capture_width)]
    stat = source.stat()
    identity = (
        f"{STORE_VERSION}|{source}|{stat.st_size}|{stat.st_mtime_ns}|{downscale}|{capture_width}"
    )
    return hashlib.sha1(identity.encode("utf-8")).hexdigest()


def stored_path(
    root: Path,
    path: Path,
    downscale: int,
    capture_width: int | None,
    decode: Callable[[Path, int, int | None], np.ndarray],
) -> Path:
    """the stored frame's file in `root`, decoded and written first when it is not there yet"""
    key = store_key(path, downscale, capture_width)
    target = Path(root) / f"{key}.npy"
    if not target.exists():
        pixels = decode(path, downscale, capture_width)
        target.parent.mkdir(parents=True, exist_ok=True)
        # written aside, then moved in whole: a stopped run never leaves half a frame behind
        partial = target.with_name(f"{target.stem}.{os.getpid()}.partial.npy")
        np.save(partial, pixels)
        os.replace(partial, target)
    if Path(path).is_file():
        _note_stored(path, key, downscale, capture_width)
    return target


def read_stored(target: Path) -> np.ndarray:
    """a stored frame read whole into memory, no mapping: freed once its batch is built"""
    return np.load(target)


def stored_frame(
    root: Path,
    path: Path,
    downscale: int,
    capture_width: int | None,
    decode: Callable[[Path, int, int | None], np.ndarray],
) -> np.ndarray:
    """the frame memory-mapped from `root`, decoded and written first when it is not there yet"""
    return np.load(stored_path(root, path, downscale, capture_width, decode), mmap_mode="r")


def fill_store(
    root: Path,
    paths: Iterable[Path],
    downscale: int,
    capture_width: int | None,
    decode: Callable[[Path, int, int | None], np.ndarray],
    workers: int | None = None,
) -> dict[Path, np.ndarray]:
    """every path's stored frame, the missing ones decoded on `workers` threads"""
    paths = list(dict.fromkeys(paths))
    with ThreadPoolExecutor(max_workers=workers or decode_workers()) as pool:
        frames = pool.map(
            lambda path: stored_frame(root, path, downscale, capture_width, decode), paths
        )
        return dict(zip(paths, frames, strict=True))


def fill_paths(
    root: Path,
    paths: Iterable[Path],
    downscale: int,
    capture_width: int | None,
    decode: Callable[[Path, int, int | None], np.ndarray],
    workers: int | None = None,
) -> dict[Path, Path]:
    """every path's stored file, the missing ones decoded on `workers` threads; nothing is mapped"""
    paths = list(dict.fromkeys(paths))
    with ThreadPoolExecutor(max_workers=workers or decode_workers()) as pool:
        files = pool.map(
            lambda path: stored_path(root, path, downscale, capture_width, decode), paths
        )
        return dict(zip(paths, files, strict=True))


def decode_workers() -> int:
    """threads that decode frames: every core but two, which feed the gpu and run the loop"""
    return max(2, (os.cpu_count() or 4) - 2)
