"""decoded training frames kept on disk, so a frame is decoded once and not once per epoch, run and
setting.

a stored frame is the net's input as uint8, channels-first, one `.npy` per frame per downscale and
capture width. training reads a batch's frames from disk as the batch is built and lets them go
after it (`fill_paths`, `read_stored`): the os file cache keeps hot ones, and a run's resident
memory stays a few batches, not the whole store; `stored_frame` still maps one frame. its name
hashes the source's path, size and mtime with the settings, so a changed frame or setting never
reads an old one
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

# bumped when a stored frame's meaning changes, so old files are never read as new ones
STORE_VERSION = 1


def store_key(path: Path, downscale: int, capture_width: int | None) -> str:
    """the stored frame's name: the source file's identity and the settings it was decoded with"""
    source = Path(path).resolve()
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
    target = Path(root) / f"{store_key(path, downscale, capture_width)}.npy"
    if not target.exists():
        pixels = decode(path, downscale, capture_width)
        target.parent.mkdir(parents=True, exist_ok=True)
        # written aside, then moved in whole: a stopped run never leaves half a frame behind
        partial = target.with_name(f"{target.stem}.{os.getpid()}.partial.npy")
        np.save(partial, pixels)
        os.replace(partial, target)
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
