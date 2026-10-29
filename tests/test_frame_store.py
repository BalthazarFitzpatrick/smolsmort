"""decoded frames held as uint8 and stored on disk: the same values the float path gives, each
frame decoded once, a changed source never read from an old entry"""

from __future__ import annotations

import os

import numpy as np

from smolsmort.detect import frame_store
from smolsmort.detect.dataset import Example
from smolsmort.detect.frame_store import fill_paths, fill_store, store_key, stored_frame
from smolsmort.detect.train import _load_input, _load_pixels, train


def _frame(path, height=405, width=720, seed=0):
    from PIL import Image

    rng = np.random.default_rng(seed)
    Image.fromarray(rng.integers(0, 255, (height, width, 3), dtype=np.uint8)).save(path)
    return path


def test_uint8_pixels_give_the_float_inputs_exactly(tmp_path):
    path = _frame(tmp_path / "frame.png", 810, 1440)
    for downscale, capture in ((2, 1440), (4, 1440), (2, 1024)):
        pixels = _load_pixels(path, downscale, capture)
        assert pixels.dtype == np.uint8
        np.testing.assert_array_equal(
            pixels.astype(np.float32) / 255.0, _load_input(path, downscale, capture)
        )


def test_a_stored_frame_is_decoded_once_and_read_back_mapped(tmp_path):
    path = _frame(tmp_path / "frame.png")
    calls = []

    def decode(*args):
        calls.append(args)
        return _load_pixels(*args)

    first = stored_frame(tmp_path / "store", path, 2, None, decode)
    again = stored_frame(tmp_path / "store", path, 2, None, decode)
    assert len(calls) == 1 and isinstance(again, np.memmap)
    np.testing.assert_array_equal(first, again)
    assert not list((tmp_path / "store").glob("*.partial.npy"))


def test_a_changed_source_or_setting_takes_a_new_entry(tmp_path):
    path = _frame(tmp_path / "frame.png")
    key = store_key(path, 2, None)
    assert store_key(path, 4, None) != key and store_key(path, 2, 1440) != key
    _frame(path, seed=1)
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    assert store_key(path, 2, None) != key


def test_fill_store_maps_every_path_once(tmp_path):
    paths = [_frame(tmp_path / f"frame_{i}.png", seed=i) for i in range(3)]
    frames = fill_store(tmp_path / "store", paths + paths[:1], 2, None, _load_pixels, workers=2)
    assert list(frames) == paths
    assert len(list((tmp_path / "store").glob("*.npy"))) == 3


def test_training_from_the_store_matches_training_from_ram(tmp_path):
    examples = [
        Example(
            path=_frame(tmp_path / f"frame_{i}.png", 810, 1440, seed=i), centres=[(700.0, 200.0)]
        )
        for i in range(3)
    ]
    _, in_ram = train(examples, epochs=2, batch=4, capture_width=1440, seed=3)
    _, stored = train(
        examples, epochs=2, batch=4, capture_width=1440, seed=3, frame_store=tmp_path / "store"
    )
    np.testing.assert_allclose(stored, in_ram, rtol=1e-5)


def test_the_batch_feed_trains_the_same_whatever_its_thread_count(tmp_path):
    examples = [
        Example(
            path=_frame(tmp_path / f"frame_{i}.png", 810, 1440, seed=i), centres=[(700.0, 200.0)]
        )
        for i in range(4)
    ]
    runs = [
        train(examples, epochs=2, batch=2, capture_width=1440, seed=5, feed_workers=workers)[1]
        for workers in (1, 3)
    ]
    np.testing.assert_allclose(runs[0], runs[1], rtol=1e-6)


def test_fill_paths_stores_every_frame_and_maps_none(tmp_path):
    paths = [_frame(tmp_path / f"frame_{i}.png", seed=i) for i in range(3)]
    files = fill_paths(tmp_path / "store", paths + paths[:1], 2, None, _load_pixels, workers=2)
    assert list(files) == paths
    assert all(f.suffix == ".npy" and f.exists() for f in files.values())
    read = frame_store.read_stored(files[paths[0]])
    assert type(read) is np.ndarray
    np.testing.assert_array_equal(
        read, stored_frame(tmp_path / "store", paths[0], 2, None, _load_pixels)
    )


def test_a_stored_run_reads_each_batch_from_disk_and_maps_nothing(tmp_path, monkeypatch):
    examples = [
        Example(
            path=_frame(tmp_path / f"frame_{i}.png", 810, 1440, seed=i), centres=[(700.0, 200.0)]
        )
        for i in range(3)
    ]
    reads = []
    real = frame_store.read_stored

    def counted(target):
        reads.append(target)
        return real(target)

    import smolsmort.detect.train as trainer

    monkeypatch.setattr(trainer, "read_stored", counted)
    monkeypatch.setattr(trainer, "fill_store", None, raising=False)
    train(examples, epochs=2, batch=4, capture_width=1440, seed=3, frame_store=tmp_path / "store")
    # every window of every batch read its frame from disk: 2 epochs, 3 // 4 -> 1 batch of 4
    assert len(reads) == 2 * 4


def test_a_stored_frame_outlives_its_deleted_source(tmp_path):
    path = _frame(tmp_path / "frame.png", 810, 1440)
    first = stored_frame(tmp_path / "store", path, 2, 1440, _load_pixels)
    key = store_key(path, 2, 1440)
    assert frame_store.sidecar_of(path).is_file()
    path.unlink()

    def decode(*args):
        raise AssertionError("a stored frame is never decoded again")

    assert frame_store.source_present(path)
    assert frame_store.source_size(path) == (1440, 810)
    assert store_key(path, 2, 1440) == key
    np.testing.assert_array_equal(stored_frame(tmp_path / "store", path, 2, 1440, decode), first)


def test_a_deleted_source_never_stored_at_a_setting_is_refused(tmp_path):
    import pytest

    path = _frame(tmp_path / "frame.png")
    stored_frame(tmp_path / "store", path, 2, None, _load_pixels)
    path.unlink()
    with pytest.raises(FileNotFoundError):
        store_key(path, 4, None)
    with pytest.raises(FileNotFoundError):
        store_key(tmp_path / "never.png", 2, None)
    assert not frame_store.source_present(tmp_path / "never.png")


def test_training_after_deleting_the_stored_sources_matches_training_with_them(tmp_path):
    examples = [
        Example(
            path=_frame(tmp_path / f"frame_{i}.png", 810, 1440, seed=i), centres=[(700.0, 200.0)]
        )
        for i in range(3)
    ]
    store = tmp_path / "store"
    _, kept = train(examples, epochs=2, batch=4, capture_width=1440, seed=3, frame_store=store)
    for example in examples:
        example.path.unlink()
    _, gone = train(examples, epochs=2, batch=4, capture_width=1440, seed=3, frame_store=store)
    np.testing.assert_allclose(gone, kept, rtol=1e-6)
