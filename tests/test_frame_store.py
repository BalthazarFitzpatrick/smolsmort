"""decoded frames held as uint8 and stored on disk: the same values the float path gives, each
frame decoded once, a changed source never read from an old entry"""

from __future__ import annotations

import os

import numpy as np

from smolsmort.detect.dataset import Example
from smolsmort.detect.frame_store import fill_store, store_key, stored_frame
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
