"""bounded frame storage, unchanged pixels, and synthetic cpu training"""

from threading import Barrier, get_ident

import numpy as np
import pytest
from PIL import Image

from smolsmort.boxes import train as box_train
from smolsmort.boxes.model import scale_for
from smolsmort.detect.dataset import Example


@pytest.fixture
def frames(tmp_path):
    paths = []
    rng = np.random.default_rng(14)
    for index in range(6):
        path = tmp_path / f"frame-{index}.png"
        pixels = rng.integers(0, 256, (320 + index * 16, 480 + index * 16, 3), dtype=np.uint8)
        Image.fromarray(pixels).save(path)
        paths.append(path)
    return paths


def test_scale_uses_headers_without_decoding(frames, monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("pixels must not be decoded")

    monkeypatch.setattr(box_train, "load_input", refuse)
    monkeypatch.setattr(Image.Image, "load", refuse)
    cache = box_train.InputCache(frames, 256, 1)
    for index, path in enumerate(frames):
        assert cache.scale_of(path) == scale_for(480 + index * 16, 320 + index * 16, 256)
    assert cache.stored_bytes == 0


def test_lru_budget_and_recency(frames):
    cache = box_train.InputCache(frames, 480, 1)
    first, second, third = frames[:3]
    cache.get(first)
    cache.get(second)
    cache.get(first)
    cache.get(third)
    assert second not in cache._frames
    assert list(cache._frames) == [first, third]
    for path in frames:
        cache.get(path)
        assert cache.stored_bytes == sum(image.nbytes for image, _ in cache._frames.values())
        assert cache.stored_bytes <= cache.budget_bytes
        assert all(image.dtype == np.uint8 for image, _ in cache._frames.values())
    assert cache.peak_stored_bytes <= 1024 * 1024


@pytest.mark.parametrize("budget", [None, 0, 1])
def test_cache_matches_load_input(frames, budget):
    cache = box_train.InputCache(frames, 256, budget)
    for path in frames:
        expected, scale = box_train.load_input(path, 256)
        for _ in range(2):
            actual, actual_scale = cache.get(path)
            np.testing.assert_array_equal(actual, expected)
            assert actual.dtype == np.float32
            assert actual_scale == scale


def test_oversize_frame_is_not_retained(frames):
    cache = box_train.InputCache(frames, 1024, 1)
    actual, _ = cache.get(frames[0])
    assert actual.nbytes > cache.budget_bytes
    assert cache.stored_bytes == 0


def test_get_many_uses_pool_and_preserves_order(frames, monkeypatch):
    original = box_train.load_input_uint8
    barrier = Barrier(3)
    threads = set()

    def decode(path, long_side):
        threads.add(get_ident())
        barrier.wait(timeout=10)
        return original(path, long_side)

    monkeypatch.setattr(box_train, "load_input_uint8", decode)
    cache = box_train.InputCache(frames, 256, 1, workers=3)
    order = [frames[2], frames[0], frames[1], frames[2]]
    actual = cache.get_many(order)
    monkeypatch.setattr(box_train, "load_input_uint8", original)
    assert len(threads) == 3
    for path, (image, scale) in zip(order, actual, strict=True):
        expected, expected_scale = box_train.load_input(path, 256)
        np.testing.assert_array_equal(image, expected)
        assert scale == expected_scale


def test_reach_refusal_needs_only_headers(frames, monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("reach check must not decode")

    monkeypatch.setattr(box_train, "load_input_uint8", refuse)
    cache = box_train.InputCache(frames, 480, 1)
    example = Example(frames[0], centres=[(240, 160)], sizes=[(400, 300)])
    with pytest.raises(box_train.BoxTrainError, match="smaller long_side"):
        box_train._refuse_out_of_reach([example], cache.scale_of, 100)


def test_training_is_bounded_and_repeatable(frames, monkeypatch):
    torch = pytest.importorskip("torch")
    caches = []
    original = box_train.InputCache

    class TrackedCache(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            caches.append(self)

    monkeypatch.setattr(box_train, "InputCache", TrackedCache)
    examples = [Example(path, centres=[(240, 160)], sizes=[(24, 20)]) for path in frames]
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        histories = []
        for _ in range(2):
            _, history = box_train.train(
                examples,
                epochs=2,
                batch=2,
                steps_per_epoch=3,
                device="cpu",
                seed=12,
                cache_mb=1,
                long_side=480,
                crop=64,
                widths=(2, 4, 4, 8, 8),
            )
            histories.append(history)
    finally:
        torch.set_num_threads(previous_threads)
    assert len(histories[0]) == 2
    assert np.isfinite(histories).all()
    assert histories[0] == histories[1]
    for cache in caches:
        assert 0 < cache.peak_stored_bytes <= 1024 * 1024
        assert cache.peak_stored_bytes < 2 * 1024 * 1024
        assert len(cache._frames) < len(frames)


def test_backend_passes_cache_budget(monkeypatch):
    from smolsmort.boxes.backend import BoxBackend

    received = {}

    def train(examples, **kwargs):
        received.update(kwargs)
        return object(), []

    monkeypatch.setattr(box_train, "train", train)
    BoxBackend(cache_mb=17).train([], classes={})
    assert received["cache_mb"] == 17
