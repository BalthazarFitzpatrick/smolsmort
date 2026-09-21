"""the shared window policy reproduces what each backend's own sampler produced before it moved:
tests/golden/window_sampling.json was written by the old _crop_window and _window with these
seeds and examples, and both must still land the same windows, targets and masks"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pytest

from smolsmort.detect.dataset import Example

torch = pytest.importorskip("torch")

GOLDEN = json.loads((Path(__file__).parent / "golden" / "window_sampling.json").read_text())


def _examples():
    out = []
    for i, (w, h) in enumerate(((320, 200), (256, 256), (400, 180))):
        rng = np.random.default_rng(i)
        centres = [(60.0 + 40 * i, 50.0), (200.0, 120.0 + 10 * i)]
        ex = Example(
            path=Path(f"/tmp/f{i}.png"),
            centres=centres,
            labels=["a", "b"],
            sizes=[(40, 14), (30, 12)],
        )
        ex.negatives.extend([(120.0, 90.0), (250.0, 40.0)])
        ex.ignore.append((10.0, 10.0, 70.0, 40.0))
        out.append((ex, rng.random((3, h, w), dtype=np.float32), w))
    return out


def _rows():
    from smolsmort.boxes import train as box_train
    from smolsmort.detect import train as detect_train

    rows = []
    for ex, image, w in _examples():
        for seed in (0, 1, 2, 7):
            rng = random.Random(seed)
            win, target, mask = detect_train._crop_window(ex, image, rng, {"a": 0, "b": 1}, 128, 2)
            rows.append(
                {
                    "backend": "detect",
                    "frame": w,
                    "seed": seed,
                    "shape": list(win.shape),
                    "win_sum": float(win.sum()),
                    "target_sum": float(target.sum()),
                    "mask_sum": float(mask.sum()),
                }
            )
            rng = random.Random(seed)
            win, tgt = box_train._window(ex, image, 0.5, rng, {"a": 0, "b": 1}, 128)
            rows.append(
                {
                    "backend": "box",
                    "frame": w,
                    "seed": seed,
                    "shape": list(win.shape),
                    "win_sum": float(win.sum()),
                    "heat_sum": float(tgt.heat.sum()),
                    "mask_sum": float(tgt.mask.sum()),
                    "size_sum": float(tgt.size.sum()),
                }
            )
    return rows


def test_both_samplers_reproduce_their_goldens():
    for got, want in zip(_rows(), GOLDEN, strict=True):
        assert got["backend"] == want["backend"] and got["seed"] == want["seed"]
        for key, value in want.items():
            if isinstance(value, float):
                assert got[key] == pytest.approx(value, rel=1e-6), (
                    want["backend"],
                    want["seed"],
                    key,
                )
            else:
                assert got[key] == value, (want["backend"], want["seed"], key)
