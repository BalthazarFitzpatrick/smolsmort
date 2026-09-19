"""the capture:input factor is a setting of the model, not a constant every caller must remember"""

from __future__ import annotations

import numpy as np
from PIL import Image

from smolsmort.detect.model import (
    DEFAULT_DOWNSCALE,
    LEGACY_DOWNSCALE,
    STRIDE,
    build_model,
    decode_peaks,
    downscale_of,
)
from smolsmort.detect.train import _load_input, load, minimum_window, save


def test_a_model_is_built_at_the_default_factor_unless_told_otherwise():
    assert downscale_of(build_model()) == DEFAULT_DOWNSCALE
    assert downscale_of(build_model(downscale=3)) == 3


def test_the_factor_survives_a_save_and_load(tmp_path):
    save(build_model(downscale=3), tmp_path / "m.pt")
    assert downscale_of(load(tmp_path / "m.pt", device="cpu")) == 3


def test_a_checkpoint_saved_before_the_factor_was_stored_loads_at_the_old_one(tmp_path):
    import torch

    state = build_model().state_dict()
    del state["downscale"]
    torch.save(state, tmp_path / "old.pt")
    assert downscale_of(load(tmp_path / "old.pt", device="cpu")) == LEGACY_DOWNSCALE


def test_decode_scales_a_peak_back_by_the_factor_it_is_given():
    heat = np.zeros((20, 20), dtype=np.float32)
    heat[5, 7] = 0.9
    for factor in (2, 4):
        (peak,) = decode_peaks(heat, downscale=factor)
        assert (peak.x, peak.y) == (7 * STRIDE * factor, 5 * STRIDE * factor)


def test_a_capture_is_cut_down_by_the_factor_it_is_given(tmp_path):
    Image.new("RGB", (1440, 936)).save(tmp_path / "f.png")
    assert _load_input(tmp_path / "f.png", 2).shape == (3, 468, 720)
    assert _load_input(tmp_path / "f.png", 4).shape == (3, 234, 360)


def test_the_window_that_holds_a_box_shrinks_with_a_larger_factor():
    assert minimum_window(226, 35, downscale=4) < minimum_window(226, 35, downscale=2)
