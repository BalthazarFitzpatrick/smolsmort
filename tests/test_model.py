"""the heatmap net: shapes, loss behaviour, decoding, and that it can actually learn.

synthetic plates here - a gold bar on noisy green - because no corrected real dataset exists yet.
these prove the MACHINERY is sound so that the moment real labels arrive the only variable left is
the data.
"""

from __future__ import annotations

import numpy as np
import pytest

from smolsmort.detect.model import (
    DOWNSCALE,
    STRIDE,
    Peak,
    build_model,
    count_parameters,
    decode_peaks,
    focal_loss,
    gaussian_target,
)

torch = pytest.importorskip("torch")


def synthetic(height=96, width=160, centres=((40, 30), (110, 60)), rng=None):
    """a fake frame at INPUT scale with flat gold bars, plus its heatmap target"""
    rng = rng or np.random.default_rng(5)
    image = rng.integers(20, 60, size=(height, width, 3)).astype(np.float32)
    image[:, :, 1] += 25  # greenish ground, like the real captures
    for cx, cy in centres:
        left, right = cx - 16, cx + 16  # 132px capture bar / DOWNSCALE ~= 33px at input scale
        image[cy - 1 : cy + 2, left:right] = (170, 140, 60)
    target = gaussian_target(
        (height // STRIDE, width // STRIDE), [(cx / STRIDE, cy / STRIDE) for cx, cy in centres]
    )
    tensor = torch.from_numpy(image.transpose(2, 0, 1) / 255.0).float().unsqueeze(0)
    return tensor, torch.from_numpy(target).unsqueeze(0).unsqueeze(0)


def test_the_model_is_small():
    """the whole argument for a custom net is that a fixed-size target needs almost no capacity"""
    assert count_parameters(build_model()) < 100_000


def test_the_heatmap_comes_out_at_stride_4_and_one_channel():
    image, _ = synthetic()
    out = build_model()(image)
    assert out.shape[1] == 1
    assert out.shape[2] == image.shape[2] // STRIDE
    assert out.shape[3] == image.shape[3] // STRIDE


def test_it_is_fully_convolutional_so_it_runs_on_any_size():
    """trained on crops, run on whole frames - a dense layer would forbid that"""
    model = build_model()
    for h, w in ((96, 160), (128, 256)):
        image, _ = synthetic(h, w, centres=((40, 30),))
        assert model(image).shape[2:] == (h // STRIDE, w // STRIDE)


def test_the_gaussian_target_peaks_at_the_plate_and_falls_away():
    target = gaussian_target((40, 40), [(20.0, 10.0)])
    assert target[10, 20] == pytest.approx(1.0)
    assert target[10, 26] < 0.05
    assert target.max() <= 1.0


def test_two_plates_both_appear_in_the_target():
    target = gaussian_target((40, 40), [(10.0, 10.0), (30.0, 20.0)])
    assert target[10, 10] == pytest.approx(1.0)
    assert target[20, 30] == pytest.approx(1.0)


def test_a_centre_off_the_edge_is_dropped_not_wrapped():
    target = gaussian_target((20, 20), [(-50.0, -50.0)])
    assert target.max() == 0.0


def test_focal_loss_punishes_a_confident_miss_more_than_a_confident_hit():
    target = torch.from_numpy(gaussian_target((16, 16), [(8.0, 8.0)])).view(1, 1, 16, 16)
    hit = torch.full((1, 1, 16, 16), -4.0)
    hit[0, 0, 8, 8] = 4.0
    miss = torch.full((1, 1, 16, 16), -4.0)
    miss[0, 0, 2, 2] = 4.0
    assert focal_loss(miss, target) > focal_loss(hit, target)


def test_decode_returns_capture_coordinates_not_heatmap_ones():
    heat = np.zeros((20, 20), dtype=np.float32)
    heat[5, 7] = 0.9
    (peak,) = decode_peaks(heat)
    scale = STRIDE * DOWNSCALE
    assert peak.x == 7 * scale + scale // 2
    assert peak.y == 5 * scale + scale // 2
    assert peak.score == pytest.approx(0.9)


def test_decode_suppresses_a_neighbouring_cell_of_the_same_blob():
    heat = np.zeros((20, 20), dtype=np.float32)
    heat[5, 7] = 0.9
    heat[5, 8] = 0.8  # same plate, one cell over
    assert len(decode_peaks(heat)) == 1


def test_decode_keeps_two_genuinely_separate_plates():
    heat = np.zeros((20, 20), dtype=np.float32)
    heat[5, 2] = 0.9
    heat[5, 15] = 0.7
    peaks = decode_peaks(heat)
    assert len(peaks) == 2
    assert [p.score for p in peaks] == sorted([p.score for p in peaks], reverse=True)


def test_everything_below_threshold_decodes_to_nothing():
    assert decode_peaks(np.full((10, 10), 0.05, dtype=np.float32)) == []


def test_a_non_2d_heatmap_is_refused():
    with pytest.raises(Exception, match="2d heatmap"):
        decode_peaks(np.zeros((1, 1, 8, 8), dtype=np.float32))


def test_the_net_can_actually_learn_a_synthetic_plate():
    """the end-to-end check: overfit one frame and confirm the peak lands on the real plate.

    if the architecture, target and loss did not agree this would never converge, so it is the one
    test that exercises all three together.
    """
    torch.manual_seed(0)
    image, target = synthetic(centres=((40, 30),))
    model = build_model(channels=16)
    optimiser = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(120):
        optimiser.zero_grad()
        loss = focal_loss(model(image), target)
        loss.backward()
        optimiser.step()

    model.eval()
    with torch.no_grad():
        heat = torch.sigmoid(model(image))[0, 0].numpy()
    peaks = decode_peaks(heat, min_score=0.3)
    assert peaks, "the net learned nothing"
    best = peaks[0]
    expected_x, expected_y = 40 * DOWNSCALE, 30 * DOWNSCALE
    assert abs(best.x - expected_x) <= STRIDE * DOWNSCALE
    assert abs(best.y - expected_y) <= STRIDE * DOWNSCALE


def test_peak_is_hashable_so_detections_can_go_in_a_set():
    assert len({Peak(1, 2, 0.5), Peak(1, 2, 0.5)}) == 1


def test_a_limit_returns_the_same_strongest_peaks_it_would_have_anyway():
    """cutting the decode short must not change WHICH peaks come back, only how many.

    candidates are walked strongest first, so stopping after `limit` accepted peaks is exactly the
    head of the unlimited result - not an approximation of it.
    """
    rng = np.random.default_rng(7)
    heat = rng.random((60, 90)).astype(np.float32)

    full = decode_peaks(heat, min_score=0.05)
    capped = decode_peaks(heat, min_score=0.05, limit=12)

    assert len(capped) == 12
    assert capped == full[:12]


def test_a_low_floor_without_a_limit_is_the_slow_path_this_guards():
    """THE BUG THIS PINS: suppression compares each candidate against every peak already KEPT, so
    the cost grows with how many are kept - and at a low floor an undertrained model answers warm
    nearly everywhere. MEASURED on a real 15-channel model at floor 0.05: 33,112 candidates and
    43.9 SECONDS for one frame, of which the sweep then kept twelve.

    the assertion is about the SHAPE of the failure, not a timing - a floor that low really does
    admit thousands of peaks, and that is the thing a caller must bound rather than discover.
    """
    rng = np.random.default_rng(11)
    heat = rng.random((60, 90)).astype(np.float32)

    assert len(decode_peaks(heat, min_score=0.05, limit=8)) == 8
    assert len(decode_peaks(heat, min_score=0.05)) > 100, (
        "a low floor admits many peaks - which is why a caller wanting a few must pass a limit"
    )
