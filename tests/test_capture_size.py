"""capture width and downscale: recorded in the weights, resampled on a mismatch, set per set"""

from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

from smolsmort import backends
from smolsmort.detect.dataset import Example
from smolsmort.detect.model import build_model, capture_width_of, downscale_of
from smolsmort.detect.train import TrainError, load, save, sweep, train
from smolsmort.review import setconfig

BOX = (40, 16)


def draw_frame(path, width: int, centre=(0.62, 0.5), seed: int = 0):
    """a dark noisy frame with one bright bar; the bar scales with the frame width"""
    height = round(width * 0.625)
    rng = np.random.default_rng(seed)
    pixels = rng.integers(0, 30, (height, width, 3), dtype=np.uint8)
    bar_w, bar_h = round(BOX[0] * width / 320), round(BOX[1] * width / 320)
    cx, cy = round(centre[0] * width), round(centre[1] * height)
    pixels[cy - bar_h // 2 : cy + bar_h // 2, cx - bar_w // 2 : cx + bar_w // 2] = 255
    Image.fromarray(pixels).save(path)
    return cx, cy


def example_at(path, width: int) -> Example:
    cx, cy = draw_frame(path, width)
    scale = width / 320
    return Example(
        path=path,
        centres=[(cx, cy)],
        labels=["bar"],
        sizes=[(BOX[0] * scale, BOX[1] * scale)],
    )


def tiny_train(examples, **options):
    settings = {"epochs": 120, "batch": 2, "channels": 8, "device": "cpu", "crop": 100}
    return train(examples, **(settings | options))


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    folder = tmp_path_factory.mktemp("cap")
    example = example_at(folder / "a.png", 320)
    model, _ = tiny_train([example], seed=1)
    return folder, model


def test_a_run_records_the_capture_width_of_its_frames(trained):
    _, model = trained
    assert capture_width_of(model) == 320
    assert downscale_of(model) == 2


def test_capture_width_and_downscale_survive_save_and_load(tmp_path):
    model = build_model(downscale=3, capture_width=1440)
    save(model, tmp_path / "m.pt")
    back = load(tmp_path / "m.pt", device="cpu")
    assert (capture_width_of(back), downscale_of(back)) == (1440, 3)


def test_a_legacy_checkpoint_loads_as_unknown_capture_at_downscale_4(tmp_path):
    import torch

    state = build_model().state_dict()
    del state["downscale"], state["capture_width"]
    torch.save(state, tmp_path / "old.pt")
    back = load(tmp_path / "old.pt", device="cpu")
    assert capture_width_of(back) is None
    assert downscale_of(back) == 4


def test_the_sidecar_and_the_backend_round_trip_keep_both(trained, tmp_path):
    folder, model = trained
    backend = backends.get_backend("heatmap", device="cpu")
    from smolsmort.detect.backend import HeatmapWeights

    weights = HeatmapWeights(model=model, classes={"bar": 0}, box=BOX)
    backend.save(weights, tmp_path / "w.pt")
    meta = json.loads(backends.sidecar(tmp_path / "w.pt").read_text())
    assert (meta["capture_width"], meta["downscale"]) == (320, 2)
    back = backend.load(tmp_path / "w.pt")
    assert (back.capture_width, back.downscale) == (320, 2)


def test_training_at_a_smaller_capture_width_resamples_and_records_it(tmp_path):
    example = example_at(tmp_path / "big.png", 320)
    model, _ = tiny_train([example], capture_width=160, epochs=2, crop=40)
    assert capture_width_of(model) == 160


def test_predict_on_another_width_maps_boxes_back_to_the_original_pixels(trained):
    folder, model = trained
    classes = {"bar": 0}
    # the same picture at half the width: the net sees it resampled up to its capture width
    cx, cy = draw_frame(folder / "half.png", 160)
    found = sweep(model, classes, [folder / "half.png"], width=BOX[0], height=BOX[1], device="cpu")
    assert found, "nothing decoded on the resampled frame"
    best = max(found, key=lambda c: c["score"])
    assert best["resampled_from"] == 160
    assert (best["width"], best["height"]) == (BOX[0] // 2, BOX[1] // 2)
    centre = (best["left"] + best["width"] / 2, best["top"] + best["height"] / 2)
    assert abs(centre[0] - cx) <= 10 and abs(centre[1] - cy) <= 10


def test_a_frame_of_the_trained_width_is_not_reported_as_resampled(trained):
    folder, model = trained
    cx, cy = draw_frame(folder / "same.png", 320)
    found = sweep(model, {"bar": 0}, [folder / "same.png"], width=40, height=16, device="cpu")
    best = max(found, key=lambda c: c["score"])
    assert "resampled_from" not in best
    assert abs(best["left"] + 20 - cx) <= 10 and abs(best["top"] + 8 - cy) <= 10


def test_fine_tuning_at_another_downscale_is_refused_naming_both(trained):
    folder, model = trained
    example = example_at(folder / "a.png", 320)
    with pytest.raises(TrainError, match=r"downscale 2.*asks for 4"):
        tiny_train([example], init_model=model, downscale=4, epochs=1)


def test_fine_tuning_keeps_the_weights_capture_width(trained):
    folder, model = trained
    example = example_at(folder / "a.png", 320)
    with pytest.warns(UserWarning, match="keeps the weights' capture width 320"):
        tuned, _ = tiny_train([example], init_model=model, capture_width=160, epochs=1)
    assert capture_width_of(tuned) == 320


# ---------------------------------------------------------------- set config and routes


def test_set_config_validates_and_keeps_optional_keys(tmp_path, monkeypatch):
    from smolsmort.review import paths

    monkeypatch.setattr(paths, "DATASETS_DIR", tmp_path)
    assert setconfig.write_set_config("s", "heatmap") == {
        "backend": "heatmap",
        "size_mode": "uniform",
    }
    saved = setconfig.write_set_config("s", "heatmap", capture_width=1280, downscale=3)
    assert saved["capture_width"] == 1280 and saved["downscale"] == 3
    # left out keeps, null clears
    assert setconfig.write_set_config("s", "heatmap")["downscale"] == 3
    assert "downscale" not in setconfig.write_set_config("s", "heatmap", downscale=None)
    for bad in (
        {"downscale": 9},
        {"downscale": 0},
        {"capture_width": 63},
        {"capture_width": 20000},
    ):
        with pytest.raises(setconfig.SetConfigError):
            setconfig.write_set_config("s", "heatmap", **bad)
    with pytest.raises(setconfig.SetConfigError):
        setconfig.write_set_config("s", "heatmap", downscale="2")
