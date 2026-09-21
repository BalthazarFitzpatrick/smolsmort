"""the size-aware box backend: targets and decode, reach, refusals, the registry and the seam."""

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pytest

from smolsmort import backends
from smolsmort.detect.dataset import Example

# ---------------------------------------------------------------- registry and seam, no torch


def test_listing_backends_does_not_import_torch():
    """torch is optional - naming the backends must not pull it in"""
    code = "import sys, smolsmort.backends as b; b.names(); print('torch' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


def test_an_unknown_backend_names_the_known_ones():
    with pytest.raises(backends.BackendError, match="box, heatmap"):
        backends.get_backend("nope")


def test_register_plugs_in_an_outside_backend():
    backends.register("fake", "test_loop", "FakeBackend")
    try:
        assert type(backends.get_backend("fake")).__name__ == "FakeBackend"
    finally:
        backends._REGISTRY.pop("fake")


def test_both_built_in_backends_have_the_seam_shape():
    from test_loop import ModelBackend

    for name in backends.names():
        assert isinstance(backends.get_backend(name), ModelBackend), name


def test_a_checkpoint_with_no_sidecar_is_a_heatmap_one(tmp_path):
    """every checkpoint saved before backends named themselves was the heatmap cnn"""
    checkpoint = tmp_path / "old.pt"
    checkpoint.write_bytes(b"x")
    assert backends.backend_of(checkpoint) == "heatmap"


def test_a_sidecar_names_its_backend(tmp_path):
    checkpoint = tmp_path / "a.pt"
    backends.sidecar(checkpoint).write_text(json.dumps({"backend": "box"}))
    assert backends.backend_of(checkpoint) == "box"


def test_a_loop_dict_with_one_size_gives_every_object_that_size():
    item = {"path": "f.png", "centres": [(1, 2), (3, 4)], "labels": [None, None]}
    example = backends.example_from({**item, "width": 10, "height": 5})
    assert example.sizes == [(10, 5), (10, 5)]


def test_per_object_sizes_win_over_the_single_size():
    item = {"path": "f.png", "centres": [(1, 2), (3, 4)], "width": 10, "height": 5}
    example = backends.example_from({**item, "sizes": [(10, 5), (40, 20)]})
    assert example.sizes == [(10, 5), (40, 20)]


# ---------------------------------------------------------------- the model, torch required


def _outputs_from(target, torch):
    """a perfect net's answer for a target: hot only at the centre cells, sizes and offsets exact"""
    heat = np.where(target.heat >= 0.99, 10.0, -10.0).astype(np.float32)
    return (
        torch.from_numpy(heat)[None],
        torch.from_numpy(target.size)[None],
        torch.from_numpy(target.offset)[None],
    )


def test_a_box_survives_target_and_decode_unchanged():
    torch = pytest.importorskip("torch")
    from smolsmort.boxes.model import box_target, decode_boxes

    target = box_target((64, 64), [(100.0, 60.0, 140.0, 180.0)], [0], 1)
    (found,) = decode_boxes(_outputs_from(target, torch), scale=1.0)
    assert found.left == pytest.approx(100, abs=1) and found.top == pytest.approx(60, abs=1)
    assert found.width == pytest.approx(40, abs=1) and found.height == pytest.approx(120, abs=1)


def test_decode_scales_back_to_capture_pixels():
    torch = pytest.importorskip("torch")
    from smolsmort.boxes.model import box_target, decode_boxes

    target = box_target((64, 64), [(100.0, 60.0, 140.0, 180.0)], [0], 1)
    (found,) = decode_boxes(_outputs_from(target, torch), scale=0.5)
    assert found.width == pytest.approx(80, abs=2) and found.height == pytest.approx(240, abs=2)


def test_a_box_checkpoint_built_narrower_than_the_default_reloads(tmp_path):
    torch = pytest.importorskip("torch")
    from smolsmort.boxes.model import build_model, head_in, widths_in
    from smolsmort.boxes.train import load, save

    # load used to rebuild the default net and load_state_dict refused these shapes
    widths, head = (8, 16, 32, 48, 64), 24
    model = build_model(classes=2, widths=widths, head=head)
    path = save(model, tmp_path / "narrow.pt")
    state = torch.load(path, map_location="cpu")
    assert widths_in(state) == widths and head_in(state) == head
    back = load(path, device="cpu")
    assert back.heat[-1].weight.shape[0] == 2
    assert back.stem[0][0].weight.shape[0] == 8 and back.lat4.weight.shape[0] == head


def test_two_overlapping_objects_both_decode():
    """detect's decode drops a peak within 3 cells of a stronger one; overlapping centres are
    closer than that, and the second object is the one that must survive"""
    torch = pytest.importorskip("torch")
    from smolsmort.boxes.model import box_target, decode_boxes

    boxes = [(80.0, 80.0, 160.0, 120.0), (88.0, 84.0, 168.0, 124.0)]  # centres 2 cells apart
    target = box_target((64, 64), boxes, [0, 0], 1)
    assert len(decode_boxes(_outputs_from(target, torch), scale=1.0)) == 2


def test_the_net_can_see_its_largest_synthetic_object():
    pytest.importorskip("torch")
    from smolsmort.boxes.model import WORK_LONG_SIDE, build_model, receptive_field
    from smolsmort.boxes.synthetic import FRAME_SIZES, MAX_LENGTH

    largest = MAX_LENGTH * WORK_LONG_SIDE / min(max(size) for size in FRAME_SIZES)
    assert receptive_field(build_model()) >= 1.5 * largest


def test_the_net_stays_small():
    pytest.importorskip("torch")
    from smolsmort.boxes.model import build_model

    assert sum(p.numel() for p in build_model(classes=3).parameters()) < 1_000_000


def test_any_frame_shape_runs_once_padded():
    torch = pytest.importorskip("torch")
    from smolsmort.boxes.model import build_model, padded

    image = padded(np.zeros((3, 432, 768), dtype=np.float32))
    with torch.no_grad():
        heat, size, offset = build_model(classes=2).eval()(torch.from_numpy(image)[None])
    assert heat.shape == (1, 2, 112, 192) and size.shape == offset.shape == (1, 2, 112, 192)


def _frame(tmp_path, width=640, height=480):
    from PIL import Image

    path = tmp_path / "f.png"
    Image.new("RGB", (width, height)).save(path)
    return path


def test_an_object_too_big_to_see_is_refused_before_training(tmp_path):
    pytest.importorskip("torch")
    from smolsmort.boxes.train import BoxTrainError, train

    example = Example(path=_frame(tmp_path), centres=[(320.0, 240.0)], sizes=[(600.0, 400.0)])
    with pytest.raises(BoxTrainError, match="smaller long_side"):
        train([example], epochs=1, device="cpu")


def test_an_object_without_a_size_is_refused(tmp_path):
    pytest.importorskip("torch")
    from smolsmort.boxes.train import BoxTrainError, train

    example = Example(path=_frame(tmp_path), centres=[(320.0, 240.0)])
    with pytest.raises(BoxTrainError, match="needs one"):
        train([example], epochs=1, device="cpu")
