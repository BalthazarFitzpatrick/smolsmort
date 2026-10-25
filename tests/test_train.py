"""the heatmap trainer's own logic, without needing a gpu or a real training run."""

from __future__ import annotations

import numpy as np
import pytest

from smolsmort.detect.dataset import Example
from smolsmort.detect.model import STRIDE, gaussian_target
from smolsmort.detect.train import _crop_window, masked_focal_loss

torch = pytest.importorskip("torch")


def test_a_fractional_centre_produces_no_positive_cell():
    """REGRESSION, and it silently broke the whole thing. gaussian_target builds its blob from the
    coordinate given, so a fractional centre never reaches 1.0 - and the loss counts positives with
    target >= 0.99. With no positives only the negative term trained and the net learned to answer a
    flat value everywhere. The trainer therefore rounds centres to cells."""
    fractional = gaussian_target((16, 16), [(8.4, 8.6)])
    assert (fractional >= 0.99).sum() == 0
    rounded = gaussian_target((16, 16), [(8.0, 9.0)])
    assert (rounded >= 0.99).sum() == 1


def test_masked_loss_ignores_masked_cells():
    logits = torch.zeros(1, 1, 8, 8)
    target = torch.zeros(1, 1, 8, 8)
    target[0, 0, 4, 4] = 1.0
    all_on = torch.ones(1, 1, 8, 8)
    # masking everything except the positive leaves only the positive term
    only_positive = torch.zeros(1, 1, 8, 8)
    only_positive[0, 0, 4, 4] = 1.0
    assert float(masked_focal_loss(logits, target, only_positive)) < float(
        masked_focal_loss(logits, target, all_on)
    )


def test_masked_loss_falls_back_when_there_is_no_positive():
    logits = torch.zeros(1, 1, 4, 4)
    target = torch.zeros(1, 1, 4, 4)
    mask = torch.ones(1, 1, 4, 4)
    assert float(masked_focal_loss(logits, target, mask)) > 0


def test_masked_loss_takes_logits_not_probabilities():
    """model.focal_loss applies its own sigmoid; passing it probabilities double-applies
    one. this takes logits, so a confident correct logit must score better than a neutral zero."""
    target = torch.zeros(1, 1, 4, 4)
    target[0, 0, 2, 2] = 1.0
    mask = torch.ones(1, 1, 4, 4)
    confident = torch.full((1, 1, 4, 4), -6.0)
    confident[0, 0, 2, 2] = 6.0
    assert float(masked_focal_loss(confident, target, mask)) < float(
        masked_focal_loss(torch.zeros(1, 1, 4, 4), target, mask)
    )


def test_crop_window_shapes_line_up():
    import random

    image = np.zeros((3, 360, 640), dtype=np.float32)
    example = Example(path=None, centres=[(800.0, 600.0)])
    window, target, mask = _crop_window(example, image, random.Random(0))
    assert window.shape[1] == window.shape[2]
    assert target.shape == mask.shape == (window.shape[1] // STRIDE,) * 2


def test_an_ignore_region_never_blanks_a_confirmed_plate():
    """an object can sit inside an ignore box; the mask must not erase it from the loss"""
    import random

    image = np.zeros((3, 360, 640), dtype=np.float32)
    centre = (400.0, 300.0)
    example = Example(path=None, centres=[centre], ignore=[(0, 0, 2560, 1440)])
    _, target, mask = _crop_window(example, image, random.Random(1))
    assert mask[target > 0.3].min() == 1.0


def test_frames_of_mixed_sizes_train_in_one_batch(tmp_path):
    """a set mixing frames shorter than the window with full ones trains: the window shrinks to
    the smallest frame, in whole cells, instead of each short frame cutting its own"""
    from PIL import Image

    from smolsmort.detect.train import train

    rng = np.random.default_rng(0)
    examples = []
    for index, (height, width) in enumerate([(810, 1440), (410, 1440), (810, 1440)]):
        path = tmp_path / f"frame_{index}.png"
        Image.fromarray(rng.integers(0, 255, (height, width, 3), dtype=np.uint8)).save(path)
        examples.append(Example(path=path, centres=[(700.0, 200.0)], exhaustive=True))
    model, history = train(examples, epochs=1, batch=4, capture_width=1440)
    assert history


# --------------------------------------- the crop window must be a whole number of cells


def test_the_model_and_the_target_agree_on_cell_count_for_snapped_windows():
    """THE BUG, at its root. Both stride-2 convs use padding=1 so the network produces ceil(S/4)
    cells, while build_target lays out S // 4 - equal only when S divides by STRIDE. A window of
    286 gave a 72-cell prediction against a 71-cell target and torch raised "size of tensor a (72)
    must match the size of tensor b (71)" inside the loss, naming neither the crop nor the setting.
    """
    torch = pytest.importorskip("torch")
    from smolsmort.detect.model import STRIDE, build_model
    from smolsmort.detect.train import snapped_window

    model = build_model(channels=4, classes=1).eval()
    for asked in (256, 285, 286, 287, 300, 301):
        window = snapped_window(asked)
        assert window % STRIDE == 0
        assert window >= asked, "snapping must never narrow a window a human widened on purpose"
        with torch.no_grad():
            out = model(torch.zeros(1, 3, window, window))
        assert out.shape[-1] == window // STRIDE, f"window {window} disagrees with the target grid"
        assert out.shape[-2] == window // STRIDE


def test_an_unsnapped_window_really_does_disagree():
    """the guard is not theoretical - without snapping these shapes differ, which is the crash"""
    torch = pytest.importorskip("torch")
    from smolsmort.detect.model import STRIDE, build_model

    with torch.no_grad():
        out = build_model(channels=4, classes=1).eval()(torch.zeros(1, 3, 286, 286))
    assert out.shape[-1] != 286 // STRIDE


def test_on_progress_returning_true_stops_training_after_that_epoch(tmp_path):
    from PIL import Image

    from smolsmort.detect.train import train

    path = tmp_path / "frame.png"
    rng = np.random.default_rng(0)
    Image.fromarray(rng.integers(0, 255, (810, 1440, 3), dtype=np.uint8)).save(path)
    examples = [Example(path=path, centres=[(700.0, 200.0)], exhaustive=True)]
    model, history = train(
        examples, epochs=10, batch=2, capture_width=1440, on_progress=lambda p: p.epoch >= 3
    )
    assert len(history) == 3


def test_plateau_stalls_after_patience_epochs_without_a_gain():
    from smolsmort.detect.train import Plateau

    plateau = Plateau(patience=10, min_delta=0.01)
    assert plateau.update(5, 0.50)
    assert plateau.update(10, 0.60)
    assert not plateau.update(15, 0.605)  # under min_delta
    assert not plateau.stalled(15)
    assert not plateau.update(20, 0.59)
    assert plateau.stalled(20) and plateau.best_epoch == 10


def test_plateau_tracks_a_loss_where_lower_is_better():
    from smolsmort.detect.train import Plateau

    plateau = Plateau(patience=2, higher_is_better=False)
    assert plateau.update(1, 1.0)
    assert plateau.update(2, 0.5)
    assert not plateau.update(3, 0.7)
    assert not plateau.update(4, 0.6)
    assert plateau.stalled(4) and plateau.best == 0.5


def test_a_set_over_the_memory_budget_trains_from_frames_decoded_per_epoch(tmp_path):
    """frames that would not fit in memory are decoded each epoch on worker threads"""
    from PIL import Image

    from smolsmort.detect.train import _input_shape, _load_input, train

    rng = np.random.default_rng(0)
    examples = []
    for index, (height, width) in enumerate([(810, 1440), (600, 1024), (810, 1440)]):
        path = tmp_path / f"frame_{index}.png"
        Image.fromarray(rng.integers(0, 255, (height, width, 3), dtype=np.uint8)).save(path)
        assert _input_shape(path, 2, 1440) == _load_input(path, 2, 1440).shape[1:]
        examples.append(Example(path=path, centres=[(700.0, 200.0)], exhaustive=True))
    model, history = train(examples, epochs=2, batch=4, capture_width=1440, cache_bytes=0)
    assert len(history) == 2 and all(np.isfinite(history))
