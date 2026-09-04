"""one heatmap channel per class primitive.

Balthazar Fitzpatrick, 2026-09-01, proposing it: "if you can look at all my crops through the eyes of a cnn, it
will be able to draw a heatmap per 10 categories via 10 output channels, and that's all the x and y
you need". he is right, and it is what lets the cnn replace the template matcher: a plate is a
FIXED KNOWN SIZE, so there is nothing to regress - the only unknowns are where and which.

the five primitives are friendly green, neutral yellow, hostile red, player blue and tagged grey,
each in a normal and a dim variant.
"""

from __future__ import annotations

import pytest

from smolsmort.detect.model import build_model, class_target, count_parameters

torch = pytest.importorskip("torch")


def test_the_head_has_one_channel_per_class():
    assert build_model(classes=10)[-1].out_channels == 10


def test_one_class_stays_the_default():
    """every checkpoint and caller written before this keeps working"""
    assert build_model()[-1].out_channels == 1


def test_ten_classes_costs_almost_nothing():
    """a 1x1 conv over the same trunk - the reason this is cheap rather than a redesign"""
    extra = count_parameters(build_model(classes=10)) - count_parameters(build_model(classes=1))
    assert 0 < extra < 2000


def test_a_plate_marks_only_its_own_channel():
    """a red plate is not a negative example for green - it simply leaves that channel empty"""
    target = class_target((20, 30), [(5.0, 5.0)], [2], classes=10)
    assert target.shape == (10, 20, 30)
    assert target[2].max() == pytest.approx(1.0)
    assert all(target[c].max() == 0.0 for c in range(10) if c != 2)


def test_two_primitives_in_one_frame_land_in_two_channels():
    target = class_target((20, 30), [(5.0, 5.0), (15.0, 10.0)], [0, 7], classes=10)
    assert target[0].max() == pytest.approx(1.0)
    assert target[7].max() == pytest.approx(1.0)
    assert target[0][10, 15] == 0.0, "one plate must not blob into the other's position"


def test_two_of_the_same_primitive_share_a_channel():
    target = class_target((20, 40), [(5.0, 5.0), (30.0, 12.0)], [4, 4], classes=10)
    assert (target[4] > 0.9).sum() >= 2


def test_a_label_outside_the_range_is_ignored_not_crashed():
    """a stale label from an older class list must not take a training run down"""
    target = class_target((10, 10), [(5.0, 5.0)], [99], classes=10)
    assert target.max() == 0.0


def test_the_model_runs_and_gives_a_map_per_class():
    model = build_model(classes=10)
    model.eval()
    with torch.no_grad():
        out = model(torch.zeros(1, 3, 64, 128))
    assert out.shape[1] == 10
    assert out.shape[2] == 64 // 4 and out.shape[3] == 128 // 4, "stride 4, as decode_peaks assumes"


# ------------------------------------------- the seam that carries a cluster label into training


def test_a_confirmed_box_carries_its_class(tmp_path):
    """cluster names a tile; nothing carried that name into the dataset, so the ten channels had
    no labels to learn from. Example.labels is that seam, parallel to centres
    """
    from PIL import Image

    from smolsmort.detect.dataset import build_from

    frames = tmp_path / "frames"
    frames.mkdir()
    Image.new("RGB", (400, 200)).save(frames / "f.jpg")
    boxes = [
        {"path": "f.jpg", "left": 10, "top": 10, "width": 40, "height": 12},
        {"path": "f.jpg", "left": 200, "top": 60, "width": 40, "height": 12},
    ]
    keeps = {"0": {"keep": True}, "1": {"keep": True}}
    examples = build_from(boxes, keeps, frames, labels={0: "hostile npc"})
    example = examples[0]
    assert len(example.centres) == len(example.labels) == 2
    assert example.labels[0] == "hostile npc"
    assert example.labels[1] is None, "unlabelled must be None, not guessed"


def test_an_unlabelled_box_is_still_a_plate(tmp_path):
    """a drawn box that has not been through cluster yet still trains the single-channel model -
    otherwise drawing would be useless until every box had been named
    """
    from PIL import Image

    from smolsmort.detect.dataset import build_from

    frames = tmp_path / "frames"
    frames.mkdir()
    Image.new("RGB", (400, 200)).save(frames / "f.jpg")
    boxes = [{"path": "f.jpg", "left": 10, "top": 10, "width": 40, "height": 12}]
    examples = build_from(boxes, {"0": {"keep": True}}, frames, labels=None)
    assert examples[0].centres and examples[0].labels == [None]


def test_a_discarded_box_carries_no_label(tmp_path):
    """discard means "not wanted", so it becomes an ignore region and never reaches labels"""
    from PIL import Image

    from smolsmort.detect.dataset import build_from

    frames = tmp_path / "frames"
    frames.mkdir()
    Image.new("RGB", (400, 200)).save(frames / "f.jpg")
    boxes = [{"path": "f.jpg", "left": 10, "top": 10, "width": 40, "height": 12}]
    examples = build_from(boxes, {"0": {"keep": False}}, frames, labels={0: "green"})
    assert examples[0].centres == [] and examples[0].labels == []
    assert examples[0].ignore


# ------------------------------------------------------- training with one channel per class


def _tiny_examples(tmp_path, labels):
    from PIL import Image

    from smolsmort.detect.dataset import build_from

    frames = tmp_path / "frames"
    frames.mkdir()
    for name in ("a.jpg", "b.jpg"):
        Image.new("RGB", (256, 256), (40, 60, 40)).save(frames / name)
    boxes = [
        {"path": "a.jpg", "left": 40, "top": 40, "width": 60, "height": 16},
        {"path": "a.jpg", "left": 150, "top": 120, "width": 60, "height": 16},
        {"path": "b.jpg", "left": 60, "top": 90, "width": 60, "height": 16},
    ]
    keeps = {str(i): {"keep": True} for i in range(len(boxes))}
    return build_from(boxes, keeps, frames, uniform_width=60, uniform_height=16, labels=labels)


def test_training_without_classes_keeps_one_channel(tmp_path):
    """every checkpoint and caller from before this still behaves identically"""
    from smolsmort.detect.train import train

    model, history = train(_tiny_examples(tmp_path, None), epochs=1, batch=2)
    assert model[-1].out_channels == 1
    assert history


def test_training_with_classes_grows_the_head(tmp_path):
    from smolsmort.detect.train import train

    classes = {"hostile npc": 0, "player blue": 1, "green": 2}
    examples = _tiny_examples(tmp_path, {0: "hostile npc", 1: "player blue", 2: "green"})
    model, history = train(examples, epochs=1, batch=2, classes=classes)
    assert model[-1].out_channels == 3
    assert history


def test_an_unlabelled_plate_still_trains_when_multi_class(tmp_path):
    """a partly-clustered set must be trainable - otherwise every box has to be named before the
    first run, which defeats drawing quickly and labelling later
    """
    from smolsmort.detect.train import train

    classes = {"hostile npc": 0, "player blue": 1}
    examples = _tiny_examples(tmp_path, {0: "hostile npc"})  # 1 and 2 unlabelled
    model, history = train(examples, epochs=1, batch=2, classes=classes)
    assert model[-1].out_channels == 2 and history


def test_a_checkpoint_must_be_loaded_with_its_own_class_count(tmp_path):
    """a 10-channel head will not accept a 1-channel state dict, and torch's error is opaque"""
    from smolsmort.detect.train import load, save, train

    classes = {"a": 0, "b": 1, "c": 2}
    model, _ = train(_tiny_examples(tmp_path, {0: "a"}), epochs=1, batch=2, classes=classes)
    path = save(model, tmp_path / "m.pt")
    assert load(path, device="cpu", classes=3)[-1].out_channels == 3
    with pytest.raises(RuntimeError):
        load(path, device="cpu", classes=1)


def test_a_promoted_set_loads_back_and_trains(tmp_path):
    """the seam between promote and the cnn: the picker offers a training set, so the trainer has
    to be able to read one. one file, many recordings - that is the point of promoting rather than
    training off whichever candidates queue happened to be bound
    """
    import json

    from PIL import Image

    from smolsmort.detect.dataset import build_training_set
    from smolsmort.detect.train import train

    sessions = tmp_path / "recordings"
    for rec in ("one", "two"):
        (sessions / rec / "frames").mkdir(parents=True)
        Image.new("RGB", (256, 256), (40, 60, 40)).save(sessions / rec / "frames" / "f.jpg")

    rows = [
        {
            "recording": "one",
            "frame": "f.jpg",
            "left": 40,
            "top": 40,
            "width": 60,
            "height": 16,
            "label": "hostile npc",
            "source": "one",
        },
        {
            "recording": "two",
            "frame": "f.jpg",
            "left": 90,
            "top": 120,
            "width": 60,
            "height": 16,
            "label": "green",
            "source": "two",
        },
    ]
    path = tmp_path / "plates.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))

    examples, classes = build_training_set(path, sessions)
    assert len(examples) == 2, "one example per frame, across both recordings"
    assert classes == {"green": 0, "hostile npc": 1}, "sorted, so it is stable between runs"

    model, history = train(examples, epochs=1, batch=2, classes=classes)
    assert model[-1].out_channels == 2 and history


def test_an_empty_training_set_says_so(tmp_path):
    from smolsmort.detect.dataset import DatasetError, build_training_set

    path = tmp_path / "plates.jsonl"
    path.write_text("")
    with pytest.raises(DatasetError, match="promote"):
        build_training_set(path, tmp_path)


def test_an_excluded_candidate_becomes_a_hard_negative(tmp_path):
    """the warm loop's other half. a swept candidate that gets discarded is somewhere the model
    FIRED and was told no - the one kind of background worth sampling deliberately, since a random
    window almost never contains the model's own mistake. plain absence teaches nothing
    """
    import json

    from PIL import Image

    from smolsmort.detect.dataset import build_training_set

    sessions = tmp_path / "recordings"
    (sessions / "one" / "frames").mkdir(parents=True)
    Image.new("RGB", (256, 256), (40, 60, 40)).save(sessions / "one" / "frames" / "f.jpg")

    rows = [
        {
            "recording": "one",
            "frame": "f.jpg",
            "left": 40,
            "top": 40,
            "width": 60,
            "height": 16,
            "label": "hostile npc",
            "negative": False,
            "source": "one",
        },
        {
            "recording": "one",
            "frame": "f.jpg",
            "left": 150,
            "top": 90,
            "width": 60,
            "height": 16,
            "label": None,
            "negative": True,
            "source": "one",
        },
    ]
    path = tmp_path / "plates.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))

    examples, classes = build_training_set(path, sessions)
    example = examples[0]
    assert len(example.centres) == 1, "a negative is not a plate"
    assert len(example.negatives) == 1
    assert classes == {"hostile npc": 1 - 1}, "a negative contributes no class"


def test_negatives_do_not_break_multi_class_training(tmp_path):
    import json

    from PIL import Image

    from smolsmort.detect.dataset import build_training_set
    from smolsmort.detect.train import train

    sessions = tmp_path / "recordings"
    (sessions / "one" / "frames").mkdir(parents=True)
    Image.new("RGB", (256, 256), (40, 60, 40)).save(sessions / "one" / "frames" / "f.jpg")
    rows = [
        {
            "recording": "one",
            "frame": "f.jpg",
            "left": 40,
            "top": 40,
            "width": 60,
            "height": 16,
            "label": "a",
            "negative": False,
            "source": "one",
        },
        {
            "recording": "one",
            "frame": "f.jpg",
            "left": 150,
            "top": 90,
            "width": 60,
            "height": 16,
            "label": "b",
            "negative": False,
            "source": "one",
        },
        {
            "recording": "one",
            "frame": "f.jpg",
            "left": 20,
            "top": 200,
            "width": 60,
            "height": 16,
            "label": None,
            "negative": True,
            "source": "one",
        },
    ]
    path = tmp_path / "plates.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    examples, classes = build_training_set(path, sessions)
    model, history = train(examples, epochs=1, batch=2, classes=classes)
    assert model[-1].out_channels == 2 and history
