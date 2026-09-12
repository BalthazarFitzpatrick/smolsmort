"""the trainer only ever sees the train split.

splits.py ITSELF HAS MOVED to smolsmort/review/splits.py - see tests/test_splits.py for its own
behaviour. What is left here is what still depends on snapshot/review/train.py, not yet ported: the
trainer's own use of a set's split column."""

from __future__ import annotations

import json


def _old_set(tmp_path, frames=10):
    path = tmp_path / "old.jsonl"
    rows = [
        {"recording": "synthetic/run", "frame": f"{i}.jpg", "negative": False}
        for i in range(frames)
    ]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows * 3))
    path.with_suffix(".meta.json").write_text(json.dumps({"name": "old", "rows": len(rows) * 3}))
    return path


def test_the_trainer_loads_only_the_train_split(tmp_path, monkeypatch):
    """the holdouts are written into the set, so the trainer must drop them itself"""
    import json

    from PIL import Image

    from snapshot.review import paths
    from snapshot.review.train import TrainState

    sessions = tmp_path / "sessions"
    frames = sessions / "synthetic" / "run" / "frames"
    frames.mkdir(parents=True)
    sets = tmp_path / "sets"
    sets.mkdir()
    rows = []
    for name, split in (
        ("a.jpg", "train"),
        ("b.jpg", "train"),
        ("c.jpg", "val"),
        ("d.jpg", "test"),
    ):
        Image.new("RGB", (64, 64)).save(frames / name)
        rows.append(
            {"frame": name, "left": 4, "top": 4, "width": 20, "height": 6, "label": "hostile",
             "negative": False, "recording": "synthetic/run", "split": split}
        )  # fmt: skip
    (sets / "s.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    monkeypatch.setattr(paths, "SESSIONS_DIR", sessions)
    monkeypatch.setattr(paths, "DATASETS_DIR", sets)

    trainer = TrainState(state=None)
    trainer.training_set = "s"
    assert sorted(e.path.name for e in trainer.dataset()) == ["a.jpg", "b.jpg"]
    info = trainer.info()
    assert info["frames"] == 2
    assert info["split"] == {"train": 2, "val": 1, "test": 1}


def test_the_val_split_is_scored_and_nothing_else_is(tmp_path, monkeypatch):
    """recall and false positives come from val frames only - train and test are never run"""
    import json

    import numpy as np
    from smolsmort.detect import train as detect_train

    from snapshot.review import paths
    from snapshot.review.train import TrainState

    sessions = tmp_path / "sessions"
    frames = sessions / "synthetic" / "run" / "frames"
    frames.mkdir(parents=True)
    sets = tmp_path / "sets"
    sets.mkdir()
    rows = []
    for name, split, negative in (
        ("a.jpg", "train", False),
        ("c.jpg", "val", False),
        ("e.jpg", "val", True),
        ("d.jpg", "test", False),
    ):
        (frames / name).touch()
        rows.append(
            {"frame": name, "left": 300, "top": 157, "width": 40, "height": 6,
             "label": None if negative else "hostile", "negative": negative,
             "recording": "synthetic/run", "split": split}
        )  # fmt: skip
    (sets / "s.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    monkeypatch.setattr(paths, "SESSIONS_DIR", sessions)
    monkeypatch.setattr(paths, "DATASETS_DIR", sets)

    # a hit on the plate frame, and a peak on the empty one - a false positive
    ran = []

    def heatmaps(model, path):
        ran.append(path.name)
        heat = np.zeros((1, 40, 60), dtype=np.float32)
        heat[0, 10, 20] = 0.9
        return heat

    monkeypatch.setattr(detect_train, "heatmaps_for", heatmaps)
    trainer = TrainState(state=None)
    trainer.training_set = "s"
    trainer.dataset()
    got = trainer._holdout_score(model=None)
    assert sorted(ran) == ["c.jpg", "e.jpg"]
    assert (got["with_plate"], got["without_plate"]) == (1, 1)
    assert got["recall"] == 1.0
    assert got["false_positive_rate"] == 1.0
    assert got["passes"] is False


def test_a_set_without_a_val_split_reports_no_holdout(tmp_path, monkeypatch):
    from snapshot.review import paths
    from snapshot.review.train import TrainState

    path = _old_set(tmp_path)
    monkeypatch.setattr(paths, "DATASETS_DIR", path.parent)
    monkeypatch.setattr(paths, "SESSIONS_DIR", tmp_path / "sessions")
    trainer = TrainState(state=None)
    trainer.training_set = path.stem
    trainer.holdout = []
    assert trainer._holdout_score(model=None) is None
