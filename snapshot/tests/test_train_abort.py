"""a training run can be stopped from the page, at the next epoch, without saving anything"""

from __future__ import annotations

import threading
from types import SimpleNamespace

from snapshot.review.train import TrainState


def _trainer(tmp_path, monkeypatch):
    """a trainer whose train() only counts epochs - the abort is wiring, not learning"""
    from smolsmort.detect import train as detect_train

    seen: list[int] = []
    saved: list = []
    release = threading.Event()

    def fake_train(examples, epochs, on_progress, **_):
        for epoch in range(1, epochs + 1):
            release.wait(5)
            seen.append(epoch)
            on_progress(SimpleNamespace(epoch=epoch, epochs=epochs, loss=1.0 / epoch, seen=0))
        return object(), []

    monkeypatch.setattr(detect_train, "train", fake_train)
    monkeypatch.setattr(detect_train, "save", lambda model, path: saved.append(path))
    trainer = TrainState(state=SimpleNamespace(uniform_width=132, uniform_height=12))
    trainer.MODEL_PATH = tmp_path / "model.pt"
    monkeypatch.setattr(trainer, "dataset", lambda: [])
    monkeypatch.setattr(
        trainer, "window_floor", lambda: {"floor": 64, "box": [132, 12], "default": 256}
    )
    return trainer, release, saved, seen


def test_an_abort_stops_at_the_next_epoch_and_saves_nothing(tmp_path, monkeypatch):
    trainer, release, saved, seen = _trainer(tmp_path, monkeypatch)
    assert trainer.start(epochs=50, batch=8) == {"ok": True}
    assert trainer.abort() == {"ok": True}
    release.set()
    trainer._worker.join(5)

    job = trainer.status()
    assert job["running"] is False and job["finished"] is False
    assert job["aborted"] == 1
    assert seen == [1]
    assert saved == []


def test_an_abort_with_nothing_running_is_refused(tmp_path, monkeypatch):
    trainer, *_ = _trainer(tmp_path, monkeypatch)
    assert "error" in trainer.abort()


def test_the_run_after_an_abort_is_not_aborted_too(tmp_path, monkeypatch):
    """the flag is cleared by start, or one abort would kill every later run at epoch one"""
    trainer, release, saved, _ = _trainer(tmp_path, monkeypatch)
    trainer.start(epochs=3, batch=8)
    trainer.abort()
    release.set()
    trainer._worker.join(5)

    trainer.start(epochs=3, batch=8)
    trainer._worker.join(5)
    job = trainer.status()
    assert job["finished"] is True and job["aborted"] is None
    assert saved == [trainer.MODEL_PATH]
