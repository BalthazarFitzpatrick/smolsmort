"""TrainApi.start hands the config menu's choices and the window to the run."""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from smolsmort import backends
from smolsmort.review import hyperparams
from smolsmort.review.train import TrainState
from smolsmort.review.train_api import TrainApi

EXAMPLE = {
    "path": "a.png",
    "centres": [(20, 20)],
    "labels": ["x"],
    "sizes": [(30, 10)],
    "negatives": [],
    "ignore": [],
    "width": 30,
    "height": 10,
}


class OptionBackend:
    """declares the options a backend may take, and remembers what it was built with"""

    seen: dict = {}

    def __init__(
        self,
        *,
        epochs: int = 1,
        optimizer: str = "adamw",
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        channels: int = 24,
    ):
        type(self).seen = {
            "epochs": epochs,
            "optimizer": optimizer,
            "momentum": momentum,
            "weight_decay": weight_decay,
            "channels": channels,
        }

    def train(self, examples, *, classes, on_progress=None, window=None):
        if on_progress:
            on_progress(1, 1)
        return object()


@pytest.fixture
def api():
    backends.register("optfake", "test_review_train_options", "OptionBackend")
    try:
        trainer = TrainState("optfake")
        trainer.bind([EXAMPLE], {"x": 0}, training_set="s")
        yield TrainApi(SimpleNamespace(), trainer)
    finally:
        backends._REGISTRY.pop("optfake", None)


def wait_done(api: TrainApi) -> None:
    deadline = time.time() + 10
    while api.status()["running"] and time.time() < deadline:
        time.sleep(0.02)


def test_optimizer_settings_reach_the_backend(api):
    api.start({"optimizer": "sgd", "momentum": "0.8", "weight_decay": 0.01, "size": "small"})
    wait_done(api)
    seen = OptionBackend.seen
    assert seen["optimizer"] == "sgd" and seen["momentum"] == 0.8
    assert seen["weight_decay"] == 0.01


def test_an_option_the_backend_does_not_declare_is_left_out(api):
    # optfake has no learning_rate or seed; sending them must not raise
    assert api.start({"learning_rate": 0.001, "seed": 3}) == {"ok": True}
    wait_done(api)


def test_a_size_name_becomes_the_backends_own_setting(api):
    api.trainer.backend_name = "heatmap"
    assert api._model_options({"size": "large"})["channels"] == 32
    assert api._model_options({"size": "custom", "custom_channels": 40})["channels"] == 40
    with pytest.raises(hyperparams.HyperparamError):
        api._model_options({"size": "nonsense"})


def test_the_window_reaches_the_run_and_a_too_small_one_is_refused(api):
    pytest.importorskip("torch")
    api.trainer.backend_name = "heatmap"
    refused = api.start({"crop": 8})
    assert "30x10" in refused["error"] and "8" in refused["error"], refused
    assert not api.status()["running"]


def test_a_backend_without_a_window_knob_is_not_handed_one(api, monkeypatch):
    calls = []
    monkeypatch.setattr(api.trainer, "start", lambda **kw: calls.append(kw) or {"ok": True})
    monkeypatch.setattr(OptionBackend, "train", lambda self, examples, *, classes: None)
    api.start({"crop": 256})
    assert calls == [{"window": None}]


def test_the_window_is_passed_to_trainstate_start(api, monkeypatch):
    calls = []
    monkeypatch.setattr(api.trainer, "start", lambda **kw: calls.append(kw) or {"ok": True})
    api.start({"crop": 256})
    api.start({})
    assert calls == [{"window": 256}, {"window": None}]
