"""smolsmort/review_ui/server.py: the standalone http.server end to end - the page, its assets,
and both /api endpoints, with and without a bound TrainState."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request

import pytest

from smolsmort import backends
from smolsmort.review.train import TrainState
from smolsmort.review_ui.server import serve

torch = pytest.importorskip("torch")


@pytest.fixture(autouse=True)
def _fake_backend():
    backends.register("fake", "test_loop", "FakeBackend")
    try:
        yield
    finally:
        backends._REGISTRY.pop("fake", None)


@pytest.fixture
def running_server():
    server = serve()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        thread.join(timeout=5)


def _get(server, path):
    with urllib.request.urlopen(f"http://127.0.0.1:{server.server_port}{path}") as resp:
        return resp.status, resp.read()


def _post(server, path, payload):
    body = json.dumps(payload).encode()
    request = urllib.request.Request(
        f"http://127.0.0.1:{server.server_port}{path}",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def test_the_page_and_its_assets_serve(running_server):
    status, body = _get(running_server, "/")
    assert status == 200 and b"smolsmort review" in body

    status, body = _get(running_server, "/hyperparams.js")
    assert status == 200 and b"applyPreset" in body

    for name in ("core.js", "tab_find.js", "tab_select.js", "tab_train.js", "review.css"):
        status, body = _get(running_server, f"/{name}")
        assert status == 200 and body, name

    status, body = _get(running_server, "/ui/menu.js")
    assert status == 200 and b"class Menu" in body


def test_static_serving_stays_inside_the_static_folder(running_server):
    import http.client

    conn = http.client.HTTPConnection("127.0.0.1", running_server.server_port)
    conn.request("GET", "/../server.py")
    assert conn.getresponse().status == 404


def test_hyperparams_menu_options_over_http(running_server):
    status, body = _get(running_server, "/api/hyperparams?backend=heatmap")
    payload = json.loads(body)
    assert status == 200
    assert payload["optimizers"] == ["adamw", "sgd"]
    assert payload["sizes"]["small"] < payload["sizes"]["large"]


def test_an_unknown_backend_is_a_400(running_server):
    status, payload = _post(running_server, "/api/train-start", {"backend": "nope"})
    assert status == 400 and "nope" in payload["error"]


def test_train_start_with_no_bound_trainer_only_resolves(running_server):
    status, payload = _post(
        running_server, "/api/train-start", {"backend": "heatmap", "learning_rate": 5e-4}
    )
    assert status == 200
    assert payload["started"] is False
    assert payload["resolved"]["learning_rate"] == pytest.approx(5e-4)


def test_train_start_against_a_bound_fake_trainer_actually_starts():
    """with a real TrainState hook, the submitted hyperparams reach backend_options and the run
    actually starts - proven against the fake backend, no torch training needed"""
    trainer = TrainState("fake")
    trainer.bind(
        [{"path": "a.jpg", "centres": [(1, 1)], "labels": ["x"], "width": 4, "height": 4}],
        {"x": 0},
    )
    server = serve(train_state_for=lambda: trainer)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, payload = _post(server, "/api/train-start", {"backend": "heatmap"})
        assert status == 200
        assert payload["started"] is True
        assert trainer.backend_options["optimizer"] == "adamw"
    finally:
        server.shutdown()
        thread.join(timeout=5)
