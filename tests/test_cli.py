"""`uv run smolsmort` - the cli starts the real review server and serves the new page."""

from __future__ import annotations

import json
import threading
import urllib.request

import pytest

import smolsmort.cli as cli_module
from smolsmort.cli import main
from smolsmort.review import paths
from smolsmort.review.server import serve as real_serve


@pytest.fixture
def cli_port(tmp_path, monkeypatch):
    # settings files land beside the labels dir, so keep that inside tmp_path
    monkeypatch.setattr(paths, "LABELS_DIR", tmp_path / "training" / "boxes")
    holder: dict = {}
    started = threading.Event()

    def capturing_serve(app, host="127.0.0.1", port=0):
        server = real_serve(app, host, port)
        holder["server"] = server
        started.set()
        return server

    monkeypatch.setattr(cli_module, "serve", capturing_serve)
    thread = threading.Thread(target=main, args=(["--port", "0"],), daemon=True)
    thread.start()
    assert started.wait(timeout=5)
    server = holder["server"]
    port = server.server_address[1]
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.shutdown()
        thread.join(timeout=5)


def get(base: str, path: str) -> tuple[int, bytes]:
    with urllib.request.urlopen(base + path, timeout=10) as resp:
        return resp.status, resp.read()


def test_main_serves_the_review_page_on_the_given_port(cli_port):
    status, body = get(cli_port, "/")
    assert status == 200 and b"smolsmort review" in body


def test_the_page_scripts_and_shared_assets_are_reachable(cli_port):
    for name in ("core.js", "tab_find.js", "tab_select.js", "tab_train.js", "review.css"):
        status, body = get(cli_port, f"/ui/{name}")
        assert status == 200 and body, name
    status, body = get(cli_port, "/ui/menu.js")
    assert status == 200 and b"class Menu" in body


def test_the_real_routes_and_the_config_menu_answer(cli_port):
    _, body = get(cli_port, "/api/crop-settings")
    assert "crop_mode" in json.loads(body)
    _, body = get(cli_port, "/api/hyperparams?backend=heatmap")
    assert json.loads(body)["optimizers"] == ["adamw", "sgd"]
