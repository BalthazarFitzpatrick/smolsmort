"""`uv run smolsmort` - the cli starts the review ui server and serves its page."""

from __future__ import annotations

import threading
import urllib.request

import smolsmort.cli as cli_module
from smolsmort.cli import main
from smolsmort.review_ui.server import serve as real_serve


def test_main_serves_the_review_ui_on_the_given_port(monkeypatch):
    port_holder: dict[str, int] = {}
    started = threading.Event()

    def capturing_serve(port=0, train_state_for=None):
        server = real_serve(port=port, train_state_for=train_state_for)
        port_holder["port"] = server.server_address[1]
        started.set()
        return server

    monkeypatch.setattr(cli_module, "serve", capturing_serve)
    thread = threading.Thread(target=main, args=(["--port", "0"],), daemon=True)
    thread.start()
    assert started.wait(timeout=5)
    port = port_holder["port"]
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=5) as resp:
        assert resp.status == 200
