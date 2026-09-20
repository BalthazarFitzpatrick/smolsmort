"""a minimal stdlib server for the hyperparams dropdown, standalone until a real train tab exists.

WHY STANDALONE. smolsmort has trainers and backends (smolsmort.detect, smolsmort.boxes, the
ModelBackend seam) but no served web tool yet - the one that used to have a train tab lives only in
snapshot/, which imports `parent.*` modules that were deliberately never copied here and cannot run.
Rather than build the dropdown into code that cannot start, this serves it on its own: the same
ui_base Menu, the same /api shape a future train tab's routes.py would offer, so wiring it into a
real tab later is a page move, not a rewrite.

WHAT IT ACTUALLY TRAINS. `bind_examples`/`train_state_for` are optional hooks - pass a callable that
returns a `smolsmort.review.train.TrainState` already bound to real examples, and /api/train-start
drives it for real. With no hook, the endpoint only resolves and clamps the submitted hyperparams
and reports the model size they would build - which is what a menu with no examples to bind can
honestly do.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from ui_base import UiBaseError, read_asset

from smolsmort.review_ui.logic import RequestError, menu_options, resolve_hyperparams

STATIC = Path(__file__).resolve().parent / "static"
_CONTENT_TYPES = {".html": "text/html", ".js": "text/javascript", ".css": "text/css"}


def _static_file(path: str) -> Path | None:
    """the page or one of its scripts and styles, by url path - None for anything else"""
    name = "index.html" if path == "/" else path.removeprefix("/")
    target = (STATIC / name).resolve()
    inside = target.parent == STATIC
    return target if inside and target.suffix in _CONTENT_TYPES and target.is_file() else None


def make_handler(train_state_for: Callable[[], object] | None = None):
    """a request handler bound to an optional TrainState factory - a fresh TrainState per call,
    since a training run mutates it and a shared instance would let one viewer's run cancel
    another's."""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args) -> None:  # quiet by default, same as the review tool's own
            pass

        def _json(self, payload: dict, status: int = 200) -> None:
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _payload(self) -> dict:
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length) if length else b"{}"
            return json.loads(raw) if raw else {}

        def do_GET(self) -> None:  # noqa: N802 - http.server's own naming
            path = self.path.split("?", 1)[0]
            static = _static_file(path)
            if static is not None:
                body = static.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", _CONTENT_TYPES[static.suffix])
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
                return
            if path.startswith("/ui/"):
                name = path.removeprefix("/ui/")
                try:
                    body = read_asset(name)
                except UiBaseError as exc:
                    # not a shared file: this page's own scripts are addressed under /ui/ too
                    own = _static_file("/" + name)
                    if own is None:
                        self._json({"error": str(exc)}, status=404)
                        return
                    body = own.read_bytes()
                self.send_response(200)
                self.send_header(
                    "Content-Type", "text/javascript" if path.endswith(".js") else "text/css"
                )
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if path == "/api/hyperparams":
                from urllib.parse import parse_qs, urlparse

                backend = parse_qs(urlparse(self.path).query).get("backend", ["heatmap"])[0]
                try:
                    self._json(menu_options(backend))
                except RequestError as exc:
                    self._json({"error": str(exc)}, status=400)
                return
            self._json({"error": "not found"}, status=404)

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/api/train-start":
                self._json({"error": "not found"}, status=404)
                return
            payload = self._payload()
            backend = payload.get("backend", "heatmap")
            try:
                resolved = resolve_hyperparams(backend, payload)
            except RequestError as exc:
                self._json({"error": str(exc)}, status=400)
                return
            if train_state_for is None:
                self._json({"resolved": resolved, "started": False})
                return
            trainer = train_state_for()
            trainer.backend_options.update(
                {k: v for k, v in resolved.items() if k not in ("backend", "parameters")}
            )
            result = trainer.start()
            self._json({"resolved": resolved, "started": "error" not in result, **result})

    return Handler


def serve(
    port: int = 0, train_state_for: Callable[[], object] | None = None
) -> ThreadingHTTPServer:
    """starts listening and returns the server - caller runs `serve_forever` on a thread and
    `shutdown` to stop, same lifecycle as the review tool's own http.server"""
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(train_state_for))
    return server
