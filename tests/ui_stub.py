"""a stub server for the review page: the real static files plus canned json for the api contract,
so the browser check can run without the routes. state is mutable through POST /stub/set."""

from __future__ import annotations

import io
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from PIL import Image
from ui_base import read_asset

from smolsmort.review_ui.server import _CONTENT_TYPES, _static_file

TREES = {
    "tiles": {
        "/t": {"parent": None, "entries": [("a", True), ("set1", False)]},
        "/t/a": {"parent": "/t", "entries": [("deep", False)]},
    },
    "checkpoints": {
        "/c": {"parent": None, "entries": [("runs", True), ("best.pt", False)]},
        "/c/runs": {"parent": "/c", "entries": [("r1.pt", False)]},
    },
}
ROOTS = {"tiles": "/t", "checkpoints": "/c", "recordings": "/r"}


def fresh_state() -> dict:
    return {
        "log": [],
        "pending": 0,
        "crop": {
            "pad_x": 0,
            "pad_y": 0,
            "crop_mode": "percent",
            "crop_w": 100,
            "crop_h": 100,
            "aspect": "free",
        },
        "train": {
            "running": False,
            "epoch": 0,
            "epochs": 0,
            "checkpoint_count": 0,
            "history": [],
            "train_loss": None,
            "val_loss": None,
            "test_loss": None,
        },
    }


def png_bytes(width: int, height: int, color=(60, 90, 120)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buffer, "PNG")
    return buffer.getvalue()


def start_stub() -> tuple[ThreadingHTTPServer, dict]:
    state = fresh_state()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args) -> None:
            pass

        def _send(self, body: bytes, content_type: str) -> None:
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, payload) -> None:
            self._send(json.dumps(payload).encode(), "application/json")

        def do_GET(self) -> None:  # noqa: N802 - http.server's own naming
            url = urlparse(self.path)
            query = {k: v[0] for k, v in parse_qs(url.query).items()}
            path = url.path
            state["log"].append(("GET", self.path, None))
            static = _static_file(path)
            if static is not None:
                self._send(static.read_bytes(), _CONTENT_TYPES[static.suffix])
            elif path.startswith("/ui/"):
                body = read_asset(path.removeprefix("/ui/"))
                self._send(body, "text/javascript" if path.endswith(".js") else "text/css")
            elif path.startswith("/draw-frame/"):
                self._send(png_bytes(320, 200), "image/png")
            elif path.startswith("/unsorted-thumb/"):
                self._send(png_bytes(48, 48, (120, 90, 60)), "image/png")
            else:
                self._json(self._api(path.removeprefix("/api/"), query))

        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
            path = urlparse(self.path).path
            state["log"].append(("POST", path, body))
            if path == "/stub/set":
                for key, value in body.items():
                    if isinstance(state.get(key), dict):
                        state[key].update(value)
                    else:
                        state[key] = value
                self._json({})
                return
            self._json(self._api_post(path.removeprefix("/api/"), body))

        def _api(self, name: str, query: dict):
            if name == "crop-settings":
                return state["crop"]
            if name == "draw-frames":
                return {
                    "ready": True,
                    "dataset": "rec1",
                    "frames": ["0001.png", "0002.png", "0003.png"],
                    "boxes": {},
                }
            if name == "bindable-recordings":
                return {"rows": [], "current": "rec1"}
            if name == "clusters":
                items = [
                    {"name": f"tile{i}.png", "excluded": False, "assigned": None} for i in range(6)
                ]
                return {
                    "clusters": [{"label": "", "items": items}],
                    "boxes_loaded": 6,
                    "classes_assigned": 1,
                }
            if name == "classdefs":
                if "name" in query:
                    return {
                        "name": "demo",
                        "slug": "demo",
                        "dimensions": [{"name": "kind", "members": ["alpha", "beta"]}],
                    }
                return {"active": "demo", "definitions": ["demo"]}
            if name == "labels-buffer":
                return {"pending": state["pending"]}
            if name == "pool-sources":
                return {"sources": [{"tag": "set1", "tiles": 6, "closed": False}]}
            if name == "openable-datasets":
                return {"datasets": []}
            if name == "training-sets":
                return {"sets": []}
            if name == "train-info":
                return {"set": None, "device": "cpu", "model_exists": False, "backend": "heatmap"}
            if name == "window-floor":
                return {"floor": 64, "box": [10, 10], "downscale": 2, "default": 256}
            if name == "train-status":
                return state["train"]
            if name == "saved-checkpoints":
                return {
                    "weights": [
                        {"name": "ck-new", "final": False},
                        {"name": "ck-old", "final": False},
                        {"name": "ck-final", "final": True},
                    ]
                }
            if name == "dir-tree":
                root = query.get("root", "tiles")
                here = query.get("under") or ROOTS[root]
                node = TREES.get(root, {}).get(here, {"parent": None, "entries": []})
                entries = [
                    {"name": n, "path": f"{here}/{n}", "has_children": kids}
                    for n, kids in node["entries"]
                ]
                return {"here": here, "parent": node["parent"], "entries": entries}
            if name == "dir-list":
                return {"bases": {"root": "/base", "recordings": "/base/r", "tiles": ["/base/t"]}}
            if name == "housekeeping-recordings":
                return {"recordings": []}
            if name == "sweep-recordings":
                return {"recordings": []}
            return {}

        def _api_post(self, name: str, body: dict):
            if name == "crop-settings":
                state["crop"].update(body)
                return state["crop"]
            if name == "manual-label":
                state["pending"] += len(body.get("names", []))
                return {"pending": state["pending"]}
            if name == "save-labels":
                saved, state["pending"] = state["pending"], 0
                return {"saved": saved}
            if name == "close-source":
                if state["pending"] and not body.get("discard"):
                    return {"unsaved": state["pending"]}
                state["pending"] = 0
                return {"closed": 6, "unsaved": 0}
            return {}

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, state
