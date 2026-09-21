"""every http route the review tool answers, and nothing else.

THIN ON PURPOSE. a route parses its request, calls one method on a state object and returns what
comes back; logic that crept in here could only be tested through a socket. routes are entries in
two tables (`GET_ROUTES`, `POST_ROUTES`) plus the byte routes in `IMAGE_ROUTES`, so the full list
is data a test can read rather than a scrape of an if-chain.

A HOST ADDS TABS WITHOUT EDITING THIS FILE: pass `Tab` objects in `App.tabs`. a tab brings its own
routes (full paths, under `/api/` for json) and its own image routes; a path that collides with a
core route is refused when the handler is built, not silently shadowed.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from smolsmort.review import classscheme, housekeeping, paths, setconfig
from smolsmort.review.bases import BaseRootError
from smolsmort.review.playback_state import PlaybackState
from smolsmort.review.state import ReviewState
from smolsmort.review.tiles import TileError
from smolsmort.review.train_api import TrainApi

Handler = Callable[["App", dict], object]
ImageGetter = Callable[["App", str], bytes | None]


class RequestError(Exception):
    """a bad request: answered as json with this status, never as a dropped connection"""

    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


@dataclass
class Tab:
    """an extra tab a host registers: its routes, and nothing else it needs from the loop.

    `get` and `post` map a full path ("/api/mytab-things") to a function taking (app, query) or
    (app, payload) and returning a json-able dict. `images` maps a path prefix to a getter
    (app, name) -> bytes | None, plus its content type.
    """

    name: str
    get: dict[str, Handler] = field(default_factory=dict)
    post: dict[str, Handler] = field(default_factory=dict)
    images: dict[str, tuple[ImageGetter, str]] = field(default_factory=dict)


@dataclass
class App:
    """everything a route may touch, built once and shared by every request"""

    state: ReviewState
    trainer: TrainApi
    playback: PlaybackState | None = None
    tabs: list[Tab] = field(default_factory=list)
    # the ui served at / and /ui/. read at request time so a markup edit shows on refresh
    ui_dir: Path | None = None
    # name -> ClassScheme (the class-scheme seam). the default reads definitions from disk
    load_scheme: Callable[[str], object] = classscheme.load


GET_ROUTES: dict[str, Handler] = {}
POST_ROUTES: dict[str, Handler] = {}


def get(path: str):
    def register(fn: Handler) -> Handler:
        GET_ROUTES[path] = fn
        return fn

    return register


def post(path: str):
    def register(fn: Handler) -> Handler:
        POST_ROUTES[path] = fn
        return fn

    return register


def _int(query: dict, key: str, default: int) -> int:
    return int(query.get(key, default))


# ---------------------------------------------------------------- find


@get("/api/bindable-recordings")
def _bindable(app: App, q: dict):
    return app.state.bindable_recordings()


@get("/api/draw-frames")
def _draw_frames(app: App, q: dict):
    percent = q.get("percent")
    return app.state.draw_frames(_int(q, "sample", 40), int(percent) if percent else None)


@get("/api/crop-settings")
def _crop_settings(app: App, q: dict):
    return app.state.crop_settings()


@post("/api/crop-settings")
def _set_crop_settings(app: App, p: dict):
    try:
        return app.state.set_crop_settings(
            p.get("pad_x"),
            p.get("pad_y"),
            crop_mode=p.get("crop_mode"),
            crop_w=p.get("crop_w"),
            crop_h=p.get("crop_h"),
            aspect=p.get("aspect"),
        )
    except ValueError as exc:
        raise RequestError(str(exc)) from exc


@get("/api/tile-size")
def _tile_size(app: App, q: dict):
    return {"width": app.state.uniform_width, "height": app.state.uniform_height}


@post("/api/tile-size")
def _set_tile_size(app: App, p: dict):
    state = app.state
    state.uniform_height = max(4, state.uniform_height + int(p.get("height_delta", 0)))
    return {"width": state.uniform_width, "height": state.uniform_height}


@post("/api/height-delta")
def _height_delta(app: App, p: dict):
    delta = app.state.set_height_delta(p["index"], int(p.get("delta", 0)))
    return {"height_delta": delta, "effective_height": app.state.effective_height(p["index"])}


@post("/api/find-run")
def _find_run(app: App, p: dict):
    # find is drawing only: the boxes are the ones a human drew, nothing is mined
    # `set` names the training set being drawn for, whose size_mode decides if sizes are kept
    return app.state.save_drawn(
        p.get("boxes", []), negatives=p.get("negatives") or [], set_name=p.get("set") or None
    )


@post("/api/bind-recording")
def _bind_recording(app: App, p: dict):
    try:
        return app.state.bind_recording(p.get("kind"), p.get("name", ""))
    except FileNotFoundError as exc:
        raise RequestError(f"not found: {exc}", 404) from exc
    except ValueError as exc:
        raise RequestError(str(exc)) from exc


@post("/api/unbind-recording")
def _unbind_recording(app: App, p: dict):
    return app.state.unbind()


@post("/api/reviewed")
def _reviewed(app: App, p: dict):
    app.state.mark_reviewed(p["index"])
    return {"ok": True}


@get("/api/page")
def _page(app: App, q: dict):
    size = _int(q, "size", paths.PAGE_SIZE)
    return app.state.page(_int(q, "page", 0), size, q.get("hide_done", "0") == "1")


def _playback(app: App) -> PlaybackState:
    if app.playback is None:
        raise RequestError("no playback reader is configured for this server", 404)
    return app.playback


@get("/api/playback-sessions")
def _playback_sessions(app: App, q: dict):
    return _playback(app).sessions()


@post("/api/playback-open")
def _playback_open(app: App, p: dict):
    return _playback(app).open(p.get("name", ""))


@get("/api/playback-frame")
def _playback_frame(app: App, q: dict):
    return _playback(app).frame(float(q.get("t", 0)))


# ---------------------------------------------------------------- select


@get("/api/clusters")
def _clusters(app: App, q: dict):
    # any stale `k` is ignored: the grid sorts by class, there is nothing to choose
    return app.state.clusters()


@get("/api/pool-sources")
def _pool_sources(app: App, q: dict):
    return app.state.pool_sources()


@get("/api/pool-box")
def _pool_box(app: App, q: dict):
    return app.state.pool_box_info(q.get("name", ""))


@get("/api/source-state")
def _source_state(app: App, q: dict):
    return app.state.pool_source_state(q.get("tag", ""))


@get("/api/openable-datasets")
def _openable(app: App, q: dict):
    return app.state.openable_datasets()


@post("/api/open-dataset")
def _open_dataset(app: App, p: dict):
    return app.state.open_dataset(str(p.get("name", "")))


@post("/api/open-source")
def _open_source(app: App, p: dict):
    return app.state.open_pool_source(p.get("tag", ""))


@post("/api/close-source")
def _close_source(app: App, p: dict):
    # unsaved labels are reported ({"unsaved": N}) and the source left open; `discard` drops them
    return app.state.close_pool_source(p.get("tag", ""), discard=bool(p.get("discard")))


@post("/api/realign-pool-tile")
def _realign(app: App, p: dict):
    return app.state.realign_pool_tile(
        str(p.get("name", "")),
        float(p.get("left", paths.MARGIN_X)),
        float(p.get("top", paths.MARGIN_Y)),
    )


@post("/api/recut-pool")
def _recut(app: App, p: dict):
    return app.state.recut_pool()


@post("/api/exclude")
def _exclude(app: App, p: dict):
    # DECLARED, NOT TOGGLED. a caller that omits the field is a stale page still expecting a flip,
    # and answering by guessing would invert half of whatever it sent
    if "excluded" not in p:
        raise RequestError("exclude needs an 'excluded' boolean - it no longer toggles")
    # a selection is one press, so it is one request
    names = p.get("names") or [p.get("name")]
    wanted = bool(p["excluded"])
    done, unknown = [], []
    for one in names:
        if one is None:
            continue
        (unknown if app.state.set_excluded(one, wanted) is None else done).append(one)
    if not done and unknown:
        return _Reply({"error": "no such tile", "unknown": unknown}, 404)
    return {"ok": True, "excluded": wanted, "done": done, "unknown": unknown}


@post("/api/manual-label")
def _manual_label(app: App, p: dict):
    """record the class picked for one or several tiles IN MEMORY. nothing is written until
    /api/save-labels, and promotion reads disk only."""
    try:
        scheme = app.load_scheme(p["definition"])
        label = scheme.label_for(p.get("picked", {}))
    except classscheme.ClassDefError as exc:
        raise RequestError(str(exc)) from exc
    slug = getattr(scheme, "slug", str(p["definition"]))
    pending = None
    for one in p.get("names") or [p.get("name")]:
        if one is not None:
            pending = app.state.buffer_label(one, label, slug) or pending
    if pending is None:
        raise RequestError(f"no tile called {p.get('name')}")
    return {"ok": True, "label": label, "pending": pending}


@get("/api/frame-modes")
def _frame_modes(app: App, q: dict):
    recording = q.get("recording", "")
    _check_recording(recording)
    return {"frames": setconfig.read_frame_modes(recording)}


@post("/api/frame-mode")
def _frame_mode(app: App, p: dict):
    recording, frame = str(p.get("recording", "")), str(p.get("frame", ""))
    _check_recording(recording)
    if not frame or "exhaustive" not in p:
        raise RequestError("frame-mode needs a 'frame' and an 'exhaustive' boolean")
    setconfig.write_frame_mode(recording, Path(frame).name, bool(p["exhaustive"]))
    return {
        "ok": True,
        "recording": recording,
        "frame": Path(frame).name,
        "exhaustive": bool(p["exhaustive"]),
    }


def _check_recording(recording: str) -> None:
    """a frame mode belongs to a recording that exists under the sessions base"""
    root = paths.SESSIONS_DIR.resolve()
    target = (root / recording).resolve() if recording else root
    if not recording or root not in target.parents or not (target / "frames").is_dir():
        raise RequestError(f"no recording called {recording!r}", 404)


@post("/api/save-labels")
def _save_labels(app: App, p: dict):
    return app.state.save_labels()


@get("/api/labels-buffer")
def _labels_buffer(app: App, q: dict):
    return app.state.labels_buffer()


@post("/api/promote-training")
def _promote(app: App, p: dict):
    from smolsmort.review.promote import DEFAULT_SET_NAME

    return app.state.promote_to_training(p.get("name") or DEFAULT_SET_NAME, p.get("mode") or "new")


@get("/api/training-sets")
def _training_sets(app: App, q: dict):
    return app.state.training_sets()


@get("/api/classdefs")
def _classdefs(app: App, q: dict):
    wanted = q.get("name", "")
    names = classscheme.available()
    if not wanted:
        return {"definitions": names, "active": names[0] if names else None}
    try:
        return classscheme.load(wanted).as_json()
    except classscheme.ClassDefError as exc:
        raise RequestError(str(exc)) from exc


@post("/api/classdef-save")
def _classdef_save(app: App, p: dict):
    try:
        definition = classscheme.ClassDef(
            name=str(p.get("name", "")),
            dimensions=[
                classscheme.Dimension(name=d["name"], members=list(d.get("members", [])))
                for d in p.get("dimensions", [])
            ],
        )
        classscheme.parse(definition.as_toml())  # the same validation a loaded one gets
        classscheme.save(definition)
    except (classscheme.ClassDefError, KeyError, TypeError) as exc:
        raise RequestError(str(exc)) from exc
    return {"ok": True, **definition.as_json()}


# ---------------------------------------------------------------- train


@get("/api/train-info")
def _train_info(app: App, q: dict):
    return app.trainer.info()


@get("/api/train-classes")
def _train_classes(app: App, q: dict):
    return app.trainer.class_names()


@get("/api/train-frames")
def _train_frames(app: App, q: dict):
    return {"frames": app.trainer.frames()}


@get("/api/train-status")
def _train_status(app: App, q: dict):
    # whatever TrainState.status() reports goes through untouched
    return app.trainer.status()


@get("/api/window-floor")
def _window_floor(app: App, q: dict):
    return app.trainer.window_floor()


@get("/api/peak-distribution")
def _peak_distribution(app: App, q: dict):
    return app.trainer.peak_distribution()


@get("/api/saved-checkpoints")
def _saved_checkpoints(app: App, q: dict):
    return app.trainer.saved_checkpoints()


@get("/api/checkpoint-folders")
def _checkpoint_folders(app: App, q: dict):
    return app.trainer.checkpoint_folders(q.get("under", ""))


@get("/api/sweep-recordings")
def _sweep_recordings(app: App, q: dict):
    return app.trainer.sweep_recordings()


@get("/api/sweep-status")
def _sweep_status(app: App, q: dict):
    return app.trainer.sweep_status()


@post("/api/train-set-backend")
def _train_set_backend(app: App, p: dict):
    # body: {name, backend, size_mode?, capture_width?, downscale?}. size_mode defaults to native
    # for box, uniform otherwise; capture_width/downscale keep their stored value when left out
    # and clear it when null
    extra = {key: p[key] for key in ("capture_width", "downscale") if key in p}
    return app.trainer.set_backend(
        str(p.get("name", "")), str(p.get("backend", "")), p.get("size_mode"), **extra
    )


@post("/api/train-bind")
def _train_bind(app: App, p: dict):
    return app.trainer.bind(str(p.get("name", "")))


@post("/api/train-start")
def _train_start(app: App, p: dict):
    return app.trainer.start(p)


@post("/api/train-abort")
def _train_abort(app: App, p: dict):
    return app.trainer.abort()


@post("/api/load-checkpoint")
def _load_checkpoint(app: App, p: dict):
    return app.trainer.load_checkpoint(str(p.get("name", "")))


@post("/api/save-checkpoint")
def _save_checkpoint(app: App, p: dict):
    return app.trainer.save_checkpoint(p.get("name"), str(p.get("folder") or ""))


@post("/api/sweep-start")
def _sweep_start(app: App, p: dict):
    return app.trainer.start_sweep(
        str(p.get("recording", "")),
        max(1, min(100, int(p.get("percent", 100)))),
        float(p.get("min_score", 0.5)),
    )


@post("/api/sweep-send")
def _sweep_send(app: App, p: dict):
    return app.trainer.send_sweep(float(p.get("min_score", 0.5)))


# ---------------------------------------------------------------- shared


@get("/api/dir-list")
def _dir_list(app: App, q: dict):
    return app.state.dir_list()


@get("/api/dir-tree")
def _dir_tree(app: App, q: dict):
    try:
        return app.state.dir_tree(q.get("under") or None, q.get("root") or None)
    except BaseRootError as exc:
        raise RequestError(str(exc)) from exc


@post("/api/set-bases")
def _set_bases(app: App, p: dict):
    return {"ok": True, "bases": app.state.set_bases(p)}


@get("/api/housekeeping-recordings")
def _hk_recordings(app: App, q: dict):
    return {
        "recordings": [vars(r) for r in housekeeping.all_recordings()],
        "orphaned_sets": sorted(housekeeping.orphaned_sets()),
    }


@get("/api/housekeeping-assets")
def _hk_assets(app: App, q: dict):
    return housekeeping.assets_for(q.get("tag", ""))


def _housekeeping_action(action: Callable[[list[str]], dict], p: dict):
    try:
        return action(list(p.get("paths", [])))
    except housekeeping.HousekeepingError as exc:
        raise RequestError(str(exc)) from exc


@post("/api/housekeeping-archive")
def _hk_archive(app: App, p: dict):
    return _housekeeping_action(housekeeping.archive_paths, p)


@post("/api/housekeeping-delete")
def _hk_delete(app: App, p: dict):
    return _housekeeping_action(housekeeping.delete_paths, p)


@get("/api/tabs")
def _tabs(app: App, q: dict):
    """the extra tabs a host registered, so the page can offer them"""
    return {"tabs": [tab.name for tab in app.tabs]}


# ---------------------------------------------------------------- bytes


def _overlay(app: App, name: str) -> bytes | None:
    return app.trainer.overlay_bytes(int(name), -1)


# ONE TABLE for the byte routes: each takes the name out of the path, asks something for bytes,
# answers 404 when there are none and sends them otherwise. a getter that raises is "not there".
IMAGE_ROUTES: dict[str, tuple[ImageGetter, str]] = {
    "/draw-frame/": (lambda app, name: app.state.frame_bytes(name), "image/jpeg"),
    "/unsorted-thumb/": (lambda app, name: app.state.unsorted_thumb_bytes(name), "image/png"),
    "/tile/": (lambda app, name: app.state.pool_tile_bytes(name), "image/png"),
    "/crop/": (lambda app, name: app.state.crop_bytes(int(name)), "image/png"),
    "/train-overlay/": (_overlay, "image/png"),
}


class _Reply:
    """a json answer with a non-200 status, for the few routes that answer 404 with a body"""

    def __init__(self, body: dict, status: int):
        self.body, self.status = body, status


def check_tabs(tabs: list[Tab]) -> None:
    """refuse a tab route that would shadow a core one or another tab's"""
    seen_get, seen_post, seen_image = set(GET_ROUTES), set(POST_ROUTES), set(IMAGE_ROUTES)
    for tab in tabs:
        for seen, added in (
            (seen_get, tab.get),
            (seen_post, tab.post),
            (seen_image, tab.images),
        ):
            clash = seen & set(added)
            if clash:
                raise ValueError(f"tab {tab.name!r} redefines existing route(s): {sorted(clash)}")
            seen.update(added)


def make_handler(app: App) -> type[BaseHTTPRequestHandler]:
    check_tabs(app.tabs)
    get_routes = {**GET_ROUTES}
    post_routes = {**POST_ROUTES}
    image_routes = {**IMAGE_ROUTES}
    for tab in app.tabs:
        get_routes.update(tab.get)
        post_routes.update(tab.post)
        image_routes.update(tab.images)

    class ReviewHandler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass  # one line per request would drown the terminal; a host can subclass to log

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            # nothing this server sends is worth caching: a tile is re-cut in place under an
            # unchanged url, and a cached copy shows the OLD crop after a successful realign
            self.send_header("Cache-Control", "no-store, must-revalidate")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, payload, status: int = 200) -> None:
            if isinstance(payload, _Reply):
                payload, status = payload.body, payload.status
            self._send(status, json.dumps(payload).encode(), "application/json")

        def _payload(self) -> dict:
            """the request's json body as a dict. empty or unparseable is {} rather than an
            error: several routes legitimately take no body."""
            length = int(self.headers.get("Content-Length", 0))
            if not length:
                return {}
            try:
                body = json.loads(self.rfile.read(length)) or {}
            except json.JSONDecodeError:
                return {}
            return body if isinstance(body, dict) else {}

        def _guarded(self, dispatch: Callable[[], None]) -> None:
            """a route that raises answers NOTHING - the socket just closes and the page sees a
            network error it cannot explain. a bad REQUEST becomes a 400 naming the problem; a
            genuine fault inside the tool still propagates and stays visible."""
            try:
                dispatch()
            except RequestError as exc:
                self._json({"error": str(exc)}, exc.status)
            except (KeyError, TypeError, ValueError) as exc:
                self._json({"error": f"bad request for {self.path}: {exc}"}, 400)

        def do_GET(self) -> None:
            self._guarded(self._dispatch_get)

        def do_POST(self) -> None:
            self._guarded(self._dispatch_post)

        def _serve_image(self, path: str) -> bool:
            for prefix, (getter, content_type) in image_routes.items():
                if not path.startswith(prefix):
                    continue
                try:
                    data = getter(app, unquote(path[len(prefix) :]))
                except (TileError, OSError, ValueError, IndexError, KeyError):
                    # /crop/999999 used to raise out of the handler and kill the connection; a
                    # stale tab after rebinding asks for exactly that
                    data = None
                if data is None:
                    self._send(404, b"", "text/plain")
                else:
                    self._send(200, data, content_type)
                return True
            return False

        def _serve_ui_asset(self, name: str) -> None:
            """one file: the shared ones from ui_base first (so a stale local copy can never
            shadow the pinned one), then this tool's own from ui_dir, confined to it - resolve()
            collapses any "../" before the check, however it is spelled in the url."""
            data = None
            try:
                from ui_base import UiBaseError, read_asset
            except ImportError:
                pass
            else:
                try:
                    data = read_asset(name)
                except UiBaseError:
                    data = None
            if data is None and app.ui_dir is not None:
                target = (app.ui_dir / name).resolve()
                if app.ui_dir.resolve() in target.parents and target.is_file():
                    data = target.read_bytes()
            if data is None:
                self._send(404, b"", "text/plain")
                return
            kind = {".css": "text/css", ".js": "application/javascript"}
            self._send(200, data, kind.get(Path(name).suffix, "text/plain"))

        def _serve_index(self) -> None:
            page = app.ui_dir / "index.html" if app.ui_dir is not None else None
            if page is None or not page.is_file():
                self._json({"error": "no page installed - this server has no ui directory"}, 404)
                return
            body = (
                page.read_text(encoding="utf-8")
                .replace("{page_size}", str(paths.PAGE_SIZE))
                .replace("{margin_x}", str(paths.MARGIN_X))
                .replace("{margin_y}", str(paths.MARGIN_Y))
            )
            self._send(200, body.encode(), "text/html")

        def _dispatch_get(self) -> None:
            parsed = urlparse(self.path)
            query = {k: v[0] for k, v in parse_qs(parsed.query).items()}
            if parsed.path == "/":
                self._serve_index()
            elif parsed.path.startswith("/ui/"):
                self._serve_ui_asset(unquote(parsed.path[len("/ui/") :]))
            elif parsed.path in get_routes:
                self._json(get_routes[parsed.path](app, query))
            elif not self._serve_image(parsed.path):
                self._send(404, b"", "text/plain")

        def _dispatch_post(self) -> None:
            path = urlparse(self.path).path
            handler = post_routes.get(path)
            if handler is None:
                self._send(404, b"", "text/plain")
                return
            self._json(handler(app, self._payload()))

    return ReviewHandler
