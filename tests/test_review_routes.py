"""the review tool through a real socket: every route answers, the guard holds, and the behaviour
changes (buffered labels, counts, crop rule, rooted trees) are visible over http.

fake seams only - a torch-free backend and a scripted playback reader (review_world.py)."""

from __future__ import annotations

import json
import socket
import threading
import time
import urllib.error
import urllib.request
from types import SimpleNamespace

import pytest
import review_world
from review_world import FakeReader, drawn_boxes, make_world, save_definition

from smolsmort.review import paths, routes
from smolsmort.review.playback_state import PlaybackState
from smolsmort.review.server import build_app, serve


@pytest.fixture(autouse=True)
def _world_backend():
    review_world.register_world_backend()
    try:
        yield
    finally:
        review_world.unregister_world_backend()


@pytest.fixture
def world(tmp_path, monkeypatch):
    world = make_world(tmp_path, monkeypatch, recordings=("rec_a", "set/two"))
    (world.ui / "index.html").write_text(
        "<!doctype html><html><body>page {page_size} {margin_x}</body></html>"
    )
    (world.ui / "app.js").write_text("console.log(1)")
    return world


def _start(world, **extra):
    app = build_app(
        backend="world",
        ui_dir=world.ui,
        pool=world.tiles,
        bases_file=world.root / "bases.json",
        crop_file=world.root / "crop.json",
        playback=PlaybackState(FakeReader()),
        **extra,
    )
    httpd = serve(app)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    address = httpd.server_address
    for _ in range(50):
        with socket.socket() as probe:
            if probe.connect_ex(address) == 0:
                break
        time.sleep(0.05)
    return app, httpd, f"http://127.0.0.1:{address[1]}"


@pytest.fixture
def server(world):
    """the real ThreadingHTTPServer on an ephemeral port - never a fixed one"""
    app, httpd, url = _start(world)
    yield app, url
    httpd.shutdown()
    httpd.server_close()


def get(base: str, path: str) -> tuple[int, bytes]:
    try:
        with urllib.request.urlopen(base + path, timeout=20) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def post(base: str, path: str, payload: dict | None = None) -> tuple[int, bytes]:
    """a POST that reports what came back, including nothing at all for a dropped connection"""
    request = urllib.request.Request(
        base + path,
        data=json.dumps(payload or {}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()
    except urllib.error.URLError:
        return 0, b""


def jget(base, path):
    status, body = get(base, path)
    assert status == 200, f"{path} answered {status}: {body[:300]!r}"
    return json.loads(body)


def jpost(base, path, payload=None):
    status, body = post(base, path, payload)
    assert status == 200, f"{path} answered {status}: {body[:300]!r}"
    return json.loads(body)


# ---------------------------------------------------------------- the route tables


def test_the_route_tables_are_not_empty():
    """a table that silently held nothing would make every test below vacuously pass"""
    assert len(routes.GET_ROUTES) > 25 and len(routes.POST_ROUTES) > 25
    for path in (
        "/api/find-run",
        "/api/manual-label",
        "/api/save-labels",
        "/api/promote-training",
        "/api/train-start",
        "/api/close-source",
        "/api/set-bases",
    ):
        assert path in routes.POST_ROUTES
    for path in ("/api/labels-buffer", "/api/clusters", "/api/dir-tree", "/api/train-status"):
        assert path in routes.GET_ROUTES


def test_the_removed_routes_are_gone():
    gone = ("smart-center", "library", "vlm", "interface", "nav", "camera", "control", "script")
    for path in [*routes.GET_ROUTES, *routes.POST_ROUTES, *routes.IMAGE_ROUTES]:
        assert not any(word in path for word in gone), path


@pytest.mark.parametrize("route", sorted(routes.GET_ROUTES))
def test_every_get_route_answers_unbound_without_a_server_error(server, route):
    """UNBOUND ON PURPOSE: no dataset, no model - the state the server starts in. a route that
    only works once something is loaded must say so in json, not raise."""
    _, url = server
    status, body = get(url, route)
    assert status < 500, f"{route} returned {status}: {body[:300]!r}"
    if status == 200:
        json.loads(body)


@pytest.mark.parametrize("route", sorted(routes.POST_ROUTES))
def test_no_post_route_can_kill_the_connection(server, route):
    """A ROUTE THAT RAISES ANSWERS NOTHING - the socket closes and the page sees a network error
    it cannot explain. an empty body reaches every route."""
    _, url = server
    status, body = post(url, route, {})
    assert status in (200, 400, 404), f"{route} answered {status}"
    assert body, f"{route} answered nothing at all"


def test_an_unknown_route_is_a_404_not_a_crash(server):
    _, url = server
    assert get(url, "/api/no-such-endpoint")[0] == 404
    assert post(url, "/api/no-such-endpoint")[0] == 404


def test_a_post_with_no_body_is_answered_rather_than_dropped(server):
    _, url = server
    for path in ("/api/save-checkpoint", "/api/promote-training", "/api/crop-settings"):
        request = urllib.request.Request(url + path, data=b"", method="POST")
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                status = response.status
        except urllib.error.HTTPError as exc:
            status = exc.code
        assert status == 200, path


def test_a_bad_request_says_what_was_missing(server):
    _, url = server
    status, body = post(url, "/api/height-delta", {})
    assert status == 400 and "index" in json.loads(body)["error"]


def test_a_bad_query_is_a_400_never_a_dropped_socket(server):
    _, url = server
    for path in (
        "/api/playback-frame?t=abc",
        "/api/draw-frames?percent=abc",
        "/api/page?page=abc",
        "/crop/abc",
    ):
        status, body = get(url, path)
        assert status in (200, 400, 404), path
        if status == 400:
            assert body


def test_every_image_route_answers_404_rather_than_dropping_the_connection(server):
    _, url = server
    for path in (
        "/draw-frame/nope.png",
        "/unsorted-thumb/nope",
        "/tile/nope",
        "/crop/999999",
        "/crop/not-a-number",
        "/train-overlay/0",
    ):
        assert get(url, path)[0] == 404, path


def test_every_response_forbids_caching(server):
    """a tile is re-cut in place under an unchanged url, so a cached copy shows the OLD crop"""
    _, url = server
    for path in ("/", "/api/train-info", "/tile/nope"):
        try:
            with urllib.request.urlopen(url + path, timeout=20) as response:
                headers = response.headers
        except urllib.error.HTTPError as exc:
            headers = exc.headers
        assert "no-store" in headers.get("Cache-Control", ""), path


# ---------------------------------------------------------------- the page and its assets


def test_the_index_is_served_from_the_ui_directory_with_its_slots_filled(server):
    _, url = server
    status, body = get(url, "/")
    assert status == 200 and f"page {paths.PAGE_SIZE} {paths.MARGIN_X}" in body.decode()


def test_the_index_is_404_json_when_no_ui_is_installed(world):
    world.ui.joinpath("index.html").unlink()
    app, httpd, url = _start(world)
    try:
        status, body = get(url, "/")
        assert status == 404 and "no page" in json.loads(body)["error"]
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_own_assets_are_served_and_traversal_is_not(server, world):
    _, url = server
    assert get(url, "/ui/app.js")[1] == b"console.log(1)"
    (world.root / "secret.txt").write_text("nope")
    for attempt in ("/ui/../secret.txt", "/ui/%2e%2e/secret.txt", "/ui/..%2fsecret.txt"):
        assert get(url, attempt)[0] == 404, attempt


# ---------------------------------------------------------------- host tabs


def test_a_host_tab_registers_its_own_routes(world):
    def things(app, query):
        return {"things": [query.get("q")], "bound": app.state.session_tag}

    def make(app, payload):
        return {"made": payload["name"]}

    tab = routes.Tab(
        "extras",
        get={"/api/extras-things": things},
        post={"/api/extras-make": make},
        images={"/extras-img/": (lambda app, name: name.encode(), "image/png")},
    )
    _app, httpd, url = _start(world, tabs=[tab])
    try:
        assert jget(url, "/api/extras-things?q=x") == {"things": ["x"], "bound": ""}
        assert jpost(url, "/api/extras-make", {"name": "n"}) == {"made": "n"}
        assert get(url, "/extras-img/abc") == (200, b"abc")
        assert jget(url, "/api/tabs") == {"tabs": ["extras"]}
        assert post(url, "/api/extras-make", {})[0] == 400, "a missing field is still a 400"
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_a_tab_may_not_shadow_a_core_route(world):
    clash = routes.Tab("bad", get={"/api/clusters": lambda app, q: {}})
    with pytest.raises(ValueError, match="redefines"):
        build_app(backend="world", ui_dir=world.ui, pool=world.tiles, tabs=[clash])
        routes.make_handler(
            build_app(backend="world", ui_dir=world.ui, pool=world.tiles, tabs=[clash])
        )


def test_two_tabs_may_not_share_a_route(world):
    one = routes.Tab("one", post={"/api/shared": lambda app, p: {}})
    two = routes.Tab("two", post={"/api/shared": lambda app, p: {}})
    with pytest.raises(ValueError, match="redefines"):
        routes.check_tabs([one, two])


# ---------------------------------------------------------------- find over http


def _bind_and_draw(url):
    jpost(url, "/api/bind-recording", {"kind": "session", "name": "rec_a"})
    return jpost(url, "/api/find-run", {"boxes": drawn_boxes()})


def test_binding_and_drawing_over_http(server):
    _, url = server
    listing = jget(url, "/api/bindable-recordings")
    assert {r["name"] for r in listing["rows"]} == {"rec_a", "set/two"}
    assert jpost(url, "/api/bind-recording", {"kind": "session", "name": "rec_a"})["ok"]
    frames = jget(url, "/api/draw-frames?percent=100")
    assert frames["ready"] and len(frames["frames"]) == 3
    assert get(url, "/draw-frame/f00.png")[0] == 200
    saved = jpost(url, "/api/find-run", {"boxes": drawn_boxes()})
    assert saved["count"] == 6 and saved["tiles"] == 6
    assert get(url, "/crop/0")[0] == 200
    assert jget(url, "/api/page?size=4")["total"] == 6
    assert jpost(url, "/api/unbind-recording")["ok"]


def test_binding_something_that_is_not_there_is_a_404_and_a_bad_kind_a_400(server):
    _, url = server
    assert post(url, "/api/bind-recording", {"kind": "session", "name": "ghost"})[0] == 404
    assert post(url, "/api/bind-recording", {"kind": "banana", "name": "x"})[0] == 400


def test_height_delta(server):
    _, url = server
    _bind_and_draw(url)
    out = jpost(url, "/api/height-delta", {"index": 0, "delta": 2})
    assert out["height_delta"] == 2 and out["effective_height"] == 16
    assert jpost(url, "/api/reviewed", {"index": 0})["ok"]
    assert jget(url, "/api/page")["total"] == 5


def test_crop_settings_carry_the_new_fields_and_still_take_the_old_ones(server, world):
    _, url = server
    assert jget(url, "/api/crop-settings")["crop_mode"] == "percent"
    legacy = jpost(url, "/api/crop-settings", {"pad_x": 0.5, "pad_y": 1.0})
    assert (legacy["pad_x"], legacy["pad_y"], legacy["aspect"]) == (0.5, 1.0, "free")
    new = jpost(
        url,
        "/api/crop-settings",
        {"crop_mode": "absolute", "crop_w": 80, "crop_h": 50, "aspect": "free"},
    )
    assert (new["crop_mode"], new["crop_w"], new["crop_h"]) == ("absolute", 80, 50)
    saved = json.loads((world.root / "crop.json").read_text())
    assert saved["crop_mode"] == "absolute" and saved["crop_w"] == 80
    status, body = post(url, "/api/crop-settings", {"crop_mode": "diagonal"})
    assert status == 400 and "crop_mode" in json.loads(body)["error"]


def test_the_crop_rule_reaches_the_tiles_a_save_cuts(server, world):
    import numpy as np

    _, url = server
    jpost(url, "/api/crop-settings", {"crop_mode": "absolute", "crop_w": 60, "crop_h": 40})
    _bind_and_draw(url)
    with np.load(world.tiles / "rec_a_k00000.npz") as data:
        assert data["rgb"].shape[:2] == (40, 60)


def test_playback_routes_go_through_the_reader_seam(server):
    _, url = server
    assert jget(url, "/api/playback-sessions") == {"sessions": [{"name": "rec_a"}]}
    opened = jpost(url, "/api/playback-open", {"name": "rec_a"})
    assert opened["name"] == "rec_a" and opened["labelled"] == 1
    assert jget(url, "/api/playback-frame?t=1.5") == {"when": 11.5}


def test_playback_without_a_reader_says_so(world):
    app = build_app(backend="world", ui_dir=world.ui, pool=world.tiles)
    httpd = serve(app)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        status, body = get(f"http://127.0.0.1:{httpd.server_address[1]}", "/api/playback-sessions")
        assert status == 404 and "reader" in json.loads(body)["error"]
    finally:
        httpd.shutdown()
        httpd.server_close()


# ---------------------------------------------------------------- select over http


def _pool(url) -> list[str]:
    _bind_and_draw(url)
    return [i["name"] for c in jget(url, "/api/clusters")["clusters"] for i in c["items"]]


def test_manual_label_buffers_and_save_labels_flushes(server, world):
    _, url = server
    save_definition()
    names = _pool(url)
    out = jpost(
        url,
        "/api/manual-label",
        {"names": names[:2], "definition": "kinds", "picked": {"kind": "alpha"}},
    )
    assert out["pending"] == 2 and out["label"] == "alpha"
    assert not (world.tiles / "_labels.json").exists(), "manual-label must not write to disk"
    buffered = jget(url, "/api/labels-buffer")
    assert buffered == {"pending": 2, "datasets": [{"tag": "rec_a", "pending": 2}]}
    assert jpost(url, "/api/save-labels") == {"saved": 2}
    assert (world.tiles / "_labels.json").is_file()
    assert jget(url, "/api/labels-buffer") == {"pending": 0, "datasets": []}


def test_manual_label_refuses_bad_picks_and_unknown_tiles(server):
    _, url = server
    save_definition()
    names = _pool(url)
    bad = post(
        url,
        "/api/manual-label",
        {"name": names[0], "definition": "kinds", "picked": {"kind": "gamma"}},
    )
    assert bad[0] == 400 and "gamma" in json.loads(bad[1])["error"]
    status, body = post(
        url,
        "/api/manual-label",
        {"name": "ghost_k00000", "definition": "kinds", "picked": {"kind": "alpha"}},
    )
    assert status == 400 and "ghost" in json.loads(body)["error"]
    assert post(url, "/api/manual-label", {"name": names[0]})[0] == 400, "definition is required"


def test_closing_a_dataset_reports_unsaved_and_drops_only_on_discard(server):
    _, url = server
    save_definition()
    names = _pool(url)
    jpost(
        url,
        "/api/manual-label",
        {"name": names[0], "definition": "kinds", "picked": {"kind": "beta"}},
    )
    assert jpost(url, "/api/close-source", {"tag": "rec_a"}) == {"tag": "rec_a", "unsaved": 1}
    assert jget(url, "/api/labels-buffer")["pending"] == 1
    assert jget(url, "/api/source-state?tag=rec_a")["unsaved"] == 1
    out = jpost(url, "/api/close-source", {"tag": "rec_a", "discard": True})
    assert out["closed"] == 6
    assert jget(url, "/api/labels-buffer")["pending"] == 0
    assert jpost(url, "/api/open-source", {"tag": "rec_a"})["tiles"] == 6


def test_clusters_report_counts(server):
    _, url = server
    save_definition()
    names = _pool(url)
    assert jget(url, "/api/clusters")["counts"] == {"boxes_loaded": 6, "classes_assigned": 0}
    jpost(
        url,
        "/api/manual-label",
        {"names": names[:3], "definition": "kinds", "picked": {"kind": "alpha"}},
    )
    assert jget(url, "/api/clusters")["counts"] == {"boxes_loaded": 6, "classes_assigned": 3}


def test_promote_over_http_reads_disk_only(server, world):
    _, url = server
    save_definition()
    names = _pool(url)
    for i, name in enumerate(names):
        jpost(
            url,
            "/api/manual-label",
            {"name": name, "definition": "kinds", "picked": {"kind": "alpha" if i % 2 else "beta"}},
        )
    assert "nothing to promote" in jpost(url, "/api/promote-training", {"name": "s"})["error"]
    jpost(url, "/api/save-labels")
    out = jpost(url, "/api/promote-training", {"name": "s"})
    assert out["rows"] == 6 and out["classes"] == {"alpha": 3, "beta": 3}
    assert jpost(url, "/api/promote-training", {"name": "s"})["exists"] is True
    assert jpost(url, "/api/promote-training", {"name": "s", "mode": "merge"})["mode"] == "merge"
    assert [s["name"] for s in jget(url, "/api/training-sets")["sets"]] == ["s"]


def test_exclude_is_declarative_over_http(server):
    _, url = server
    names = _pool(url)
    status, body = post(url, "/api/exclude", {"name": names[0]})
    assert status == 400 and "excluded" in json.loads(body)["error"]
    out = jpost(url, "/api/exclude", {"names": names[:2], "excluded": True})
    assert out["done"] == names[:2]
    status, body = post(url, "/api/exclude", {"names": ["a_k00000", "b_k00001"], "excluded": True})
    assert status == 404 and sorted(json.loads(body)["unknown"]) == ["a_k00000", "b_k00001"]
    assert post(url, "/api/exclude", {"name": "a_k00000", "excluded": True})[0] == 404
    assert jget(url, "/api/clusters")["clusters"][-1]["label"] == "not a class"


def test_pool_box_thumb_realign_and_recut(server):
    _, url = server
    names = _pool(url)
    assert jget(url, f"/api/pool-box?name={names[0]}")["rect"]["left"] == paths.MARGIN_X
    assert get(url, f"/tile/{names[0]}")[0] == 200
    assert get(url, f"/unsorted-thumb/{names[0]}")[0] == 200
    moved = jpost(url, "/api/realign-pool-tile", {"name": names[0], "left": 35, "top": 15})
    assert moved["name"] == names[0]
    assert jpost(url, "/api/recut-pool")["recut"][0]["tiles"] == 6
    assert "error" in jget(url, "/api/pool-box?name=ghost_k00000")


def test_openable_datasets_and_open_dataset(server, world):
    _, url = server
    _pool(url)
    jpost(url, "/api/close-source", {"tag": "rec_a"})
    for path in world.tiles.glob("*.npz"):
        path.unlink()
    listed = jget(url, "/api/openable-datasets")["datasets"]
    assert listed[0]["tag"] == "rec_a"
    assert jpost(url, "/api/open-dataset", {"name": listed[0]["file"]})["tiles"] == 6
    assert jget(url, "/api/pool-sources")["sources"][0]["tag"] == "rec_a"


def test_class_definitions_over_http(server):
    _, url = server
    assert jget(url, "/api/classdefs") == {"definitions": [], "active": None}
    saved = jpost(
        url,
        "/api/classdef-save",
        {"name": "Kinds", "dimensions": [{"name": "kind", "members": ["a", "b"]}]},
    )
    assert saved["classes"] == ["a", "b"]
    assert jget(url, "/api/classdefs")["definitions"] == ["kinds"]
    assert jget(url, "/api/classdefs?name=kinds")["slug"] == "kinds"
    assert (
        post(url, "/api/classdefs?name=nope", {})[0] == 404
        or get(url, "/api/classdefs?name=nope")[0] == 400
    )
    assert post(url, "/api/classdef-save", {"name": ""})[0] == 400


# ---------------------------------------------------------------- shared: trees, bases, housekeeping


def test_dir_tree_roots_over_http(server, world):
    _, url = server
    (world.checkpoints / "runs").mkdir()
    assert [e["name"] for e in jget(url, "/api/dir-tree?root=checkpoints")["entries"]] == ["runs"]
    assert {e["name"] for e in jget(url, "/api/dir-tree?root=recordings")["entries"]} == {
        "rec_a",
        "set",
    }
    assert jget(url, "/api/dir-tree?root=tiles")["root"] == str(world.tiles.resolve())
    outside = jget(url, f"/api/dir-tree?root=tiles&under={world.sessions}")
    assert outside["refused"] is True and outside["here"] == str(world.tiles.resolve())
    status, body = get(url, "/api/dir-tree?root=nonsense")
    assert status == 400 and "root must be one of" in json.loads(body)["error"]
    assert "entries" in jget(url, "/api/dir-tree"), "no root still lists the settings root"


def test_set_bases_and_dir_list_over_http(server, world, tmp_path):
    _, url = server
    out = jpost(url, "/api/set-bases", {"labels": str(tmp_path / "lab"), "bogus": "x"})
    assert out["bases"]["labels"] == str(tmp_path / "lab") and "bogus" not in out["bases"]
    assert tmp_path / "lab" == paths.LABELS_DIR
    assert len(jget(url, "/api/dir-list")["sessions"]) == 2
    assert json.loads((world.root / "bases.json").read_text())["labels"] == str(tmp_path / "lab")


def test_housekeeping_routes_list_and_refuse_outside_the_bases(server, world):
    _, url = server
    _pool(url)
    recordings = jget(url, "/api/housekeeping-recordings")
    assert {r["tag"] for r in recordings["recordings"]} >= {"rec_a"}
    assets = jget(url, "/api/housekeeping-assets?tag=rec_a")
    assert assets["tiles"] and assets["boxes"]
    status, body = post(url, "/api/housekeeping-delete", {"paths": [str(world.root / "sessions")]})
    assert status == 400 and "outside" in json.loads(body)["error"]
    archived = jpost(
        url, "/api/housekeeping-archive", {"paths": [str(world.tiles / "rec_a_k00000.npz")]}
    )
    assert len(archived["moved"]) == 1 and (world.tiles / "_archive" / "rec_a_k00000.npz").is_file()


# ---------------------------------------------------------------- train routes


def test_train_routes_report_unbound_state(server):
    _, url = server
    classes = jget(url, "/api/train-classes")
    assert set(classes) >= {"set", "weights", "classes"} and classes["set"] is None
    info = jget(url, "/api/train-info")
    assert info["set"] is None and "device" in info
    assert jget(url, "/api/train-frames") == {"frames": []}
    assert jget(url, "/api/saved-checkpoints") == {"weights": []}
    assert jget(url, "/api/sweep-status")["running"] is False
    assert "error" in jget(url, "/api/peak-distribution")
    assert jget(url, "/api/window-floor")["box"] == [64, 14]


def test_train_status_passes_through_whatever_trainstate_reports(server):
    """another change adds fields to TrainState.status(); this layer must not filter them"""
    app, url = server
    extra = {
        "running": False,
        "train_loss": 0.5,
        "val_loss": 0.6,
        "test_loss": 0.7,
        "checkpoint_count": 2,
        "final": True,
    }
    app.trainer.trainer.status = lambda: dict(extra)
    assert jget(url, "/api/train-status") == extra


def test_saved_checkpoints_entries_are_passed_through_untouched(server, monkeypatch):
    app, url = server
    entry = {"name": "a.pt", "training_set": None, "classes": [], "kb": 1, "final": True}
    monkeypatch.setattr(
        "smolsmort.review.train_api.saved_checkpoints", lambda root: {"weights": [entry]}
    )
    assert jget(url, "/api/saved-checkpoints") == {"weights": [entry]}


def test_two_unnamed_saves_in_one_second_do_not_overwrite_each_other(server, world, monkeypatch):
    app, url = server
    app.trainer.trainer.weights = review_world.WorldWeights(
        classes={"x": 0}, trained_on=1, box=(8, 8)
    )

    # the suggested name is stamped to the second; the clock is frozen so the two saves collide
    # by construction rather than by racing a second boundary (ci once landed either side of it)
    class Frozen:
        @staticmethod
        def now():
            from datetime import datetime

            return datetime(2026, 9, 21, 12, 0, 0)

    monkeypatch.setattr("smolsmort.review.train.datetime", Frozen)
    first = jpost(url, "/api/save-checkpoint", {})
    second = jpost(url, "/api/save-checkpoint", {})
    assert first["name"] != second["name"], (first, second)
    assert second["name"].endswith("-2.pt"), second
    listed = {w["name"] for w in jget(url, "/api/saved-checkpoints")["weights"]}
    assert {first["name"], second["name"]} <= listed, listed


def test_a_typed_checkpoint_name_that_is_taken_is_refused(server, world):
    app, url = server
    app.trainer.trainer.weights = review_world.WorldWeights(
        classes={"x": 0}, trained_on=1, box=(8, 8)
    )
    assert jpost(url, "/api/save-checkpoint", {"name": "r1"})["name"] == "r1.pt"
    again = jpost(url, "/api/save-checkpoint", {"name": "r1"})
    assert "already exists" in again["error"] and "error" in again


def test_a_finished_runs_own_save_numbers_a_taken_name_instead_of_refusing(server, world):
    app, _ = server
    app.trainer.trainer.weights = review_world.WorldWeights(
        classes={"x": 0}, trained_on=1, box=(8, 8)
    )
    assert app.trainer.save_checkpoint("r1")["name"] == "r1.pt"
    # nobody is at the dialog when a run saves itself, so refusing would write nothing at all
    assert app.trainer.save_checkpoint("r1", number_taken=True)["name"] == "r1-2.pt"


def test_a_saved_checkpoints_reply_names_it_the_way_the_listing_does(server, world):
    app, url = server
    (world.checkpoints / "runs").mkdir()
    app.trainer.trainer.weights = review_world.WorldWeights(
        classes={"x": 0}, trained_on=1, box=(8, 8)
    )
    saved = jpost(url, "/api/save-checkpoint", {"name": "r1", "folder": "runs"})
    assert saved["name"] == "runs/r1.pt", saved
    listed = [w["name"] for w in jget(url, "/api/saved-checkpoints")["weights"]]
    assert saved["name"] in listed, listed
    # the name a save replies with is one load-checkpoint accepts
    assert "error" not in jpost(url, "/api/load-checkpoint", {"name": saved["name"]})


def test_the_checkpoint_folder_picker_is_confined(server, world):
    _, url = server
    (world.checkpoints / "runs").mkdir()
    listing = jget(url, "/api/checkpoint-folders")
    assert listing["folders"] == ["runs"] and listing["here"] == ""
    escaped = jget(url, "/api/checkpoint-folders?under=../../..")
    assert escaped["here"] == ""
    assert post(url, "/api/save-checkpoint", {"folder": "../../evil"})[0] == 200


def test_binding_a_set_that_is_not_there_answers_json(server):
    _, url = server
    out = jpost(url, "/api/train-bind", {"name": "ghost"})
    assert "error" in out and out["set"] is None
    assert jpost(url, "/api/train-start", {})["error"].startswith("nothing bound")
    assert jpost(url, "/api/train-abort")["error"]
    assert jpost(url, "/api/sweep-start", {"recording": "rec_a"})["error"]
    assert jpost(url, "/api/sweep-send", {})["error"]
    assert "error" in jpost(url, "/api/load-checkpoint", {"name": "ghost.pt"})


# ---------------------------------------------------------------- the whole system


def _wait(url, path, done, seconds=20):
    deadline = time.time() + seconds
    while time.time() < deadline:
        status = jget(url, path)
        if done(status):
            return status
        time.sleep(0.05)
    raise AssertionError(f"{path} never finished: {status}")


def test_the_whole_loop_over_http_find_judge_promote_train_predict(server, world):
    """find -> judge (manual-label + save-labels) -> promote -> train (fake backend) -> predict, and
    the proposals re-enter judging - the smallest scenario that exercises the seams together."""
    _, url = server
    save_definition()

    # the set names its backend and keeps drawn sizes; find draws for it
    named = jpost(
        url, "/api/train-set-backend", {"name": "loop", "backend": "world", "size_mode": "native"}
    )
    assert named == {"name": "loop", "backend": "world", "size_mode": "native"}

    # find: bind a recording, draw boxes, save
    jpost(url, "/api/bind-recording", {"kind": "session", "name": "rec_a"})
    found = jpost(url, "/api/find-run", {"boxes": drawn_boxes(), "set": "loop"})
    assert found["tiles"] == 6 and found["size_mode"] == "native"

    # judge: four classes, and two discards (drawn boxes, so drawn negatives)
    names = [i["name"] for c in jget(url, "/api/clusters")["clusters"] for i in c["items"]]
    for i, name in enumerate(names[:4]):
        jpost(
            url,
            "/api/manual-label",
            {
                "name": name,
                "definition": "kinds",
                "picked": {"kind": "alpha" if i % 2 == 0 else "beta"},
            },
        )
    assert jpost(url, "/api/exclude", {"names": names[4:], "excluded": True})["done"] == names[4:]
    assert jget(url, "/api/labels-buffer")["pending"] == 4
    assert jpost(url, "/api/save-labels") == {"saved": 4}
    grid = jget(url, "/api/clusters")
    assert grid["counts"]["classes_assigned"] == 4
    assert all("not_object" not in i for c in grid["clusters"] for i in c["items"])

    # promote: one durable, named set; the two discards are negative rows, not classes
    promoted = jpost(url, "/api/promote-training", {"name": "loop"})
    assert promoted["rows"] == 4 and promoted["classes"] == {"alpha": 2, "beta": 2}
    assert promoted["negatives"] >= 2

    # train: bind, start with options, poll to the end
    bound = jpost(url, "/api/train-bind", {"name": "loop"})
    assert bound["classes"] == ["alpha", "beta"] and bound["set"] == "loop"
    info = jget(url, "/api/train-info")
    assert info["frames"] == 3 and info["objects"] == 4 and info["thinnest"]["count"] == 2
    assert info["backend"] == "world" and info["size_mode"] == "native"
    assert info["negatives"] >= 2
    assert jpost(
        url, "/api/train-start", {"epochs": 3, "learning_rate": 0.001, "seed": 7, "name": "run1"}
    ) == {"ok": True}
    finished = _wait(url, "/api/train-status", lambda s: s["finished"] or s["error"])
    assert finished["error"] is None and finished["epoch"] == 3
    assert review_world.WorldBackend.seen == {"epochs": 3, "learning_rate": 0.001, "seed": 7}

    # the named run is saved once it finishes, and is listed with its set
    saved = _wait(url, "/api/saved-checkpoints", lambda s: bool(s["weights"]))
    # the provenance lands before the weights, so the first listing already names the set
    assert (
        saved["weights"][0]["name"] == "run1.pt" and saved["weights"][0]["training_set"] == "loop"
    )

    # predict: sweep another recording, send the proposals above a threshold into the pool
    assert jpost(
        url, "/api/sweep-start", {"recording": "set/two", "percent": 100, "min_score": 0.5}
    ) == {"ok": True}
    sweep = _wait(url, "/api/sweep-status", lambda s: s["finished"] or s["error"])
    assert sweep["error"] is None and sweep["found"] == 3 and sweep["highest"] == 0.9
    assert sweep["out"].startswith("set__two.cnn-") and sweep["above"][50] == 3
    assert (world.labels / sweep["out"]).is_file()
    assert "proposals" not in sweep, "the whole list is held server-side, not polled"
    sent = jpost(url, "/api/sweep-send", {"min_score": 0.82})
    assert sent["sent"] == 2 and sent["tiles"] == 2 and sent["of"] == 3

    # the proposals re-enter judging: a new source in the pool, ready for the next round
    sources = {s["tag"]: s["tiles"] for s in jget(url, "/api/pool-sources")["sources"]}
    assert sources["rec_a"] == 6 and sources[sent["tag"]] == 2
    new_names = [
        i["name"]
        for c in jget(url, "/api/clusters")["clusters"]
        for i in c["items"]
        if i["source"] == sent["tag"]
    ]
    assert len(new_names) == 2
    jpost(
        url,
        "/api/manual-label",
        {"name": new_names[0], "definition": "kinds", "picked": {"kind": "beta"}},
    )
    jpost(url, "/api/save-labels")
    again = jpost(url, "/api/promote-training", {"name": "loop", "mode": "merge"})
    assert again["mode"] == "merge" and again["rows"] == 5
    assert {s["source"] for s in again["sources"]} >= {"rec_a", sent["tag"]}

    # a saved checkpoint loads back with its class map
    loaded = jpost(url, "/api/load-checkpoint", {"name": "run1.pt"})
    assert loaded["classes"] == ["alpha", "beta"] and loaded["weights"] == "run1.pt"


def test_binding_takes_a_set_path_inside_the_sets_folder_and_refuses_an_escape(server, world):
    _, url = server
    out = jpost(url, "/api/train-bind", {"name": "../outside"})
    assert "outside the sets folder" in out["error"]
    assert "error" in jpost(url, "/api/train-bind", {"name": "nested/ghost"})
    assert "error" in jpost(url, "/api/load-checkpoint", {"name": "../../x.pt"})


def test_a_set_without_a_backend_file_reports_heatmap_and_uniform(server, world):
    _, url = server
    save_definition()
    jpost(url, "/api/bind-recording", {"kind": "session", "name": "rec_a"})
    jpost(url, "/api/find-run", {"boxes": drawn_boxes()})
    for name in [i["name"] for c in jget(url, "/api/clusters")["clusters"] for i in c["items"]]:
        jpost(
            url,
            "/api/manual-label",
            {"name": name, "definition": "kinds", "picked": {"kind": "alpha"}},
        )
    jpost(url, "/api/save-labels")
    jpost(url, "/api/promote-training", {"name": "plain"})
    jpost(url, "/api/train-bind", {"name": "plain"})
    info = jget(url, "/api/train-info")
    assert (info["backend"], info["size_mode"]) == ("heatmap", "uniform")
    assert not list(world.sets.glob("*._backend.json"))
    # no file: the trainer keeps the backend it was started with
    assert jpost(url, "/api/train-start", {"epochs": 1}) == {"ok": True}
    _wait(url, "/api/train-status", lambda s: s["finished"] or s["error"])
    assert review_world.WorldBackend.seen["epochs"] == 1


def test_train_set_backend_persists_and_refuses_unknown_names(server, world):
    _, url = server
    saved = jpost(url, "/api/train-set-backend", {"name": "s1", "backend": "box"})
    assert saved == {"name": "s1", "backend": "box", "size_mode": "native"}
    assert json.loads((world.sets / "s1._backend.json").read_text()) == {
        "backend": "box",
        "size_mode": "native",
    }
    other = jpost(url, "/api/train-set-backend", {"name": "s2", "backend": "heatmap"})
    assert other["size_mode"] == "uniform"
    override = jpost(
        url, "/api/train-set-backend", {"name": "s3", "backend": "heatmap", "size_mode": "native"}
    )
    assert override["size_mode"] == "native"
    refused = jpost(url, "/api/train-set-backend", {"name": "s4", "backend": "nope"})
    assert (
        "nope" in refused["error"] and "heatmap" in refused["error"] and "box" in refused["error"]
    )
    assert not (world.sets / "s4._backend.json").exists()


def test_frame_modes_over_http(server, world):
    _, url = server
    assert jget(url, "/api/frame-modes?recording=rec_a") == {"frames": {}}
    reply = jpost(
        url, "/api/frame-mode", {"recording": "rec_a", "frame": "f01.png", "exhaustive": True}
    )
    assert reply == {"ok": True, "recording": "rec_a", "frame": "f01.png", "exhaustive": True}
    assert jget(url, "/api/frame-modes?recording=rec_a") == {
        "frames": {"f01.png": {"exhaustive": True}}
    }
    assert json.loads((world.labels / "rec_a._frames.json").read_text()) == {
        "f01.png": {"exhaustive": True}
    }
    jpost(url, "/api/frame-mode", {"recording": "rec_a", "frame": "f01.png", "exhaustive": False})
    assert jget(url, "/api/frame-modes?recording=rec_a") == {"frames": {}}
    assert get(url, "/api/frame-modes?recording=ghost")[0] == 404
    assert post(url, "/api/frame-mode", {"recording": "rec_a", "frame": "f01.png"})[0] == 400


def _bind_frames(app, world, name="cap", width=200):
    """bind a set whose one example is a real frame of this width, without promoting anything"""
    from smolsmort.detect.dataset import Example

    frame = world.sessions / "rec_a" / "frames" / "f00.png"
    review_world.write_frame(frame)
    (world.sets / f"{name}.jsonl").write_text("")
    example = Example(path=frame, centres=[(60.0, 50.0)], labels=["x"], sizes=[(30, 10)])
    app.trainer._examples = [example]
    app.trainer.trainer.bind([], {"x": 0}, training_set=name)


def test_train_set_backend_takes_and_validates_capture_size_over_http(server, world):
    _, url = server
    saved = jpost(
        url,
        "/api/train-set-backend",
        {"name": "c1", "backend": "heatmap", "capture_width": 1280, "downscale": 3},
    )
    assert saved == {
        "name": "c1",
        "backend": "heatmap",
        "size_mode": "uniform",
        "capture_width": 1280,
        "downscale": 3,
    }
    assert json.loads((world.sets / "c1._backend.json").read_text())["capture_width"] == 1280
    # left out keeps, null clears
    kept = jpost(url, "/api/train-set-backend", {"name": "c1", "backend": "heatmap"})
    assert kept["capture_width"] == 1280 and kept["downscale"] == 3
    cleared = jpost(
        url, "/api/train-set-backend", {"name": "c1", "backend": "heatmap", "capture_width": None}
    )
    assert "capture_width" not in cleared and cleared["downscale"] == 3
    for bad in (
        {"downscale": 9},
        {"downscale": "2"},
        {"capture_width": 10},
        {"capture_width": 1.5},
    ):
        refused = jpost(url, "/api/train-set-backend", {"name": "c2", "backend": "heatmap", **bad})
        assert "error" in refused
    assert not (world.sets / "c2._backend.json").exists()


def test_train_info_and_window_floor_follow_the_sets_capture_size(server, world):
    app, url = server
    _bind_frames(app, world)
    jpost(url, "/api/train-set-backend", {"name": "cap", "backend": "heatmap"})
    info = jget(url, "/api/train-info")
    assert (info["capture_width"], info["downscale"], info["input_width"]) == (200, 2, 100)
    assert "resampled_from" not in info
    # no override yet, so the width the picker shows is the frames' own
    assert info["capture_override"] is None and info["observed_width"] == 200
    floor_default = jget(url, "/api/window-floor")
    assert floor_default["downscale"] == 2
    jpost(
        url,
        "/api/train-set-backend",
        {"name": "cap", "backend": "heatmap", "capture_width": 100, "downscale": 1},
    )
    info = jget(url, "/api/train-info")
    assert (info["capture_width"], info["downscale"], info["input_width"]) == (100, 1, 100)
    assert info["capture_override"] == 100 and info["observed_width"] == 200
    floor = jget(url, "/api/window-floor")
    # the box halves with the capture width, and the window is cut at downscale 1
    assert floor["downscale"] == 1
    assert floor["box"] == [round(v / 2) for v in floor_default["box"]]
    assert floor["floor"] == floor_default["floor"]


def test_train_info_reports_a_capture_mismatch_with_the_loaded_model(server, world):
    app, url = server
    _bind_frames(app, world)
    app.trainer.trainer.weights = SimpleNamespace(capture_width=400, downscale=2)
    info = jget(url, "/api/train-info")
    assert info["resampled_from"] == 200 and info["capture_width"] == 200
    assert info["weights_capture_width"] == 400


def test_the_box_backend_reports_its_working_size_and_no_capture_settings(server, world):
    _, url = server
    jpost(url, "/api/train-set-backend", {"name": "bx", "backend": "box"})
    jpost(url, "/api/train-bind", {"name": "bx"})
    info = jget(url, "/api/train-info")
    assert info["working_size"] == 768
    assert "downscale" not in info and "capture_width" not in info


def test_state_changes_are_refused_while_a_sweep_reads_the_model(server, world):
    """the worker reads the backend, weights and classes per chunk; a bind, load, backend
    switch or training start under it would mix two models into one candidates file"""
    app, url = server

    class Weights:
        capture_width = 200

    class Slow:
        def predict(self, weights, frames, *, classes):
            time.sleep(0.4)
            return []

    app.trainer.trainer.weights = Weights()
    app.trainer.trainer._backend = Slow()
    assert jpost(url, "/api/sweep-start", {"recording": "rec_a", "min_score": 0.5}) == {"ok": True}
    for path, body in (
        ("/api/train-bind", {"name": "loop"}),
        ("/api/load-checkpoint", {"name": "x.pt"}),
        ("/api/train-set-backend", {"name": "loop", "backend": "heatmap"}),
        ("/api/train-start", {}),
    ):
        assert "sweep is running" in jpost(url, path, body)["error"], path
    _wait(url, "/api/sweep-status", lambda s: s["finished"] or s["error"])
    # the same set can be bound once the sweep is done
    assert "sweep is running" not in json.dumps(jpost(url, "/api/train-bind", {"name": "loop"}))


def test_bind_load_and_backend_switch_are_refused_while_a_run_is_going(server, world):
    """the run's end saves its weights under the bound set; a bind or load in between would file
    them under another name. the status dict is what start_sweep already consults, so it is
    flipped directly rather than racing a real run"""
    app, url = server
    trainer = app.trainer.trainer
    with trainer._job_lock:
        trainer.job["running"] = True
    try:
        for path, body in (
            ("/api/train-bind", {"name": "loop"}),
            ("/api/load-checkpoint", {"name": "x.pt"}),
            ("/api/train-set-backend", {"name": "loop", "backend": "heatmap"}),
        ):
            assert "training run is going" in jpost(url, path, body)["error"], path
    finally:
        with trainer._job_lock:
            trainer.job["running"] = False
    assert "run is going" not in json.dumps(jpost(url, "/api/train-bind", {"name": "loop"}))


def test_a_sweep_over_frames_of_another_width_warns(server, world):
    app, url = server

    class Weights:
        capture_width = 400

    class Fake:
        def predict(self, weights, frames, *, classes):
            return []

    app.trainer.trainer.weights = Weights()
    app.trainer.trainer._backend = Fake()
    assert jpost(url, "/api/sweep-start", {"recording": "rec_a", "min_score": 0.5}) == {"ok": True}
    status = _wait(url, "/api/sweep-status", lambda s: s["finished"] or s["error"])
    assert status["resampled_from"] == 200 and status["resampled_frames"] == 3
    assert "400" in status["warning"]
