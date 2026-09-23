"""headless chromium against the REAL review server over a tiny synthetic recording: layout and
behaviour of the find, select and train tabs. run with:
uv run --with playwright python tests/browser_check.py [shots_dir]"""

from __future__ import annotations

import json
import sys
import tempfile
import threading
import time
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlparse

import pytest
import review_world
from playwright.sync_api import Page, sync_playwright
from review_world import drawn_boxes, make_world, save_definition

from smolsmort.review.server import build_app, serve
from smolsmort.review_ui.server import STATIC
from smolsmort.review_ui.tab import hyperparams_tab

SHOTS: list[Path] = []


def snap(page: Page, name: str) -> None:
    if SHOTS:
        page.screenshot(path=str(SHOTS[0] / f"{name}.png"))


def box(page: Page, selector: str) -> dict:
    return page.locator(selector).bounding_box()


def call(base: str, path: str, body: dict | None = None) -> dict:
    """one json call to the running server, used to seed state the way a user's earlier work would"""
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(
        base + path, data=data, method="GET" if data is None else "POST"
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        return json.loads(response.read())


def posts(log: list, name: str) -> list[dict]:
    return [body for method, path, body in log if method == "POST" and path.endswith(name)]


def show_tab(page: Page, name: str) -> None:
    page.click(f'.nav-tab[data-tab="{name}"]')
    page.wait_for_timeout(250)


def dismiss(page: Page) -> None:
    # a click on empty page closes any open menu
    page.mouse.click(640, 880)
    page.wait_for_timeout(100)


def menu_labels(page: Page) -> list[str]:
    return page.locator(".menu-panel .menu-item .name").all_inner_texts()


def check_tabs_and_extension(page: Page) -> None:
    tabs = page.locator("#nav-bar .nav-tab").all_inner_texts()
    assert tabs == ["find", "select", "train", "housekeeping"], tabs
    # a runtime tab with no topic lands in vision: it joins the same nav-bar, and the topic
    # switch (only vision has tabs so far) stays hidden
    page.evaluate(
        """() => window.smolsmortTabs.register({id: 'extra', label: 'extra',
             mount: el => { el.textContent = 'hello from a host tab'; }})"""
    )
    assert page.locator("#nav-bar .nav-tab").all_inner_texts()[-1] == "extra"
    assert "hidden" in page.get_attribute("#topic-switch", "class")
    show_tab(page, "extra")
    assert "hello from a host tab" in page.locator('.tab-panel[data-panel="extra"]').inner_text()

    # a second topic makes the switch appear; its tab lives on its own row, and switching back
    # restores vision's own last tab rather than resetting to the first one
    page.evaluate(
        """() => window.smolsmortTabs.register({id: 'forecast', label: 'forecast',
             topic: 'regression',
             mount: el => { el.textContent = 'hello from regression'; }})"""
    )
    page.wait_for_timeout(100)
    assert "hidden" not in page.get_attribute("#topic-switch", "class")
    assert page.locator("#topic-switch .topic-tab").all_inner_texts() == ["vision", "regression"]
    assert page.locator("#nav-bar .nav-tab").all_inner_texts()[-1] == "extra"

    page.click('#topic-switch .topic-tab:has-text("regression")')
    page.wait_for_timeout(100)
    assert page.locator("#nav-bar .nav-tab").all_inner_texts() == ["forecast"]
    assert "hello from regression" in page.locator('.tab-panel[data-panel="forecast"]').inner_text()

    page.click('#topic-switch .topic-tab:has-text("vision")')
    page.wait_for_timeout(100)
    assert page.locator("#nav-bar .nav-tab").all_inner_texts()[-1] == "extra"


def check_train(page: Page, base: str, log: list, urls: list, app) -> None:
    show_tab(page, "train")
    # both loaders are directory browsers rooted at their own base: they drill in and out via '..'
    for head, root, folder, inner in [
        ("#train-open", "tiles", "a/", "/a"),
        ("#train-load", "checkpoints", "runs/", "/runs"),
    ]:
        page.click(head)
        page.wait_for_selector(".menu-panel .menu-heading, .menu-panel .field-label")
        assert ".." not in menu_labels(page)
        page.click(f'.menu-item:has-text("{folder}")')
        page.wait_for_function(
            f'document.querySelector(".menu-panel").innerText.includes("{inner}")'
        )
        assert ".." in menu_labels(page)
        snap(page, f"train-dir-{root}")
        page.click('.menu-item:has-text("..")')
        page.wait_for_function(
            f'!document.querySelector(".menu-panel").innerText.includes("{inner}")'
        )
        assert any(f"root={root}" in u for u in urls)
        dismiss(page)

    # picking a set sends its name and fills the summary
    page.click("#train-open")
    page.wait_for_selector(".menu-panel .menu-item")
    assert "world_set" in menu_labels(page), menu_labels(page)
    snap(page, "train-set-picker")
    page.click('.menu-item:has-text("world_set")')
    page.wait_for_function('document.getElementById("train-summary").innerText.includes("objects")')
    assert "2 objects" in page.inner_text("#train-summary")
    assert posts(log, "train-bind")[-1] == {"name": "world_set"}

    # the backend is picked from the topic's vision flavour dropdown, not this tab's own row -
    # that row hides once it wires into the dropdown
    assert "hidden" in page.get_attribute("#train-backend-row", "class")
    flavours = page.eval_on_selector_all("#topic-flavour option", "els => els.map(e => e.value)")
    assert "box" in flavours and "xgboost" not in flavours, flavours
    other = next(n for n in flavours if n != page.input_value("#topic-flavour"))
    page.select_option("#topic-flavour", other)
    # the select updates at once; the backend switch itself is async, so wait for its effect
    page.wait_for_function(
        f'document.getElementById("train-summary").innerText.includes("{other}")'
    )
    sent = posts(log, "train-set-backend")[-1]
    assert sent == {"name": "world_set", "backend": other}, sent
    mode = call(base, "/api/train-info")["size_mode"]
    assert f"sizes {mode}" in page.inner_text("#train-summary")
    assert call(base, "/api/train-info")["backend"] == other
    snap(page, "train-backend")

    # the capture picker: width and downscale are stored on the set, the head says what the net sees
    call(base, "/api/train-set-backend", {"name": "world_set", "backend": "heatmap"})
    show_tab(page, "find")
    show_tab(page, "train")
    page.wait_for_function('document.getElementById("train-capture").innerText.includes("px")')
    head = page.inner_text("#train-capture").strip()
    assert "/" in head and head.endswith("px"), head
    page.click("#train-capture")
    page.wait_for_selector('.menu-panel .menu-item:has-text("/ 4")')
    assert "downscale" in page.locator(".menu-panel").inner_text()
    page.fill(".menu-panel input", "160")
    page.click('.menu-panel .menu-item:has-text("/ 4")')
    page.click('.menu-panel .toggle:has-text("apply")')
    page.wait_for_function(
        'document.getElementById("train-capture").innerText.trim() == "160 / 4 = 40px"'
    )
    sent = posts(log, "train-set-backend")[-1]
    assert sent["capture_width"] == 160 and sent["downscale"] == 4, sent
    stored = call(base, "/api/train-info")
    assert (stored["capture_width"], stored["downscale"]) == (160, 4), stored
    assert "capture set here" in page.inner_text("#train-summary")
    snap(page, "train-capture")
    # an empty width follows the frames again
    page.click("#train-capture")
    page.wait_for_selector(".menu-panel input")
    page.fill(".menu-panel input", "")
    page.click('.menu-panel .menu-item:has-text("/ 2")')
    page.click('.menu-panel .toggle:has-text("apply")')
    page.wait_for_function(
        'document.getElementById("train-summary").innerText.includes("from the frames")'
    )
    assert call(base, "/api/train-info").get("capture_override") is None
    dismiss(page)
    page.click("#train-load")
    page.wait_for_selector(".menu-panel")
    page.wait_for_selector('.menu-item:has-text("runs/")')
    assert "ck1.pt" not in menu_labels(page), menu_labels(page)
    dismiss(page)
    # the side-by-side loaders share a row
    tiles, weights = box(page, "#train-open"), box(page, "#train-load")
    assert abs(tiles["y"] - weights["y"]) < 2 and tiles["x"] < weights["x"]

    # one stepper per row, label left, value and buttons right-aligned
    rows = page.locator("[data-stepper]")
    assert rows.count() == 3
    tops = []
    for i in range(3):
        row = rows.nth(i)
        assert row.locator("[data-step]").count() == 2
        rb = row.bounding_box()
        label, plus = (
            row.locator(".field-label").bounding_box(),
            row.locator("[data-step]:last-child").bounding_box(),
        )
        assert label["x"] < rb["x"] + 4
        assert abs((plus["x"] + plus["width"]) - (rb["x"] + rb["width"])) < 2
        tops.append(round(rb["y"]))
    assert len(set(tops)) == 3 and tops == sorted(tops)
    page.click('[data-stepper="epochs"] [data-step="50"]')
    assert page.inner_text("#train-epochs") == "250"

    # config / train / abort / save weights: an equal 2x2 grid
    cells = [
        box(page, f"#{i}") for i in ("train-config", "train-start", "train-abort", "train-save")
    ]
    assert (
        len({round(c["width"]) for c in cells}) == 1
        and len({round(c["height"]) for c in cells}) == 1
    )
    assert len({round(c["x"]) for c in cells}) == 2 and len({round(c["y"]) for c in cells}) == 2

    # the config menu is the old hyperparams page, served by the real server
    page.click("#train-config")
    page.wait_for_selector(".menu-panel .menu-item")
    assert "adamw" in page.locator(".menu-panel").inner_text()
    snap(page, "train-config-menu")
    dismiss(page)

    # save weights waits for weights to exist
    assert "disabled" in page.get_attribute("#train-save", "class")
    page.click("#train-start")
    page.wait_for_function(
        'document.getElementById("train-progress-text").innerText.startsWith("done")',
        timeout=20000,
    )
    sent = posts(log, "train-start")[-1]
    assert sent["epochs"] == 250 and sent["batch"] == 8, sent
    assert "disabled" not in page.get_attribute("#train-save", "class")
    assert "disabled" in page.get_attribute("#train-abort", "class")

    # saving names a snapshot of the current weights, then the load list offers it
    page.click("#train-save")
    page.wait_for_selector(".menu-panel input")
    page.fill(".menu-panel input", "ck1")
    page.click('.menu-panel .toggle:has-text("save")')
    page.wait_for_function(
        'document.getElementById("train-progress-text").innerText.includes("saved")'
    )
    assert posts(log, "save-checkpoint")[-1] == {"name": "ck1"}
    call(base, "/api/save-checkpoint", {"name": "r1", "folder": "runs"})
    page.click("#train-load")
    page.wait_for_selector(".menu-panel .menu-item")
    assert "ck1.pt" in menu_labels(page), menu_labels(page)
    page.click('.menu-item:has-text("runs/")')
    page.wait_for_selector('.menu-item:has-text("r1.pt")')
    snap(page, "train-load-list")
    page.click('.menu-item:has-text("r1.pt")')
    page.wait_for_function(
        'document.getElementById("train-progress-text").innerText.includes("loaded")'
    )
    assert posts(log, "load-checkpoint")[-1] == {"name": "runs/r1.pt"}
    dismiss(page)
    snap(page, "train-after-run")

    # sweep the recording and send the proposals to select
    page.click("#sweep-open")
    page.click('.menu-item:has-text("rec_a")')
    page.click("#sweep-start")
    page.wait_for_function(
        'document.getElementById("sweep-progress-text").innerText.includes("proposals")',
        timeout=20000,
    )
    snap(page, "train-sweep")
    page.click("#sweep-send")
    page.wait_for_timeout(500)
    assert page.locator('.tab-panel[data-panel="select"]:not(.hidden)').count() == 1

    # frames of another width than the weights were trained at are resampled, and the sweep says so
    class Weights:
        capture_width = 400

    class Fake:
        def predict(self, weights, frames, *, classes):
            return []

    app.trainer.trainer.weights = Weights()
    app.trainer.trainer._backend = Fake()
    show_tab(page, "train")
    assert "hidden" in page.get_attribute("#sweep-warning", "class")
    page.click("#sweep-start")
    page.wait_for_selector("#sweep-warning:not(.hidden)", timeout=20000)
    assert "400" in page.inner_text("#sweep-warning")
    # the whole sentence is readable, not clipped to one ellipsed line
    warn = box(page, "#sweep-warning")
    assert warn["height"] > 20, warn
    # loading weights unbinds the set, so the summary has no capture facts to show here - the
    # sweep line is what reports the mismatch on this path
    snap(page, "train-sweep-warning")


def check_find(page: Page, base: str, log: list) -> None:
    show_tab(page, "find")
    assert "range-slider" in page.get_attribute("#draw-slider", "class")
    assert page.get_attribute("#draw-slider", "type") == "range"
    # width sits above height, both are range sliders
    w, h = box(page, "#crop-w"), box(page, "#crop-h")
    assert w["y"] < h["y"]
    assert page.inner_text("#crop-mode") == "percent" and page.inner_text("#crop-aspect") == "free"
    page.click("#crop-mode")
    page.wait_for_timeout(400)
    assert page.inner_text("#crop-mode") == "absolute"
    assert posts(log, "crop-settings")[-1]["crop_mode"] == "absolute"
    assert page.inner_text("#crop-w-value").endswith("px")

    def slide(selector: str, value: int) -> None:
        page.eval_on_selector(
            selector,
            """(el, v) => { el.value = v; el.dispatchEvent(new Event('input', {bubbles: true})); }""",
            value,
        )

    # free aspect: sliders are independent
    slide("#crop-w", 100)
    slide("#crop-h", 50)
    assert page.input_value("#crop-h") == "50"
    # fixed aspect captures the ratio now (2:1), and the partner follows
    page.click("#crop-aspect")
    assert page.inner_text("#crop-aspect") == "fixed"
    slide("#crop-w", 80)
    assert page.input_value("#crop-h") == "40"
    slide("#crop-h", 30)
    assert page.input_value("#crop-w") == "60"
    page.wait_for_timeout(400)
    last = posts(log, "crop-settings")[-1]
    assert last["aspect"] == "fixed" and last["crop_w"] == 60 and last["crop_h"] == 30, last
    # a slider past the partner's end stops at the ratio, not off it
    slide("#crop-w", 200)
    assert page.input_value("#crop-h") == "100"
    page.wait_for_timeout(400)
    # the server kept the rule the sliders set
    assert call(base, "/api/crop-settings")["aspect"] == "fixed"

    # the picker binds the recording, its frame loads and its drawn boxes come back
    page.click("#draw-open")
    page.wait_for_selector(".menu-panel .menu-item")
    assert "rec_a" in menu_labels(page)
    page.click('.menu-panel .menu-item:has-text("rec_a")')
    page.click('.menu-panel .toggle:has-text("open")')
    page.wait_for_function('document.getElementById("draw-where").innerText.includes("rec_a")')
    page.wait_for_function("document.getElementById('draw-canvas').width > 0")
    assert posts(log, "bind-recording")[-1] == {"kind": "session", "name": "rec_a"}
    snap(page, "find-bound")
    dismiss(page)
    page.click("#crop-aspect")

    # per-frame mode: explicit by default, the toggle flips it, the badge shows, reload keeps it
    assert page.inner_text("#frame-mode") == "explicit"
    assert "hidden" in page.get_attribute("#frame-badge", "class")
    page.click("#frame-mode")
    page.wait_for_timeout(300)
    frame = call(base, "/api/draw-frames?percent=20")["frames"][0].split("/")[-1]
    sent = posts(log, "frame-mode")[-1]
    assert sent["exhaustive"] is True and sent["frame"].split("/")[-1] == frame, sent
    assert page.inner_text("#frame-mode") == "exhaustive"
    assert "hidden" not in page.get_attribute("#frame-badge", "class")
    snap(page, "find-exhaustive")
    page.reload()
    page.wait_for_timeout(500)
    show_tab(page, "find")
    page.wait_for_function("document.getElementById('draw-canvas').width > 0")
    page.wait_for_function('document.getElementById("frame-mode").innerText == "exhaustive"')
    modes = call(base, f"/api/frame-modes?recording={sent['recording']}")["frames"]
    assert modes[frame] == {"exhaustive": True}, modes
    page.click("#frame-mode")
    page.wait_for_timeout(300)
    assert page.inner_text("#frame-mode") == "explicit"


def check_select(page: Page, base: str, log: list) -> None:
    show_tab(page, "select")
    ids = page.eval_on_selector_all("#clusters-bar > *", "els => els.map(e => e.id || e.className)")
    assert ids == [
        "cluster-open-set",
        "cluster-classdef",
        "cluster-select-all",
        "cluster-clear-sel",
        "recluster",
        "boxes-loaded",
        "classes-assigned",
        "spacer",
        "cluster-save",
        "cluster-promote",
    ], ids
    bar, promote, save = (
        box(page, "#clusters-bar"),
        box(page, "#cluster-promote"),
        box(page, "#cluster-save"),
    )
    assert abs((promote["x"] + promote["width"]) - (bar["x"] + bar["width"])) < 2
    assert save["x"] + save["width"] < promote["x"]
    assert page.inner_text("#boxes-loaded") == "6 boxes loaded"
    assert page.inner_text("#classes-assigned") == "2 classes assigned"
    assert "disabled" in page.get_attribute("#cluster-save", "class")
    assert page.locator(".cluster-item img").first.evaluate("i => i.complete && i.naturalWidth > 0")

    # no library concept and no smart-center control anywhere on the page
    page_text = page.inner_text("body").lower()
    assert "library" not in page_text and "smart" not in page_text
    assert page.locator("#cluster-smart-center").count() == 0

    def pick_set() -> None:
        page.click("#cluster-open-set")
        page.wait_for_selector('.menu-panel .menu-item:has-text("rec_a")')
        page.click('.menu-panel .menu-item:has-text("rec_a")')

    def close_set() -> None:
        pick_set()
        page.click(".menu-panel .toggle:has-text('close')")
        page.wait_for_timeout(300)

    def reopen_set() -> None:
        pick_set()
        page.click(".menu-panel .toggle:has-text('open')")
        page.wait_for_timeout(300)
        dismiss(page)

    # closing a set with nothing pending closes without a dialog, and the tiles leave the grid
    close_set()
    assert page.locator("#ask-dialog").count() == 0
    assert posts(log, "close-source")[-1] == {"tag": "rec_a"}
    assert page.locator(".cluster-item").count() == 0
    snap(page, "select-closed")
    reopen_set()
    assert page.locator(".cluster-item").count() == 6

    # buffering: nothing is written, the save button counts and enables
    names = page.eval_on_selector_all(".cluster-item", "els => els.map(e => e.dataset.name)")
    page.click(".cluster-item >> nth=0")
    page.click(".cluster-item >> nth=1", modifiers=["Meta"])
    picked = [page.get_attribute(f".cluster-item >> nth={i}", "data-name") for i in (0, 1)]
    page.click(".cluster-item >> nth=1", button="right")
    page.click('.menu-panel .menu-item:has-text("alpha")')
    snap(page, "select-class-picker")
    page.click('.menu-panel .toggle:has-text("apply")')
    page.wait_for_timeout(300)
    assert page.inner_text("#cluster-save") == "save (2)"
    assert "disabled" not in page.get_attribute("#cluster-save", "class")
    assert posts(log, "save-labels") == []
    assert sorted(posts(log, "manual-label")[-1]["names"]) == sorted(picked)
    assert set(picked) <= set(names)

    # unsaved work: the dialog names the count; cancel leaves everything
    close_set()
    snap(page, "select-unsaved-dialog")
    text = page.inner_text("#ask-dialog")
    assert "you have 2 unsaved class assignments" in text
    for label in ("save and close", "close without saving", "cancel"):
        assert label in text
    page.click("#ask-cancel")
    assert page.locator("#ask-dialog").count() == 0
    assert page.inner_text("#cluster-save") == "save (2)"
    assert posts(log, "save-labels") == []
    # close without saving drops the buffer and writes nothing
    close_set()
    page.click("#ask-close-unsaved")
    page.wait_for_timeout(300)
    assert posts(log, "close-source")[-1] == {"tag": "rec_a", "discard": True}
    assert posts(log, "save-labels") == []
    assert page.inner_text("#cluster-save") == "save"
    reopen_set()
    # save writes the buffer
    page.click(".cluster-item >> nth=0")
    page.click(".cluster-item >> nth=0", button="right")
    page.click('.menu-panel .menu-item:has-text("beta")')
    page.click('.menu-panel .toggle:has-text("apply")')
    page.wait_for_timeout(300)
    page.click("#cluster-save")
    page.wait_for_timeout(300)
    assert len(posts(log, "save-labels")) == 1
    assert page.inner_text("#cluster-save") == "save"
    assert page.inner_text("#classes-assigned") == "1 classes assigned"

    # "not a class" is declared at once and needs no save
    page.click(".cluster-item >> nth=3")
    page.click(".cluster-item >> nth=3", button="right")
    page.click('.menu-panel .toggle:has-text("not a class")')
    page.wait_for_timeout(300)
    assert posts(log, "exclude")[-1]["excluded"] is True
    assert page.locator(".cluster-item.excluded").count() == 1
    assert page.inner_text("#cluster-save") == "save"
    snap(page, "select-after-judging")

    # the x on every tile: marks and unmarks "not a class", dims the tile, sinks it to the end
    assert page.locator(".tile-x").count() == 6
    assert page.get_attribute(".tile-x >> nth=0", "title") == "not a class"
    assert page.get_attribute(".tile-x >> nth=0", "aria-label") == "not a class"
    cell, mark = box(page, ".cluster-item >> nth=0"), box(page, ".tile-x >> nth=0")
    assert mark["x"] + mark["width"] <= cell["x"] + cell["width"] + 1
    assert mark["y"] + mark["height"] <= cell["y"] + cell["height"] + 1
    assert mark["x"] > cell["x"] + cell["width"] / 2 and mark["y"] > cell["y"] + cell["height"] / 2
    before = len(posts(log, "exclude"))
    target = page.get_attribute(".cluster-item:not(.excluded) >> nth=0", "data-name")
    page.click(f'.cluster-item[data-name="{target}"] .tile-x')
    page.wait_for_timeout(400)
    assert posts(log, "exclude")[before:] == [{"name": target, "excluded": True}]
    assert page.locator(".cluster-item.excluded").count() == 2
    dimmed = page.eval_on_selector(
        ".cluster-item.excluded .tile-viewport", "e => Number(getComputedStyle(e).opacity)"
    )
    assert 0.3 < dimmed < 0.6, dimmed
    flags = page.eval_on_selector_all(
        ".cluster-item", "els => els.map(e => e.classList.contains('excluded'))"
    )
    assert flags == sorted(flags), flags
    assert flags[-2:] == [True, True]
    assert page.locator(".filter-last, #filter-judged .toggle").last.inner_text() == "not a class"
    snap(page, "select-x-marked")
    page.click(f'.cluster-item[data-name="{target}"] .tile-x')
    page.wait_for_timeout(400)
    assert posts(log, "exclude")[-1] == {"name": target, "excluded": False}
    assert page.locator(".cluster-item.excluded").count() == 1
    assert "discard" not in page.inner_text("body").lower()


def seed(base: str) -> None:
    """the state earlier work leaves behind, made through the real routes: a bound recording,
    drawn boxes cut into tiles, two tiles classed and saved, and a promoted training set"""
    call(base, "/api/bind-recording", {"kind": "session", "name": "rec_a"})
    saved = call(base, "/api/find-run", {"boxes": drawn_boxes()})
    assert saved["tiles"] == 6, saved
    names = [t["name"] for c in call(base, "/api/clusters")["clusters"] for t in c["items"]]
    call(
        base,
        "/api/manual-label",
        {"names": names[:1], "definition": "kinds", "picked": {"kind": "alpha"}},
    )
    call(
        base,
        "/api/manual-label",
        {"names": names[1:2], "definition": "kinds", "picked": {"kind": "beta"}},
    )
    assert call(base, "/api/save-labels", {})["saved"] == 2
    promoted = call(base, "/api/promote-training", {"name": "world_set"})
    assert promoted.get("rows") == 2 or promoted.get("name") == "world_set", promoted
    call(base, "/api/unbind-recording", {})


class Session:
    """one live server, one page, and the request log the checks assert against"""

    def __init__(self, page: Page, base: str, app, log: list, urls: list[str]):
        self.page, self.base, self.app, self.log, self.urls = page, base, app, log, urls


@contextmanager
def session() -> Iterator[Session]:
    """the real review server over a seeded synthetic recording, and a headless page on it.
    page errors are collected while it runs and raised on exit, so a check that passed on a
    page that threw does not count"""
    errors: list[str] = []
    log: list = []
    urls: list[str] = []
    review_world.register_world_backend()
    with (
        tempfile.TemporaryDirectory() as tmp,
        pytest.MonkeyPatch.context() as patch,
    ):
        world = make_world(Path(tmp), patch, recordings=("rec_a",), frames=3)
        save_definition()
        (world.tiles / "a").mkdir()
        (world.checkpoints / "runs").mkdir()
        app = build_app(
            backend="world",
            ui_dir=STATIC,
            tabs=[hyperparams_tab()],
            pool=world.tiles,
            bases_file=world.root / "bases.json",
            crop_file=world.root / "crop.json",
        )
        server = serve(app)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        base = f"http://127.0.0.1:{server.server_port}"
        try:
            time.sleep(0.2)
            seed(base)
            with sync_playwright() as pw:
                browser = pw.chromium.launch()
                page = browser.new_page(viewport={"width": 1280, "height": 900})
                page.on("pageerror", lambda exc: errors.append(str(exc)))

                def record(request) -> None:
                    if request.method == "POST":
                        log.append(
                            (
                                "POST",
                                urlparse(request.url).path,
                                json.loads(request.post_data or "{}"),
                            )
                        )

                page.on("request", lambda r: urls.append(r.url))
                page.on("request", record)
                page.goto(base + "/")
                page.wait_for_timeout(400)
                yield Session(page, base, app, log, urls)
                browser.close()
        finally:
            server.shutdown()
            server.server_close()
            review_world.unregister_world_backend()
    assert not errors, errors


# THE ORDER IS THE LOOP: find draws the boxes select judges and train learns from, so each check
# builds on the one before it. a runner splits them into named steps but keeps this order
CHECKS = (
    ("find", lambda s: check_find(s.page, s.base, s.log)),
    ("select", lambda s: check_select(s.page, s.base, s.log)),
    ("train", lambda s: check_train(s.page, s.base, s.log, s.urls, s.app)),
)


def run(shots: Path | None = None) -> None:
    """every check in order on one session, with a screenshot of each tab when `shots` is given"""
    with session() as live:
        check_tabs_and_extension(live.page)
        for name, check in CHECKS:
            check(live)
            show_tab(live.page, name)
            live.page.wait_for_timeout(300)
            if shots:
                live.page.screenshot(path=str(shots / f"{name}.png"))
        show_tab(live.page, "housekeeping")
        live.page.wait_for_timeout(300)
        if shots:
            live.page.screenshot(path=str(shots / "housekeeping.png"))


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if out:
        out.mkdir(parents=True, exist_ok=True)
        SHOTS.append(out)
    run(out)
    print("browser checks passed")
