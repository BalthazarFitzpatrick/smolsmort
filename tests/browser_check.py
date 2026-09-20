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
    page.evaluate(
        """() => window.smolsmortTabs.register({id: 'extra', label: 'extra',
             mount: el => { el.textContent = 'hello from a host tab'; }})"""
    )
    assert page.locator("#nav-bar .nav-tab").all_inner_texts()[-1] == "extra"
    show_tab(page, "extra")
    assert "hello from a host tab" in page.locator('.tab-panel[data-panel="extra"]').inner_text()


def check_train(page: Page, base: str, log: list) -> None:
    show_tab(page, "train")
    # the set picker lists what promotion wrote; binding it fills the summary
    page.click("#train-open")
    page.wait_for_selector(".menu-panel .menu-item")
    assert menu_labels(page) == ["world_set"], menu_labels(page)
    snap(page, "train-set-picker")
    page.click('.menu-item:has-text("world_set")')
    page.wait_for_function('document.getElementById("train-summary").innerText.includes("objects")')
    assert "2 objects" in page.inner_text("#train-summary")
    assert posts(log, "train-bind")[-1] == {"name": "world_set"}
    dismiss(page)
    page.click("#train-load")
    page.wait_for_selector(".menu-panel")
    assert "no saved weights yet" in page.locator(".menu-panel").inner_text()
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
    page.click("#train-load")
    page.wait_for_selector(".menu-panel .menu-item")
    assert menu_labels(page) == ["ck1.pt"], menu_labels(page)
    snap(page, "train-load-list")
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
    for label in ("save and close", "discard and close", "cancel"):
        assert label in text
    page.click("#ask-cancel")
    assert page.locator("#ask-dialog").count() == 0
    assert page.inner_text("#cluster-save") == "save (2)"
    assert posts(log, "save-labels") == []
    # discard drops the buffer and writes nothing
    close_set()
    page.click("#ask-discard-close")
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


def run(shots: Path | None = None) -> None:
    errors: list[str] = []
    log: list = []
    review_world.register_world_backend()
    with (
        tempfile.TemporaryDirectory() as tmp,
        pytest.MonkeyPatch.context() as patch,
    ):
        world = make_world(Path(tmp), patch, recordings=("rec_a",), frames=3)
        save_definition()
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

                page.on("request", record)
                page.goto(base + "/")
                page.wait_for_timeout(400)
                check_tabs_and_extension(page)
                for name, check in (
                    ("find", check_find),
                    ("select", check_select),
                    ("train", check_train),
                ):
                    check(page, base, log)
                    show_tab(page, name)
                    page.wait_for_timeout(300)
                    if shots:
                        page.screenshot(path=str(shots / f"{name}.png"))
                show_tab(page, "housekeeping")
                page.wait_for_timeout(300)
                if shots:
                    page.screenshot(path=str(shots / "housekeeping.png"))
                browser.close()
        finally:
            server.shutdown()
            server.server_close()
            review_world.unregister_world_backend()
    assert not errors, errors


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if out:
        out.mkdir(parents=True, exist_ok=True)
        SHOTS.append(out)
    run(out)
    print("browser checks passed")
