"""headless chromium against the stub server: layout and behaviour of the find, select and train
tabs. run with: uv run --with playwright python tests/browser_check.py [shots_dir]"""

from __future__ import annotations

import sys
from pathlib import Path

from playwright.sync_api import Page, sync_playwright
from ui_stub import start_stub

SHOTS: list[Path] = []


def snap(page: Page, name: str) -> None:
    if SHOTS:
        page.screenshot(path=str(SHOTS[0] / f"{name}.png"))


def box(page: Page, selector: str) -> dict:
    return page.locator(selector).bounding_box()


def set_server(page: Page, base: str, payload: dict) -> None:
    page.evaluate(
        """([url, body]) => fetch(url, {method: 'POST', body: JSON.stringify(body)})""",
        [f"{base}/stub/set", payload],
    )


def posts(state: dict, name: str) -> list[dict]:
    return [body for method, path, body in state["log"] if method == "POST" and path.endswith(name)]


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


def check_train(page: Page, base: str, state: dict) -> None:
    show_tab(page, "train")
    # dropdowns drill in and out via '..' - each rooted at its own base
    for head, root, top, folder, inner in [
        ("#train-open", "tiles", "/t", "a/", "/t/a"),
        ("#train-load", "checkpoints", "/c", "runs/", "/c/runs"),
    ]:
        page.click(head)
        page.wait_for_selector(".menu-panel .menu-heading, .menu-panel .field-label")
        assert page.locator(".menu-panel").inner_text().count(top) >= 1
        page.click(f'.menu-item:has-text("{folder}")')
        snap(page, f"train-dir-{root}")
        page.wait_for_function(
            f'document.querySelector(".menu-panel").innerText.includes("{inner}")'
        )
        assert ".." in menu_labels(page)
        page.click('.menu-item:has-text("..")')
        page.wait_for_function(
            f'!document.querySelector(".menu-panel").innerText.includes("{inner}")'
        )
        assert any(f"root={root}" in path for method, path, _ in state["log"] if method == "GET")
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

    # save weights waits for a first checkpoint
    assert "disabled" in page.get_attribute("#train-save", "class")
    set_server(page, base, {"train": {"checkpoint_count": 1}})
    show_tab(page, "find")
    show_tab(page, "train")
    assert "disabled" not in page.get_attribute("#train-save", "class")

    # three loss boxes, and a running job is not stopped by opening the save list
    set_server(
        page,
        base,
        {
            "train": {
                "running": True,
                "epoch": 3,
                "epochs": 10,
                "checkpoint_count": 2,
                "train_loss": 0.5,
                "val_loss": 0.6,
                "test_loss": None,
                "history": [1, 0.8, 0.5],
            }
        },
    )
    show_tab(page, "find")
    show_tab(page, "train")
    assert page.inner_text("#loss-train") == "0.500"
    assert page.inner_text("#loss-val") == "0.600"
    assert page.inner_text("#loss-test") == "-"
    before = len(posts(state, "train-abort"))
    page.click("#train-save")
    page.wait_for_selector(".menu-panel .menu-item")
    snap(page, "train-running-save-list")
    assert menu_labels(page)[0] == "ck-final", menu_labels(page)
    assert len(posts(state, "train-abort")) == before
    assert "disabled" not in page.get_attribute("#train-abort", "class")
    dismiss(page)
    set_server(page, base, {"train": {"running": False}})
    show_tab(page, "find")
    show_tab(page, "train")


def check_find(page: Page, base: str, state: dict) -> None:
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
    assert posts(state, "crop-settings")[-1]["crop_mode"] == "absolute"
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
    last = posts(state, "crop-settings")[-1]
    assert last["aspect"] == "fixed" and last["crop_w"] == 60 and last["crop_h"] == 30, last
    # a slider past the partner's end stops at the ratio, not off it
    slide("#crop-w", 200)
    assert page.input_value("#crop-h") == "100"


def check_select(page: Page, base: str, state: dict) -> None:
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
    assert page.inner_text("#classes-assigned") == "1 classes assigned"
    assert "disabled" in page.get_attribute("#cluster-save", "class")

    # no library concept and no smart-center control anywhere on the page
    page_text = page.inner_text("body").lower()
    assert "library" not in page_text and "smart" not in page_text
    assert page.locator("#cluster-smart-center").count() == 0

    # closing a set with nothing pending closes without a dialog
    def close_set() -> None:
        page.click("#cluster-open-set")
        page.click('.menu-panel .menu-item:has-text("set1")')
        page.click("#close-set, .menu-panel .toggle:has-text('close')")
        page.wait_for_timeout(300)

    close_set()
    assert page.locator("#ask-dialog").count() == 0
    assert posts(state, "close-source")[-1] == {"tag": "set1"}

    # buffering: nothing is written, the save button counts and enables
    page.click(".cluster-item >> nth=0")
    page.click(".cluster-item >> nth=1", modifiers=["Meta"])
    page.click(".cluster-item >> nth=1", button="right")
    page.click('.menu-panel .menu-item:has-text("alpha")')
    snap(page, "select-class-picker")
    page.click('.menu-panel .toggle:has-text("apply")')
    page.wait_for_timeout(300)
    assert page.inner_text("#cluster-save") == "save (2)"
    assert "disabled" not in page.get_attribute("#cluster-save", "class")
    assert posts(state, "save-labels") == []
    assert posts(state, "manual-label")[-1]["names"] == ["tile0.png", "tile1.png"]

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
    assert posts(state, "save-labels") == []
    # discard drops the buffer and writes nothing
    close_set()
    page.click("#ask-discard-close")
    page.wait_for_timeout(300)
    assert posts(state, "close-source")[-1] == {"tag": "set1", "discard": True}
    assert posts(state, "save-labels") == []
    assert page.inner_text("#cluster-save") == "save"
    # save writes the buffer
    page.click(".cluster-item >> nth=0")
    page.click(".cluster-item >> nth=0", button="right")
    page.click('.menu-panel .menu-item:has-text("beta")')
    page.click('.menu-panel .toggle:has-text("apply")')
    page.wait_for_timeout(300)
    page.click("#cluster-save")
    page.wait_for_timeout(300)
    assert len(posts(state, "save-labels")) == 1
    assert page.inner_text("#cluster-save") == "save"


def run(shots: Path | None = None) -> None:
    server, state = start_stub()
    base = f"http://127.0.0.1:{server.server_port}"
    errors: list[str] = []
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            page = browser.new_page(viewport={"width": 1280, "height": 900})
            page.on("pageerror", lambda exc: errors.append(str(exc)))
            page.goto(base + "/")
            page.wait_for_timeout(400)
            check_tabs_and_extension(page)
            for name, check in (
                ("find", check_find),
                ("select", check_select),
                ("train", check_train),
            ):
                check(page, base, state)
                show_tab(page, name)
                page.wait_for_timeout(300)
                if shots:
                    page.screenshot(path=str(shots / f"{name}.png"))
            browser.close()
    finally:
        server.shutdown()
    assert not errors, errors


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if out:
        out.mkdir(parents=True, exist_ok=True)
        SHOTS.append(out)
    run(out)
    print("browser checks passed")
