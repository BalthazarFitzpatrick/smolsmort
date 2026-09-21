"""retake the README's screenshots against a synthetic world: frames from smolsmort.boxes.synthetic
(mottled backgrounds, elongated objects of three classes) drawn on, judged, promoted, trained on the
suite's fake backend and swept, then one shot per tab. run with:

    uv run --with playwright python tools/shoot_docs_images.py docs/images

every image it writes is invented data; nothing here reads a real recording
"""

from __future__ import annotations

import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from playwright.sync_api import sync_playwright

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
import review_world  # noqa: E402
from browser_check import call, dismiss, show_tab  # noqa: E402

from smolsmort.boxes import synthetic  # noqa: E402
from smolsmort.review.server import build_app, serve  # noqa: E402
from smolsmort.review_ui.server import STATIC  # noqa: E402
from smolsmort.review_ui.tab import hyperparams_tab  # noqa: E402

FRAMES = 4
SIZE = (1280, 720)
KINDS = ("ring", "bar", "blob")


def write_frames(frames_dir: Path) -> list[dict]:
    """synthetic frames with known boxes; returns one candidate dict per object"""
    rng = np.random.default_rng(7)
    backgrounds = {size: synthetic.background(*size, seed=3) for size in synthetic.FRAME_SIZES}
    boxes = []
    for i in range(FRAMES):
        # make_frame picks a size itself; a heatmap set wants one resolution, so keep one
        frame = synthetic.make_frame(rng, backgrounds, overlap_rate=0.0)
        while frame["image"].shape[1] != SIZE[0]:
            frame = synthetic.make_frame(rng, backgrounds, overlap_rate=0.0)
        name = f"f{i:02d}.png"
        Image.fromarray(frame["image"]).save(frames_dir / name)
        for (x0, y0, x1, y1), cls in zip(frame["boxes"], frame["classes"], strict=True):
            boxes.append(
                {
                    "path": name,
                    "left": int(x0),
                    "top": int(y0),
                    "width": int(x1 - x0),
                    "height": int(y1 - y0),
                    "kind": KINDS[cls % len(KINDS)],
                }
            )
    return boxes


def seed(base: str, boxes: list[dict]) -> None:
    call(base, "/api/bind-recording", {"kind": "session", "name": "demo"})
    call(
        base,
        "/api/find-run",
        {"boxes": [{k: v for k, v in b.items() if k != "kind"} for b in boxes]},
    )
    tiles = [t["name"] for c in call(base, "/api/clusters")["clusters"] for t in c["items"]]
    # tiles are named by candidate index, so the kind rides along by position
    by_kind: dict[str, list[str]] = {}
    for name in tiles:
        index = int(name.rsplit("_k", 1)[-1])
        by_kind.setdefault(boxes[index]["kind"], []).append(name)
    for kind, names in by_kind.items():
        call(
            base,
            "/api/manual-label",
            {"names": names[:-1] or names, "definition": "kinds", "picked": {"kind": kind}},
        )
    call(base, "/api/save-labels", {})
    call(base, "/api/promote-training", {"name": "demo_set"})
    call(base, "/api/unbind-recording", {})


def shoot(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp, pytest.MonkeyPatch.context() as patch:
        world = review_world.make_world(Path(tmp), patch, recordings=("demo",), frames=0)
        frames_dir = world.sessions / "demo" / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        boxes = write_frames(frames_dir)
        review_world.save_definition("kinds", KINDS)
        app = build_app(
            # the real heatmap backend: a short run on four frames gives a real loss curve and a
            # real separation readout, which the suite's fake backend cannot
            backend="heatmap",
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
            seed(base, boxes)
            with sync_playwright() as pw:
                browser = pw.chromium.launch()
                page = browser.new_page(viewport={"width": 1280, "height": 860})
                page.goto(base + "/")
                page.wait_for_timeout(400)

                # find: the recording bound, its boxes drawn on the first frame
                show_tab(page, "find")
                page.click("#draw-open")
                page.wait_for_selector('.menu-panel .menu-item:has-text("demo")')
                page.click('.menu-panel .menu-item:has-text("demo")')
                page.wait_for_selector('.menu-panel .toggle:has-text("open")')
                page.click('.menu-panel .toggle:has-text("open")')
                page.wait_for_function(
                    'document.getElementById("draw-count") && document.getElementById("draw-count").innerText.includes("box")'
                )
                dismiss(page)
                page.wait_for_timeout(300)
                page.screenshot(path=str(out / "find.jpg"), type="jpeg", quality=85)

                # select: the pool with its classes assigned
                show_tab(page, "select")
                page.wait_for_selector(".cluster-item")
                page.wait_for_timeout(300)
                page.screenshot(path=str(out / "select.jpg"), type="jpeg", quality=85)

                # train: bind the set, run, sweep
                show_tab(page, "train")
                page.click("#train-open")
                page.wait_for_selector('.menu-panel .menu-item:has-text("demo_set")')
                page.click('.menu-panel .menu-item:has-text("demo_set")')
                page.wait_for_function(
                    'document.getElementById("train-summary").innerText.includes("objects")'
                )
                dismiss(page)
                # 200 epochs is the default; twenty is enough for a curve and quick on cpu
                for _ in range(4):
                    page.click('[data-stepper="epochs"] [data-step="-50"]')
                page.click('[data-stepper="epochs"] [data-step="50"]')
                page.click('[data-stepper="epochs"] [data-step="-50"]')
                page.click("#train-start")
                page.wait_for_function(
                    'document.getElementById("train-progress-text").innerText.startsWith("done")',
                    timeout=600000,
                )
                # the summary and the separation readout refresh after the run reports done
                page.wait_for_function(
                    'document.getElementById("train-summary").innerText.includes("ready to sweep")',
                    timeout=30000,
                )
                page.wait_for_timeout(1500)
                page.evaluate("window.scrollTo(0, 0)")
                page.wait_for_timeout(200)
                page.screenshot(path=str(out / "train.jpg"), type="jpeg", quality=85)
                page.click("#sweep-open")
                page.click('.menu-item:has-text("demo")')
                page.click("#sweep-start")
                page.wait_for_function(
                    'document.getElementById("sweep-progress-text").innerText.includes("proposals")'
                    ' || document.getElementById("sweep-progress-text").innerText.includes("floor")',
                    timeout=300000,
                )
                page.wait_for_timeout(300)
                page.locator("#sweep-send").scroll_into_view_if_needed()
                page.screenshot(path=str(out / "train-sweep.jpg"), type="jpeg", quality=85)

                show_tab(page, "housekeeping")
                page.wait_for_timeout(400)
                page.screenshot(path=str(out / "housekeeping.jpg"), type="jpeg", quality=85)
                browser.close()
        finally:
            server.shutdown()
            server.server_close()
    print(f"wrote {', '.join(p.name for p in sorted(out.glob('*.jpg')))} to {out}")


if __name__ == "__main__":
    shoot(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("docs/images"))
