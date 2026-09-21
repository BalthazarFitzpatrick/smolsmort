# smolsmort

A small detector you train by judging its guesses. It finds objects on screen, proposes new ones on
frames nobody has labelled, and those proposals come back to you to judge. Every round of judging
makes the next model better. The model is swappable: a fixed-size and a variable-size one are built
in, and your own plugs in by name.

```
  frames ──find──▶ candidates ──cut──▶ tiles ──judge──▶ classes
                                                          │
                                                       promote
                                                          ▼
  proposals ◀──sweep── weights ◀──────train─────── training set
      │
      └──────────▶ judge again   (the loop closes here)
```

The loop closes at judging. A wrong proposal becomes a labelled hard negative instead of being
thrown away, and that is what makes the next model better rather than just retrained.

## Does it fit your problem?

One question picks the backend: **is the object a fixed, known size in pixels?**

- **Yes: `heatmap`.** The model only answers *where* and *which class*. It regresses no width or
  height, which is why it has about 100,000 parameters instead of millions.
- **No: `box`.** It predicts each object's own box, for objects that vary 4x and more in size, in
  frames of different resolutions. About 844,000 parameters. It is new and proven only on synthetic
  frames: it found 97% of objects standing apart, at a median IoU of 0.74, but only about half of
  those overlapping another.

| your setup | backend |
|---|---|
| Fixed camera, objects at one distance: a tank, a conveyor, a counting window, a game's UI | `heatmap` |
| Top-down camera at a fixed height | `heatmap` |
| Objects at different distances, or classes of very different sizes | `box` |
| Objects that crowd and overlap a lot | `box`, but overlap is its measured weak spot |
| A moving camera | untested with either |
| You need outlines, not positions | neither, you want segmentation |

Quick test: measure your object in twenty frames. If the largest is more than about 1.5 times the
smallest, use `box`. [docs/FISH.md](docs/FISH.md) walks through a full setup with `heatmap` on a
worked example.

## Install

Not on PyPI. Pin a commit from git:

```bash
uv add "smolsmort[vision] @ git+https://github.com/BalthazarFitzpatrick/smolsmort.git@82d7374"
```

`[vision]` pulls torch (about 2 GB). Without it you get the box maths, scoring and tracking, which
need only numpy and pillow. Pin `82d7374` or later: the `v0.1.0` tag predates a fix that moved every
decoded peak 8 px down and right.

Python 3.11 and newer. CI runs 3.11 and 3.14 on Linux.

## Quickstart

Train on labelled centres, then find objects on frames the model has not seen. This runs as written
on synthetic frames with a 132x12 bar on noise:

```python
from smolsmort.detect.box import Box
from smolsmort.detect.dataset import Example
from smolsmort.detect.model import decode_peaks
from smolsmort.detect.scoring import boxes_from_peaks, score
from smolsmort.detect.train import heatmaps_for, load, save, train

# one Example per frame: where the objects are, in capture pixels
examples = [Example(path=path, centres=[(cx, cy)]) for path, (cx, cy) in labelled]

model, history = train(examples, epochs=40, batch=8)
save(model, "bars.pt")

model = load("bars.pt")
maps = heatmaps_for(model, "frame.png")  # (classes, h, w), one channel per class
peaks = decode_peaks(maps[0], limit=3)  # centres in capture pixels, strongest first
boxes = boxes_from_peaks(peaks, width=132, height=12)

print("\n".join(score([boxes], [[Box(left, top, 132, 12)]]).lines()))
```

On 20 training frames, 40 epochs on CPU found all 4 held-out bars within 8 px. Two things to know
from that run:

- Weaker secondary peaks around 0.4 appear beside a strong one. `decode_peaks` keeps anything above
  `min_score` (0.35 by default), so raise it or pass `limit`.
- `score` reports FAIL when the truth has no empty frames, even at 100% recall. False positives can
  only be counted on frames with nothing in them, so include some.

## Adding a tab to the review UI

`uv run smolsmort` serves the review page: find, select, train and housekeeping tabs on the shared
`ui_base` kit. The tab bar is built from a list, so a host page adds its own tab by loading a
script after `boot.js` that calls:

```js
window.smolsmortTabs.register({
  id: 'mytab',          // unique, used as the panel's data-panel
  label: 'my tab',      // shown in the tab bar
  mount(panelEl) {      // runs once; fill panelEl with your markup
    panelEl.textContent = 'hello';
    return {enter() {}};  // optional: called each time the tab is shown
  },
});
```

## Swapping the model

Every backend has the same four methods: train, predict, save and load. The loop drives whichever
one it is handed and never learns which:

```python
from smolsmort import backends

backend = backends.get_backend("box")  # or "heatmap"
backends.register("mine", "my_package.backend", "MyBackend")  # your own model, no smolsmort edit
```

Listing or picking a backend never imports torch until a vision backend is actually built. A JSON
file saved beside each checkpoint names the backend that wrote it (`backends.backend_of`), and a
checkpoint without one is a heatmap checkpoint. `smolsmort.boxes.synthetic` writes frames with known
boxes, for proving a setup before any real data is labelled.

## How it works

**The model.** A fully convolutional net with one heatmap channel per class. It runs on the frame
downscaled 4x, and the heatmap has a stride of 4 on top of that, so one heatmap cell covers 16x16
capture pixels. A wide 3x9 kernel near the end helps it see across long, thin objects. Parameters:
99,769 with one class, 100,210 with ten. Fully convolutional means it trains on small crops and runs
on whole frames of any size.

**Peaks, not boxes.** `decode_peaks` finds local maxima above a threshold, suppresses neighbours
within 3 cells, and returns centres in capture pixels. Since the object size is known, the box
follows from the centre (`boxes_from_peaks`). The channel a peak comes from is its class.

**The box backend.** The same heatmap idea, plus two more outputs at each cell: the object's log
width and height, and where its centre sits inside the cell. An encoder down to stride 32 gives each
cell a view of about 440 px, enough to size the largest object it is trained on, and training
refuses an object bigger than that rather than sizing it wrong. Every frame is scaled to one working
long side (768 px) first, which is why frames of different resolutions can train together.

**Training.** Each batch is cut into 256 px windows (1024 capture pixels). Half the windows are
centred on a real object, offset up to 40% of the window so the model learns objects anywhere, not
just near the middle. A quarter are centred on hard negatives when a frame has them, the rest are
random. Targets are small Gaussian blobs snapped to cell centres, trained with a CentreNet-style
focal loss. Channels train independently, so a rare class is not drowned out by a common one.

**What a label means.** Keeping a candidate makes it a positive. Discarding one does not make it a
negative: a discard can mean misaligned or redundant just as often as "not an object". Discarded and
never-reviewed candidates become ignore regions the loss does not look at. Explicit negatives come
only from places the model fired and was told no. A frame can also be marked exhaustive, meaning
every object on it was proposed and judged. There, anything proposed and not kept becomes a
negative, except a discard centred inside a kept box. A `heatmap` training set must use one capture
resolution, since the object is a different pixel size on each screen, and mixing them is refused;
`box` scales every frame to one working size instead.

**Per-set and per-frame settings in the review tool.** Each is a small optional json file, and an
absent file means the old behaviour:
- `<set>._backend.json` beside a training set holds `{"backend", "size_mode"}`. No file means
  `heatmap` and `uniform`. `native` keeps every drawn box at its own width and height; `uniform`
  fits one size to the set. Written by `POST /api/train-set-backend {name, backend, size_mode?}`
  (`size_mode` defaults to `native` for `box`, else `uniform`; an unknown backend answers `{error}`
  listing the names), read by `GET /api/train-info`, `train-bind` and `train-start`. Drawing for a
  set passes `set` to `/api/find-run`.
- `<recording>._frames.json` in the labels folder holds `{frame: {"exhaustive": true}}`. No entry
  means explicit. `GET /api/frame-modes?recording=` and `POST /api/frame-mode {recording, frame,
  exhaustive}`. On promotion, unjudged boxes on an exhaustive frame become negatives.
- A tile record in `_labels.json` may carry `not_object: true`, the verdict "this is not an
  object". It excludes label and discard, and is a hard negative on any frame. Discard is unchanged.
  `POST /api/manual-label` takes `{names|name, not_object: true}` and buffers it until
  `save-labels`.

**Sweeping.** `sweep` runs trained weights over whole frames and returns candidates in the same
schema the judging step reads, capped per frame and strongest first. It decodes the next frames on
worker threads while the model runs: at 3420x2224, decoding a JPEG took 51 ms against 39 ms for the
forward pass, so the model was otherwise waiting on pillow.

**Scoring.** Matching is per frame and positional. Two boxes match when they cover the same span on
the same row. IoU fails on thin objects: on a 132x12 bar, an 8 px vertical offset drops IoU to 0.12.
For squarer or variable-size objects, pass `match=iou_match(0.5)` instead. Empty frames are scored,
because that is the only place a false positive can be counted cleanly. The built-in pass marks are
90% recall with at most 5% false-positive frames. `beats_the_teacher` compares against a 75% recall,
75% precision classical detector, the baseline the model was built to beat.

**Tracking.** `track` follows one object across a recording from per-frame detections:
- With no history, it takes the strongest peak. After that, it takes the strongest peak within a
  plausible step, not the nearest, so it does not chase decoys.
- `despike` drops frames that disagree with the median of their neighbours, which catches a tracker
  that re-acquired onto a decoy and followed it smoothly.
- `chrome_cells` finds detections that stay put while the camera turns. Those are part of the
  interface, not the world.

## Defaults and where they came from

Every tuned number was measured on the first consumer, a game-screen detector for long, thin bars
about 132x12 px at 2560 wide. They are parameters with a stated origin, so pass your own for a
different object.

| setting | default | where it lives |
|---|---|---|
| same object across frames | 40 px corner distance | `box.SAME_OBJECT_PX` |
| same object within a frame | half the narrower width shared, within 1.5 box heights | `box.SAME_OBJECT_MIN_SHARE`, `SAME_OBJECT_MAX_ROWS_APART` |
| input downscale / heatmap stride | 4 / 4 | `model.DOWNSCALE`, `model.STRIDE` |
| peak threshold / separation | 0.35 / 3 cells | `model.PEAK_MIN_SCORE`, `PEAK_MIN_SEPARATION` |
| training window / offset | 256 px / 40% | `train.CROP`, `train.JITTER_FRACTION` |
| tracker step / lost after | 600 px / 5 frames | `track.MAX_STEP_PX`, `LOST_AFTER_FRAMES` |
| box backend working long side | 768 px, from a synthetic spike | `boxes.model.WORK_LONG_SIDE` |

`train.minimum_window(width, height)` tells you the smallest window that never clips your object at
the offset extremes. `train.snapped_window` rounds a window up to a whole number of cells, which the
model and target need to agree on.

## In use

A private game-overlay project finds nameplates with it: 10 classes,
100,210 parameters, about 44 ms per 3420x2224 frame on an Apple M4 including the JPEG decode. Its
first honest evaluation on a hand-drawn holdout returned 24% precision, with the highest-scoring
detections the wrong ones. That is what the loop exists to find out. Read a falling loss as "training
converged", never as "the model is right". Only a holdout tells you that.

## Status

- **Here now:** the heatmap backend in `smolsmort/detect/` (box, model, dataset, train, scoring,
  track), the box backend in `smolsmort/boxes/`, and backends by name in `smolsmort/backends.py`.
- **In progress:** the review web tool, where candidates are found, judged and promoted. It is being
  ported from that project in PR #2.
- **Planned:** a train tab that picks the backend by name, and a tabular backend (xgboost) beside the
  vision ones.

The API will move until the port is finished.

## Troubleshooting

### A smortboard card stopped on `LEASE_CONFLICT`

Work on this repo runs as [smortboard](https://github.com/BalthazarFitzpatrick/smortboard) cards. A
card may only write the files its lease allows; when it writes outside it, the card stops with
`LEASE_CONFLICT` and its note names the files it needed. Widen the lease, then resume the card:

```bash
curl -s -X PATCH 127.0.0.1:8000/api/cards/<card-id> \
  -H 'content-type: application/json' \
  -d '{"leases": ["smolsmort/detect/**", "tests/**"]}'
curl -s -X POST 127.0.0.1:8000/api/cards/<card-id>/answer \
  -H 'content-type: application/json' \
  -d '{"message": "lease widened to smolsmort/detect/**, tests/** - go ahead"}'
```

The PATCH replaces the whole lease, so repeat any glob it should keep. Card ids come from
`curl -s 127.0.0.1:8000/api/boards/<board-id>/cards`. The full procedure, and how globs are
matched, is in [smortboard's troubleshooting](https://github.com/BalthazarFitzpatrick/smortboard#troubleshooting).

Common leases for this repo:

| Change | Lease |
|---|---|
| the heatmap backend | `smolsmort/detect/**`, `tests/**` |
| the box backend | `smolsmort/boxes/**`, `tests/test_boxes.py`, `tests/test_variable_boxes.py` |
| backends by name | `smolsmort/backends.py`, `tests/**` |
| the review tool | `smolsmort/review/**`, `smolsmort/review_ui/**`, `tests/**` |
| docs | `README.md`, `docs/**` |

## Licence

MIT, see [LICENSE](LICENSE).

## Extra tabs

A host adds tabs without editing the loop: build a `smolsmort.review.routes.Tab` with its own
`get` and `post` route tables (full paths, json in and out) and optional `images`, and pass it in
`build_app(tabs=[...])`. A path that collides with a core route is refused when the server is
built. `GET /api/tabs` lists the registered names so a page can offer them.
