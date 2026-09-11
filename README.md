# smolsmort

A small detector you train by judging its guesses. It finds objects that are always the same size
on screen, proposes new ones on frames nobody has labelled, and those proposals come back to you to
judge. Every round of judging makes the next model better.

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

One assumption decides it: **the object is a fixed, known size in pixels.** The model only answers
*where* and *which class*. It regresses no width or height, which is why it has about 100,000
parameters instead of millions.

| your setup | fits? |
|---|---|
| Fixed camera, objects at one distance: a tank, a conveyor, a counting window, a game's UI | yes |
| Top-down camera at a fixed height | yes |
| Objects at any distance, or a moving camera where apparent size changes | no, use a box-regression detector (YOLO, DETR) |
| You need outlines, not positions | no, you want segmentation |

Quick test: measure your object in twenty frames. If the largest is more than about 1.5 times the
smallest, the assumption is broken. [docs/FISH.md](docs/FISH.md) walks through a full setup on a
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
maps = heatmaps_for(model, "frame.png")          # (classes, h, w), one channel per class
peaks = decode_peaks(maps[0], limit=3)           # centres in capture pixels, strongest first
boxes = boxes_from_peaks(peaks, width=132, height=12)

print("\n".join(score([boxes], [[Box(left, top, 132, 12)]]).lines()))
```

On 20 training frames, 40 epochs on CPU found all 4 held-out bars within 8 px. Two things to know
from that run:

- Weaker secondary peaks around 0.4 appear beside a strong one. `decode_peaks` keeps anything above
  `min_score` (0.35 by default), so raise it or pass `limit`.
- `score` reports FAIL when the truth has no empty frames, even at 100% recall. False positives can
  only be counted on frames with nothing in them, so include some.

## How it works

**The model.** A fully convolutional net with one heatmap channel per class. It runs on the frame
downscaled 4x, and the heatmap has a stride of 4 on top of that, so one heatmap cell covers 16x16
capture pixels. A wide 3x9 kernel near the end helps it see across long, thin objects. Parameters:
99,769 with one class, 100,210 with ten. Fully convolutional means it trains on small crops and runs
on whole frames of any size.

**Peaks, not boxes.** `decode_peaks` finds local maxima above a threshold, suppresses neighbours
within 3 cells, and returns centres in capture pixels. Since the object size is known, the box
follows from the centre (`boxes_from_peaks`). The channel a peak comes from is its class.

**Training.** Each batch is cut into 256 px windows (1024 capture pixels). Half the windows are
centred on a real object, offset up to 40% of the window so the model learns objects anywhere, not
just near the middle. A quarter are centred on hard negatives when a frame has them, the rest are
random. Targets are small Gaussian blobs snapped to cell centres, trained with a CentreNet-style
focal loss. Channels train independently, so a rare class is not drowned out by a common one.

**What a label means.** Keeping a candidate makes it a positive. Discarding one does not make it a
negative: a discard can mean misaligned or redundant just as often as "not an object". Discarded and
never-reviewed candidates become ignore regions the loss does not look at. Explicit negatives come
only from places the model fired and was told no. A training set must also use one capture
resolution, since the object is a different pixel size on each screen. Mixing them is refused.

**Sweeping.** `sweep` runs trained weights over whole frames and returns candidates in the same
schema the judging step reads, capped per frame and strongest first. It decodes the next frames on
worker threads while the model runs: at 3420x2224, decoding a JPEG took 51 ms against 39 ms for the
forward pass, so the model was otherwise waiting on pillow.

**Scoring.** Matching is per frame and positional. Two boxes match when they cover the same span on
the same row. IoU fails on thin objects: on a 132x12 bar, an 8 px vertical offset drops IoU to 0.12.
Empty frames are scored, because that is the only place a false positive can be counted cleanly. The
built-in pass marks are 90% recall with at most 5% false-positive frames. `beats_the_teacher`
compares against a 75% recall, 75% precision classical detector, the baseline the model was built
to beat.

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

`train.minimum_window(width, height)` tells you the smallest window that never clips your object at
the offset extremes. `train.snapped_window` rounds a window up to a whole number of cells, which the
model and target need to agree on.

## In use

[consumer-app](https://github.com/BalthazarFitzpatrick/consumer-app) finds nameplates with it: 10 classes,
100,210 parameters, about 44 ms per 3420x2224 frame on an Apple M4 including the JPEG decode. Its
first honest evaluation on a hand-drawn holdout returned 24% precision, with the highest-scoring
detections the wrong ones. That is what the loop exists to find out. Read a falling loss as "training
converged", never as "the model is right". Only a holdout tells you that.

## Status

- **Here now:** the vision backend in `smolsmort/detect/` (box, model, dataset, train, scoring,
  track), with 75 tests.
- **In progress:** the review web tool, where candidates are found, judged and promoted. It is being
  ported from consumer-app in PR #2.
- **Planned:** plugin seams so the loop runs with other models and renderers, and a tabular backend
  (xgboost) beside the vision one.

The API will move until the port is finished.

## Licence

MIT, see [LICENSE](LICENSE).
