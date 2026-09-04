# Setting up smolsmort to recognise fish

A worked guide, written to be followed by a person or executed by an agent. Every command here has
been run; every function named exists. Where something does not exist yet, it says so rather than
describing it as though it does.

---

## 0. First decide whether this fits your problem

**Read this before installing anything.** smolsmort's vision backend makes one strong assumption,
and if your problem breaks it the rest of the guide will waste your afternoon.

> The object is a **fixed known size** on screen. The model predicts *where* something is, and
> regresses no width or height at all.

That is why it is ~50k parameters instead of millions. It is also a real limit.

| your setup | does it fit? |
|---|---|
| Fixed camera over a **tank**, **conveyor**, **counting window** or **fish ladder** — fish pass at roughly one distance | **Yes.** This is the good case. |
| **Top-down** camera at a fixed height, fish at one depth | **Yes.** |
| Free-swimming fish at any distance, or a moving/handheld camera | **No.** Apparent size varies with distance, and a centre-only model cannot express that. Use a box-regression detector (YOLO, DETR) instead. |
| You need the fish **outlined**, not located | **No.** You want segmentation. |
| You need **species**, and each species has its own channel | Yes, with a caveat — see §6. |

**Quick self-test.** Take twenty frames, measure a fish in pixels in each. If the largest is more
than roughly 1.5× the smallest, the fixed-size assumption is broken and you should stop here.

---

## 1. Install

Public repo, no credentials needed.

```bash
uv add "smolsmort[vision] @ git+https://github.com/BalthazarFitzpatrick/smolsmort.git@v0.1.0"
```

The `[vision]` extra is what pulls **torch** (~2 GB). Without it you get the loop and the box maths
but no model — deliberate, so a consumer doing non-image work is not made to install it.

Verify:

```bash
uv run python -c "from smolsmort.detect import model, train, dataset, scoring; print('ok')"
```

---

## 2. What you have to supply, and what does not exist yet

**Be clear-eyed about this.** smolsmort v0.1.0 ships the *backend*: the model, the training loop,
the sweep, the scoring, box maths, and tracking. The **human-in-the-loop review web tool — the part
that lets you draw boxes and judge crops in a browser — is not extracted yet.** It still lives in
the project smolsmort was pulled out of.

So today you must produce labelled centres yourself. Options, cheapest first:

1. **Label a few dozen frames by hand** in any tool that exports box coordinates (Label Studio,
   CVAT, even a spreadsheet of `frame,x,y`). You need centres, not boxes.
2. **Bootstrap from a classical detector** if your fish are high-contrast against the background —
   a background subtraction or blob detector gets you 70%-ish, and you correct the rest. That is
   exactly how the original project started, and the term used for it is a *teacher*.

You need **one number measured up front**: the typical fish size in pixels, width and height. Every
step below uses it.

```bash
# measure it honestly - open five frames and note a typical fish's bounding box
FISH_W=120
FISH_H=48
```

---

## 3. Build the training examples

An `Example` is one frame plus everything known about it.

```python
from pathlib import Path
from smolsmort.detect.dataset import Example

examples = [
    Example(
        path=Path("frames/0001.jpg"),
        centres=[(412.0, 233.0), (690.0, 251.0)],   # fish centres, in FRAME pixels
        labels=[None, None],                         # or ["trout", "salmon"] - see §6
        negatives=[(120.0, 400.0)],                  # places you KNOW hold no fish
        ignore=[],                                   # regions to exclude from the loss entirely
    ),
    # ...
]
```

Three fields carry more weight than they look:

- **`centres`** — the centre of the fish, not a corner. In the frame's own pixel coordinates.
- **`negatives`** — somewhere you have confirmed there is no fish. Worth more per example than a
  positive, because it teaches the model *its own* mistakes rather than ones you guessed at. Feed
  it every false positive the model produces.
- **`ignore`** — for regions you have not labelled. Frames are usually only *partly* labelled: if
  you drew two fish but a third is in the corner unmarked, that third teaches the model "fish are
  background" unless the region is ignored. This field is how you stay honest about incompleteness.

**How many frames?** Enough that thin cases appear at all. Fifty labelled frames is a real start;
the loop below is how it grows.

---

## 4. Train

```python
from smolsmort.detect.train import train, save

model, history = train(
    examples,
    epochs=30,
    crop=256,              # training window, in INPUT px. See the note below.
    on_progress=lambda p: print(p),
)
save(model, Path("weights/fish-v1.pt"))
```

**`crop` has a floor and the library will tell you where it is.** The window must be large enough
that a fish still fits after the training jitter offsets it, or you teach the net half a fish as a
whole one:

```python
from smolsmort.detect.train import minimum_window
print(minimum_window(FISH_W, FISH_H))   # the smallest crop that will not clip
```

Note the net runs on a **4×-downscaled** frame, so a 120px fish is 30px to the model — several
cells across at stride 4, which is what you want. If your fish are much smaller than ~40px in the
original frame, reduce the downscale or move the camera closer.

---

## 5. Predict, then feed the mistakes back

```python
from smolsmort.detect.train import sweep, load

model = load(Path("weights/fish-v1.pt"))
candidates = sweep(
    model, {"fish": 0}, sorted(Path("unlabelled/").glob("*.jpg")),
    width=FISH_W, height=FISH_H,
    min_score=0.30,        # deliberately low - see below
    max_per_frame=12,
)
```

Each candidate is a dict with `path`, `left`, `top`, `width`, `height`, `score`.

**Set `min_score` low and filter afterwards, not the reverse.** A floor above what the model
actually produces returns zero boxes and reports success — that happened three times in the parent
project. Look at the score *distribution* first, then choose.

**This is the loop, and it is the whole point:**

1. Sweep frames nobody has labelled.
2. Look at what came back. Right ones → new positives. Wrong ones → **new `negatives`**.
3. Retrain with the corrections included.
4. Repeat.

A wrong prediction is worth more than a right one: it is a labelled example of the exact confusion
this model makes, which is not something you could have guessed.

---

## 6. Species: one channel each

```python
model, history = train(examples, classes={"trout": 0, "salmon": 1, "pike": 2}, epochs=30)
```

Channels are trained **independently** — a trout is not a negative example for salmon, it simply
leaves the salmon channel empty where it stands. So adding a species is cheap and a rare species is
not drowned out by a common one, which a single softmax over three classes would do.

**Do not add a channel before you have examples of it.** A class with three examples produces a
channel that fires on noise, and it will look like the model working.

---

## 7. Measure it against a holdout, and be suspicious of good news

```python
from smolsmort.detect.scoring import score, boxes_from_peaks
from smolsmort.detect.box import Box

result = score(predictions, truths)   # both: one list of Box per frame, same order
print("\n".join(result.lines()))
```

**Three warnings, each learned the hard way:**

**Hold frames back that the model has never influenced.** If your labels came from correcting an
earlier model's output, scoring against them measures agreement with that model, not correctness.
In the parent project a detector reporting a healthy internal score turned out to score **24%
precision** on genuinely unseen hand-drawn frames — and the *highest-scoring* detections were the
wrong ones.

**A separation or confidence score is not accuracy.** Read it as "did training converge", never as
"is it right".

**Check your holdout is the same resolution and scale.** Comparing against frames from a different
camera setup measures the mismatch, not the model. In the parent project a holdout at a different
resolution produced apparently catastrophic results that were entirely the scale gap.

---

## 8. Tracking, if your fish move between frames

```python
from smolsmort.detect.track import track
positions = track(per_frame, ordered_paths)   # {frame path: (x, y)}
```

`per_frame` maps a frame path to its candidate list. **`ordered_paths` must be in recording order** —
each frame is read against the previous one, so shuffling silently turns tracking back into
independent peak-picking.

It handles the three failure modes that matter: no history (takes the strongest), a decoy nearby
(takes the strongest among the *plausible*, not the nearest), and a smooth-but-wrong run after a bad
re-acquire (a median over a neighbourhood, which is the only thing that catches it).

---

## Checklist for an agent

```
[ ] measured typical fish size in px (W, H); confirmed largest/smallest < 1.5x
[ ] installed smolsmort[vision]; import check passes
[ ] >= 50 frames with centres; unlabelled regions in `ignore`
[ ] crop >= minimum_window(W, H)
[ ] trained, weights saved
[ ] swept unlabelled frames with a LOW min_score; looked at the distribution
[ ] false positives added as `negatives`; retrained
[ ] scored against a holdout the model never influenced
[ ] holdout is the same camera, resolution and scale as training
```

## Where this can go wrong quietly

- **A partly-labelled frame with no `ignore`** teaches the model that fish are background.
- **A `min_score` above the model's output range** returns nothing and looks like success.
- **A holdout derived from model output** reports agreement, not accuracy.
- **Fish at varying distance** breaks the fixed-size assumption; the failure looks like a model that
  never quite converges, not like a wrong choice of tool.
