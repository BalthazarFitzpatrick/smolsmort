"""training loop for the heatmap cnn.

PIPELINE FIRST. This exists to prove the whole path works end to end - decisions to examples to
targets to a trained net to a heatmap you can look at - not to produce a good detector. There are
27 labelled frames at the time of writing and the labels are known incomplete; the honest use of
the result is "does the object light up brighter than the background", not a precision figure.

TRAINS ON CROPS, RUNS ON FRAMES. The net is fully convolutional, so it can be trained on small
windows and then applied to a whole 2560x1440 capture in one pass. Crops are what make the batch
affordable and, more importantly, what let each batch be BALANCED: a random window of a frame is
almost always empty, so sampling uniformly would feed the net thousands of blank skies for every
object. Half of each batch is centred on a real one instead.

THE IGNORE MASK IS THE POINT. Candidates that were mined but never confirmed, and candidates the
reviewer discarded, are neither positive nor negative - see detect.dataset. Their pixels are zeroed
out of the loss so the net is never told "there is nothing here" about a place that may well hold a
object nobody ruled on.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from smolsmort.detect.dataset import Example
from smolsmort.detect.model import (
    DOWNSCALE,
    STRIDE,
    _torch,
    build_model,
    class_target,
    decode_peaks,
    gaussian_target,
)

CROP = 256  # input pixels, i.e. 1024 capture pixels a side after DOWNSCALE

# how far a training window may be offset from the object it is centred on, as a fraction of the
# window. a module constant so the two settings can be A/B'd over the SAME seeds - the effect is
# smaller than the seed-to-seed spread, so a single-seed comparison proves nothing either way.
JITTER_FRACTION = 0.40


def snapped_window(window: int) -> int:
    """the nearest usable crop at or above `window`: a whole number of STRIDE cells.

    THE MODEL CEILS AND THE TARGET FLOORS. Both stride-2 convs use padding=1, so the network turns
    an S-pixel window into ceil(S/4) cells, while build_target lays out S // 4 - and those agree
    only when S divides by STRIDE. A window of 286 gives a 72-cell prediction against a 71-cell
    target and torch raises "size of tensor a (72) must match the size of tensor b (71)" deep in
    the loss, naming neither the crop nor the setting that produced it.

    Snapped UP so a human asking for a wider window never silently gets a narrower one - the reason
    to widen it is usually that something was being clipped.
    """
    if window % STRIDE == 0:
        return window
    return window + (STRIDE - window % STRIDE)


def minimum_window(box_width: int, box_height: int, jitter: float = JITTER_FRACTION) -> int:
    """the smallest training window that always contains a box of this size, whole.

    A WINDOW, A JITTER AND A BOX SIZE ARE THREE NUMBERS THAT MUST AGREE, and two of them silently
    did not. _crop_window offsets a window by up to `jitter` of its own size, so the box centre can
    sit (0.5 - jitter) of the way from an edge; the box fits only while its half-size is smaller
    than that. At CROP=256 and jitter=0.40 that limit is 25.6 input px, and the retina reference object is 226
    capture px = 28.2 half - so it was CLIPPED BY 2.7px at the extremes, teaching a half-object as a
    whole one, which is exactly what the jitter comment above says the margin was chosen to avoid.
    """
    longest = max(box_width, box_height) / DOWNSCALE
    return int(math.ceil(longest / (1.0 - 2.0 * jitter)))


class TrainError(Exception):
    pass


def masked_focal_loss(logits, target, mask, alpha: float = 2.0, beta: float = 4.0):
    """model.focal_loss, but per-element and with cells the mask zeroes left out entirely.

    the shared version returns a SCALAR already normalised by the positive count, so a mask cannot
    be applied to its result - the denominator would be wrong. it also applies its own sigmoid, so
    it takes logits, not probabilities. both of those were got wrong first time here.
    """
    torch = _torch()
    prediction = torch.sigmoid(logits).clamp(1e-4, 1 - 1e-4)
    positive = target.ge(0.99).float() * mask
    negative = (1.0 - target.ge(0.99).float()) * mask
    positive_loss = -((1 - prediction) ** alpha) * torch.log(prediction) * positive
    negative_loss = (
        -((1 - target) ** beta) * (prediction**alpha) * torch.log(1 - prediction) * negative
    )
    count = positive.sum()
    total = positive_loss.sum() + negative_loss.sum()
    return total / count if count > 0 else negative_loss.sum()


@dataclass
class Progress:
    epoch: int
    epochs: int
    loss: float
    seen: int

    @property
    def fraction(self) -> float:
        return self.epoch / self.epochs if self.epochs else 0.0


def _load_input(path: Path) -> np.ndarray:
    """a capture as the net sees it: downscaled, rgb, channels-first, 0..1"""
    from PIL import Image

    with Image.open(path) as handle:
        image = handle.convert("RGB")
        image = image.resize((image.width // DOWNSCALE, image.height // DOWNSCALE), Image.BILINEAR)
    return np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0


def _crop_window(
    example: Example,
    image: np.ndarray,
    rng: random.Random,
    classes=None,
    window: int = CROP,
):
    """one training window, half the time centred on a real object.

    returns the window, its heatmap target and the ignore mask, all already at their own scales:
    the target and mask are at STRIDE cells, the window at input pixels.
    """
    _, height, width = image.shape
    size = min(window, height, width)
    roll = rng.random()
    if example.centres and roll < 0.5:
        cx, cy = rng.choice(example.centres)
        cx, cy = cx / DOWNSCALE, cy / DOWNSCALE
        # THE OBJECT MUST LAND ANYWHERE IN THE WINDOW, NOT NEAR THE MIDDLE. this used to jitter by
        # +/- size/6, which on a 256px window pins the object within ~42px of the centre - so the net
        # was taught "objects are near the middle" and then asked to sweep whole frames, where they
        # are anywhere. Balthazar Fitzpatrick spotted the same gap from the data side, asking whether the object
        # could be off-centre in the training crops. 0.40 keeps a margin so the object stays fully
        # inside rather than clipped at the edge, which would teach a half-object as a whole one.
        jitter = size * JITTER_FRACTION
        left = int(cx - size / 2 + rng.uniform(-jitter, jitter))
        top = int(cy - size / 2 + rng.uniform(-jitter, jitter))
    elif example.negatives and roll < 0.75:
        # HARD NEGATIVES. these are places the model ITSELF fired on and a different signal (the
        # template matcher) says are not objects. a uniformly random window almost never contains
        # one, so without deliberately sampling them the model never revisits its own mistakes -
        # appending them to a list changes nothing on its own, since background is already the
        # default everywhere. this is what makes the mining actually train.
        cx, cy = rng.choice(example.negatives)
        cx, cy = cx / DOWNSCALE, cy / DOWNSCALE
        left = int(cx - size / 2 + rng.uniform(-size / 6, size / 6))
        top = int(cy - size / 2 + rng.uniform(-size / 6, size / 6))
    else:
        left, top = rng.randrange(0, max(1, width - size)), rng.randrange(0, max(1, height - size))
    left = max(0, min(left, width - size))
    top = max(0, min(top, height - size))

    window = image[:, top : top + size, left : left + size]
    cells = size // STRIDE
    # labels ride along with their centre so a window keeps them aligned after clipping. an
    # example built before labels existed has none, so pad rather than zip-strict
    example_labels = list(example.labels) + [None] * (len(example.centres) - len(example.labels))
    kept = [
        (((cx / DOWNSCALE - left) / STRIDE, (cy / DOWNSCALE - top) / STRIDE), label)
        for (cx, cy), label in zip(example.centres, example_labels, strict=True)
        if 0 <= cx / DOWNSCALE - left < size and 0 <= cy / DOWNSCALE - top < size
    ]
    centres = [c for c, _ in kept]
    # SNAP TO CELL CENTRES. gaussian_target builds its blob from the coordinate given, so a
    # fractional centre peaks below 1.0 - measured: 0.9314 for a centre at (8.4, 8.6). The loss
    # counts positives with target >= 0.99, so fractional centres produced ZERO positive cells,
    # only the negative term trained, and the net learned to answer a flat 0.145 everywhere.
    # Rounding costs at most half a cell of localisation (2 capture pixels at STRIDE 4, DOWNSCALE 4)
    # and is what CentreNet does for the same reason.
    snapped = [(round(cx), round(cy)) for cx, cy in centres]
    if classes is None:
        target = gaussian_target((cells, cells), snapped)
    else:
        # one channel per class; an unlabelled centre trains channel 0, which is what "an object,
        # class unknown" means when the model is multi-class. that keeps a partly-clustered set
        # trainable instead of demanding every box be named first
        indices = [classes.get(label, 0) if label else 0 for _, label in kept]
        target = class_target((cells, cells), snapped, indices, len(classes) or 1)

    mask = np.ones((cells, cells), dtype=np.float32)
    for x0, y0, x1, y1 in example.ignore:
        a = int((x0 / DOWNSCALE - left) / STRIDE)
        b = int((y0 / DOWNSCALE - top) / STRIDE)
        c = int(math.ceil((x1 / DOWNSCALE - left) / STRIDE))
        d = int(math.ceil((y1 / DOWNSCALE - top) / STRIDE))
        a, b = max(0, a), max(0, b)
        c, d = min(cells, c), min(cells, d)
        if a < c and b < d:
            mask[b:d, a:c] = 0.0
    # an ignore region must never blank out a confirmed object sitting inside it. the mask is
    # per-CELL while a multi-class target is per-cell-per-class, so collapse across channels: a
    # cell holding any object of any class is a cell the loss must still see
    hot = target if target.ndim == 2 else target.max(axis=0)
    mask[hot > 0.3] = 1.0
    return window, target, mask


def train(
    examples: list[Example],
    *,
    epochs: int = 30,
    batch: int = 8,
    learning_rate: float = 3e-4,
    seed: int = 0,
    device: str | None = None,
    on_progress=None,
    classes: dict[str, int] | None = None,
    crop: int | None = None,
):
    """returns (model, history). on_progress is called once per epoch with a Progress.

    classes maps a cluster label to its channel. Given one, the model grows a channel per class and
    each object trains only its own - which is what makes a rare class survive a common one, and
    what a single softmax over the same labels would not do. Left None, this behaves exactly as it
    did: one channel, every object a positive, and every existing checkpoint still loads.
    """
    # the window every training sample is cut at. below minimum_window() for this set's boxes
    # an object is clipped at the jitter extremes, so a caller taking this from a human checks first
    window = snapped_window(CROP if crop is None else int(crop))
    torch = _torch()
    usable = [e for e in examples if e.object_count]
    if not usable:
        raise TrainError("no example has a confirmed object - nothing to learn from")

    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    rng = random.Random(seed)
    torch.manual_seed(seed)

    model = build_model(classes=len(classes) if classes else 1).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
    cache = {e.path: _load_input(e.path) for e in usable}

    history = []
    seen = 0
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for _ in range(max(1, len(usable) // batch)):
            windows, targets, masks = [], [], []
            for _ in range(batch):
                example = rng.choice(usable)
                w, t, m = _crop_window(example, cache[example.path], rng, classes, window)
                windows.append(w)
                targets.append(t)
                masks.append(m)
            # ascontiguousarray because each window is a SLICE of a cached frame, and a
            # non-contiguous tensor blows up in backward with a .view() error
            x = torch.from_numpy(np.ascontiguousarray(np.stack(windows))).to(device)
            stacked = np.ascontiguousarray(np.stack(targets))
            y = torch.from_numpy(stacked).to(device)
            if y.dim() == 3:  # single channel targets arrive as (batch, h, w)
                y = y.unsqueeze(1)
            mask = torch.from_numpy(np.ascontiguousarray(np.stack(masks))).unsqueeze(1).to(device)

            loss = masked_focal_loss(model(x), y, mask)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            total += float(loss.detach().cpu())
            seen += batch

        mean = total / max(1, len(usable) // batch)
        history.append(mean)
        if on_progress:
            on_progress(Progress(epoch=epoch, epochs=epochs, loss=mean, seen=seen))
    return model, history


def heatmap_for(model, path: Path, device: str | None = None) -> np.ndarray:
    """run the net over a whole capture and return its raw heatmap"""
    torch = _torch()
    if device is None:
        device = next(model.parameters()).device
    image = _load_input(path)
    with torch.no_grad():
        model.eval()
        x = torch.from_numpy(image).unsqueeze(0).to(device)
        return torch.sigmoid(model(x))[0, 0].cpu().numpy()


def heatmaps_for(model, path: Path, device: str | None = None) -> np.ndarray:
    """EVERY channel for one capture, as (classes, h, w).

    heatmap_for returns channel 0 only, which is all a single-class model has. a sweep needs them
    all: the channel a peak appears in IS its predicted class, so collapsing them here would throw
    away the label the whole multi-class head exists to produce.
    """
    torch = _torch()
    if device is None:
        device = next(model.parameters()).device
    image = _load_input(path)
    with torch.no_grad():
        model.eval()
        x = torch.from_numpy(image).unsqueeze(0).to(device)
        return torch.sigmoid(model(x))[0].cpu().numpy()


# how finely a score distribution is bucketed, 0..1. one bar per bucket under the threshold
# slider, so this is what decides how much shape the chart can show
SCORE_BUCKETS = 40


def _heatmaps_of(model, image: np.ndarray, device) -> np.ndarray:
    """heatmaps for an ALREADY DECODED input, so a caller can decode off the model's thread"""
    torch = _torch()
    with torch.no_grad():
        model.eval()
        x = torch.from_numpy(image).unsqueeze(0).to(device)
        return torch.sigmoid(model(x))[0].cpu().numpy()


def sweep(
    model,
    classes: dict[str, int],
    frames: list[Path],
    *,
    width: int,
    height: int,
    min_score: float = 0.5,
    max_per_frame: int = 12,
    device: str | None = None,
    on_progress=None,
) -> list[dict]:
    """run a trained model over whole frames and return CANDIDATES, in the schema the review tool
    already reads - so a sweep's output opens in the discard/promote tab exactly like drawn boxes.

    THIS IS THE LOOP'S RETURN EDGE. drawing seeds a model; the model proposes on frames nobody has
    drawn on; those proposals are judged and promoted; the next model is better. each candidate
    carries the class its channel names and the peak height as its score, so the judging is a
    confirm rather than a fresh labelling.
    """
    by_channel = {index: label for label, index in classes.items()}
    # DECODE AHEAD OF THE MODEL. measured per frame at 3420x2224: 51 ms to decode and downscale the
    # jpeg against 39 ms for the forward pass, so more than half the sweep was the cpu waiting on
    # pillow with the gpu idle. two workers keep one frame ready while the current one runs; the
    # window is bounded so a long sweep never holds every decoded frame in memory at once.
    from concurrent.futures import ThreadPoolExecutor

    if device is None:
        device = next(model.parameters()).device
    pool = ThreadPoolExecutor(max_workers=2)
    pending = {i: pool.submit(_load_input, f) for i, f in enumerate(frames[:3])}
    try:
        out, highest, histogram = _sweep_frames(
            model,
            frames,
            pending,
            pool,
            device,
            by_channel,
            width=width,
            height=height,
            min_score=min_score,
            max_per_frame=max_per_frame,
            on_progress=on_progress,
        )
    finally:
        pool.shutdown(wait=False)
    return out


def _sweep_frames(
    model,
    frames,
    pending,
    pool,
    device,
    by_channel,
    *,
    width,
    height,
    min_score,
    max_per_frame,
    on_progress,
):
    out: list[dict] = []
    highest = 0.0
    # WHAT THE MODEL ANSWERED, over the frames actually swept - 2.5% buckets across 0..1. free,
    # because the heatmaps are already computed here, and it is the honest thing to choose a
    # threshold against: the response on THIS recording rather than on the training frames
    histogram = np.zeros(SCORE_BUCKETS, dtype=np.int64)
    for position, path in enumerate(frames, start=1):
        index = position - 1
        ahead = index + 3
        if ahead < len(frames):
            pending[ahead] = pool.submit(_load_input, frames[ahead])
        maps = _heatmaps_of(model, pending.pop(index).result(), device)
        highest = max(highest, float(maps.max()))
        counts, _ = np.histogram(maps.max(axis=0), bins=SCORE_BUCKETS, range=(0.0, 1.0))
        histogram += counts
        # STRONGEST FIRST, THEN CAPPED PER FRAME. an undertrained model answers warm nearly
        # everywhere - measured: 6 epochs on 126 boxes produced 2278 peaks on a single frame, which
        # is not a review queue, it is noise. a real frame holds a handful of objects, so keeping
        # the best few is both honest and what makes the output judgeable
        found = [
            (peak, channel)
            for channel in range(maps.shape[0])
            # BOUNDED PER CHANNEL. only the best max_per_frame survive across all channels below,
            # so nothing beyond that many from any single channel can ever be kept - decoding more
            # is work thrown away, and at a low floor it is minutes of it
            for peak in decode_peaks(maps[channel], min_score=min_score, limit=max_per_frame)
        ]
        found.sort(key=lambda pair: pair[0].score, reverse=True)
        for peak, channel in found[:max_per_frame]:
            out.append(
                {
                    "path": path.name,
                    # decode_peaks gives a CENTRE; a candidate box is its top-left corner
                    "left": max(0, peak.x - width // 2),
                    "top": max(0, peak.y - height // 2),
                    "width": width,
                    "height": height,
                    "matched_template": by_channel.get(channel, f"channel {channel}"),
                    "score": round(peak.score, 4),
                }
            )
        if on_progress:
            on_progress(position, len(frames), len(out), highest, histogram)
    return out, highest, histogram


def save(model, path: Path) -> Path:
    torch = _torch()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    return path


def load(path: Path, device: str | None = None, classes: int | None = None):
    """the head is sized FROM THE CHECKPOINT unless classes is given.

    it used to default to classes=1 and simply fail on anything else, with a torch size-mismatch
    error that names tensor shapes rather than the problem. That broke every reload of a real
    model: training holds its model in memory so a run looked fine, and the next restart 500'd the
    overlay with "copying a param with shape [15, 48, 1, 1], the shape in current model is
    [1, 48, 1, 1]". A checkpoint already states how many channels it has - the final conv's output
    dimension - so ask it rather than making the caller remember.
    """
    torch = _torch()
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    state = torch.load(path, map_location=device)
    if classes is None:
        # the last conv's weight is (classes, channels, 1, 1)
        heads = [v for k, v in state.items() if k.endswith(".weight") and v.ndim == 4]
        classes = int(heads[-1].shape[0]) if heads else 1
    model = build_model(classes=classes).to(device)
    model.load_state_dict(state)
    model.eval()
    return model
