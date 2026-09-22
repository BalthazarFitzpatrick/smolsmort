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
import os
import random
import warnings
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from smolsmort.detect.dataset import Example, frame_width
from smolsmort.detect.model import (
    DEFAULT_CHANNELS,
    DEFAULT_DOWNSCALE,
    LEGACY_DOWNSCALE,
    STRIDE,
    _torch,
    build_model,
    capture_width_of,
    class_target,
    decode_peaks,
    downscale_of,
    gaussian_target,
    set_capture_width,
)
from smolsmort.detect.prefetch import decode_ahead
from smolsmort.detect.windows import carve_ignore, window_origin
from smolsmort.optim import build_optimizer

CROP = 256  # input pixels, i.e. CROP * downscale capture pixels a side (512 at the default 2)

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


def minimum_window(
    box_width: int,
    box_height: int,
    jitter: float = JITTER_FRACTION,
    downscale: int = DEFAULT_DOWNSCALE,
) -> int:
    """the smallest training window that always contains a box of this size, whole.

    A WINDOW, A JITTER AND A BOX SIZE ARE THREE NUMBERS THAT MUST AGREE, and two of them silently
    did not. _crop_window offsets a window by up to `jitter` of its own size, so the box centre can
    sit (0.5 - jitter) of the way from an edge; the box fits only while its half-size is smaller
    than that. At CROP=256 and jitter=0.40 that limit is 25.6 input px, and the retina reference object is 226
    capture px = 28.2 half - so it was CLIPPED BY 2.7px at the extremes, teaching a half-object as a
    whole one, which is exactly what the jitter comment above says the margin was chosen to avoid.
    """
    longest = max(box_width, box_height) / downscale
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


def _load_frame(
    path: Path, downscale: int = DEFAULT_DOWNSCALE, capture_width: int | None = None
) -> tuple[np.ndarray, int]:
    """(the net's input, the frame's own width in px).

    a frame wider or narrower than `capture_width` is first resampled to it, aspect kept, so the
    downscale then lands the object at the size the net was trained on.
    """
    from PIL import Image

    with Image.open(path) as handle:
        return _fit(handle.convert("RGB"), downscale, capture_width)


def _fit(image, downscale: int, capture_width: int | None) -> tuple[np.ndarray, int]:
    """(the net's input, the image's own width) for an rgb pillow image: resampled to the
    capture width when it differs, then downscaled. one body for a file and for an array"""
    from PIL import Image

    original = image.width
    if capture_width and original != capture_width:
        height = max(1, round(image.height * capture_width / original))
        image = image.resize((capture_width, height), Image.BILINEAR)
    image = image.resize((image.width // downscale, image.height // downscale), Image.BILINEAR)
    return np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0, original


def frame_input(model, frame: np.ndarray) -> tuple[np.ndarray, int]:
    """(the net's input, the frame's width) from a DECODED frame - (h, w, 3) rgb uint8 in the
    frame's own px - using the capture width and downscale the weights record. the array-taking
    twin of `_load_frame`, for a caller that already holds the pixels (a live capture) and used
    to mirror the resample and downscale by hand"""
    from PIL import Image

    image = Image.fromarray(np.ascontiguousarray(frame)).convert("RGB")
    return _fit(image, downscale_of(model), capture_width_of(model))


def _device_of(model):
    """where a model or runner takes its input: a runner says so, a module has parameters"""
    found = getattr(model, "device", None)
    return found if found is not None else next(model.parameters()).device


def heatmaps_for_frame(model, frame: np.ndarray, device=None) -> tuple[np.ndarray, float]:
    """(per-class heatmaps, frame px per capture px) for one decoded frame. peaks decoded from
    the heatmaps with `decode_peaks(..., downscale=downscale_of(model))` are in capture px;
    multiply by the ratio to land on the frame"""
    if device is None:
        device = _device_of(model)
    image, original = frame_input(model, frame)
    return heatmaps_of(model, image, device), frame_ratio(model, original)


def predict_frame(
    model,
    frame: np.ndarray,
    *,
    classes: dict[str, int],
    width: int,
    height: int,
    min_score: float = 0.5,
    max_per_frame: int = 12,
    device=None,
) -> list[dict]:
    """candidates for ONE decoded frame, in the frame's own px and in the schema `sweep` writes
    (minus `path`, which an array does not have). the same steps a sweep takes per file, so a
    caller holding a live capture gets exactly what the review tool would propose for it"""
    if device is None:
        device = _device_of(model)
    image, original = frame_input(model, frame)
    maps = heatmaps_of(model, image, device)
    by_channel = {index: label for label, index in classes.items()}
    found = _decode_frame(
        maps,
        frame_ratio(model, original),
        by_channel,
        downscale_of(model),
        width=width,
        height=height,
        min_score=min_score,
        max_per_frame=max_per_frame,
    )
    capture = capture_width_of(model)
    if capture and original != capture:
        for candidate in found:
            candidate["resampled_from"] = original
    return found


def _load_input(
    path: Path, downscale: int = DEFAULT_DOWNSCALE, capture_width: int | None = None
) -> np.ndarray:
    """a capture as the net sees it: downscaled, rgb, channels-first, 0..1"""
    return _load_frame(path, downscale, capture_width)[0]


def frame_ratio(model, original_width: int) -> float:
    """frame px per capture px: what a decoded position is multiplied by to land on the original
    frame. 1.0 when the capture is unknown or already the frame's width."""
    capture = capture_width_of(model)
    return original_width / capture if capture else 1.0


def scale_example(example: Example, factor: float) -> Example:
    """an example with every coordinate multiplied by factor - a frame resampled by that much"""

    def point(p):
        return (p[0] * factor, p[1] * factor)

    return replace(
        example,
        centres=[point(c) for c in example.centres],
        sizes=[point(s) for s in example.sizes],
        negatives=[point(n) for n in example.negatives],
        ignore=[
            (r[0] * factor, r[1] * factor, r[2] * factor, r[3] * factor) for r in example.ignore
        ],
    )


def capture_examples(examples: list[Example], capture_width: int | None) -> list[Example]:
    """examples expressed in capture pixels: each frame not already that wide is scaled to it"""
    if not capture_width:
        return list(examples)
    out = []
    for example in examples:
        width = frame_width(example.path)
        out.append(
            example if width == capture_width else scale_example(example, capture_width / width)
        )
    return out


def _crop_window(
    example: Example,
    image: np.ndarray,
    rng: random.Random,
    classes=None,
    window: int = CROP,
    downscale: int = DEFAULT_DOWNSCALE,
):
    """one training window, half the time centred on a real object.

    returns the window, its heatmap target and the ignore mask, all already at their own scales:
    the target and mask are at STRIDE cells, the window at input pixels.
    """
    _, height, width = image.shape
    size = min(window, height, width)
    # THE OBJECT MUST LAND ANYWHERE IN THE WINDOW, NOT NEAR THE MIDDLE. this used to jitter by
    # +/- size/6, which on a 256px window pins the object within ~42px of the centre - so the net
    # was taught "objects are near the middle" and then asked to sweep whole frames, where they
    # are anywhere. Balthazar Fitzpatrick spotted the same gap from the data side, asking whether the object
    # could be off-centre in the training crops. 0.40 keeps a margin so the object stays fully
    # inside rather than clipped at the edge, which would teach a half-object as a whole one.
    # HARD NEGATIVES (the second roll) are places the model ITSELF fired on and a different signal
    # says are not objects; a uniformly random window almost never contains one, so without
    # deliberately sampling them the model never revisits its own mistakes.
    left, top = window_origin(
        rng,
        size,
        width,
        height,
        centres=[(cx / downscale, cy / downscale) for cx, cy in example.centres],
        negatives=[(cx / downscale, cy / downscale) for cx, cy in example.negatives],
        jitter_fraction=JITTER_FRACTION,
        object_share=0.5,
        negative_share=0.75,
    )

    window = image[:, top : top + size, left : left + size]
    cells = size // STRIDE
    # labels ride along with their centre so a window keeps them aligned after clipping. an
    # example built before labels existed has none, so pad rather than zip-strict
    example_labels = list(example.labels) + [None] * (len(example.centres) - len(example.labels))
    kept = [
        (((cx / downscale - left) / STRIDE, (cy / downscale - top) / STRIDE), label)
        for (cx, cy), label in zip(example.centres, example_labels, strict=True)
        if 0 <= cx / downscale - left < size and 0 <= cy / downscale - top < size
    ]
    centres = [c for c, _ in kept]
    # SNAP TO CELL CENTRES. gaussian_target builds its blob from the coordinate given, so a
    # fractional centre peaks below 1.0 - measured: 0.9314 for a centre at (8.4, 8.6). The loss
    # counts positives with target >= 0.99, so fractional centres produced ZERO positive cells,
    # only the negative term trained, and the net learned to answer a flat 0.145 everywhere.
    # Rounding costs at most half a cell of localisation (2 input pixels at STRIDE 4)
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
    carve_ignore(mask, example.ignore, left, top, STRIDE, lambda v: v / downscale)
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
    downscale: int = DEFAULT_DOWNSCALE,
    channels: int = DEFAULT_CHANNELS,
    optimizer: str = "adamw",
    momentum: float = 0.9,
    weight_decay: float = 0.0,
    capture_width: int | None = None,
    init_model=None,
):
    """returns (model, history). on_progress is called once per epoch with a Progress.

    classes maps a cluster label to its channel. Given one, the model grows a channel per class and
    each object trains only its own - which is what makes a rare class survive a common one, and
    what a single softmax over the same labels would not do. Left None, this behaves exactly as it
    did: one channel, every object a positive, and every existing checkpoint still loads.

    optimizer/momentum/weight_decay go through smolsmort.optim.build_optimizer - "adamw" at
    momentum 0.9 and weight_decay 0 behaves exactly as the old hardcoded Adam(lr) did, so every
    default and existing checkpoint is unaffected.

    `downscale` is the capture:input factor the model is trained at. it is stored in the model, so
    save() keeps it and load() restores it - callers decoding or sweeping read it off the model.
    `channels` is the network's base width; load() reads it back from the weights, so a wider net
    trains and reloads with no other change.

    `capture_width` (px) is the frame width the model is trained at and records. None reads it off
    the examples' frames (recorded only when they all share one width); a given width different
    from a frame's resamples that frame to it, aspect kept, BEFORE the downscale.

    `init_model` continues training that model. its downscale is architecture, so a different
    `downscale` is refused; its recorded capture width wins over `capture_width`, with a warning.
    """
    # the window every training sample is cut at. below minimum_window() for this set's boxes
    # an object is clipped at the jitter extremes, so a caller taking this from a human checks first
    window = snapped_window(CROP if crop is None else int(crop))
    torch = _torch()
    # an exhaustive frame with zero objects is a confirmed-empty frame, not an unknown one - the
    # cleanest negative there is, so it stays even though object_count is zero
    usable = [e for e in examples if e.object_count or e.exhaustive]
    if not any(e.object_count for e in usable):
        raise TrainError("no example has a confirmed object - nothing to learn from")

    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    rng = random.Random(seed)
    torch.manual_seed(seed)

    if init_model is not None and downscale_of(init_model) != int(downscale):
        raise TrainError(
            f"cannot fine-tune: the weights were trained at downscale {downscale_of(init_model)} "
            f"but this run asks for {int(downscale)}; the architecture depends on it"
        )
    widths = {frame_width(e.path) for e in usable}
    kept_capture = capture_width_of(init_model) if init_model is not None else None
    if kept_capture:
        if capture_width and int(capture_width) != kept_capture:
            warnings.warn(
                f"fine-tuning keeps the weights' capture width {kept_capture}, not {capture_width}",
                stacklevel=2,
            )
        capture = kept_capture
    elif capture_width:
        capture = int(capture_width)
    else:
        capture = widths.pop() if len(widths) == 1 else None
    usable = capture_examples(usable, capture)

    if init_model is not None:
        model = init_model.to(device)
    else:
        model = build_model(
            classes=len(classes) if classes else 1, downscale=downscale, channels=channels
        ).to(device)
    set_capture_width(model, capture)
    optimiser = build_optimizer(
        model.parameters(),
        optimizer=optimizer,
        learning_rate=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
    )
    cache = {e.path: _load_input(e.path, downscale, capture) for e in usable}

    history = []
    seen = 0
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for _ in range(max(1, len(usable) // batch)):
            windows, targets, masks = [], [], []
            for _ in range(batch):
                example = rng.choice(usable)
                w, t, m = _crop_window(
                    example, cache[example.path], rng, classes, window, downscale
                )
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


def evaluate(model, examples: list[Example], *, classes=None, window: int | None = None) -> float:
    """mean masked focal loss of model over examples: eval mode, no gradient, and every example
    cut with its own fixed-seed rng so two calls agree"""
    torch = _torch()
    size = snapped_window(CROP if window is None else int(window))
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    losses = []
    capture = capture_width_of(model)
    factor = downscale_of(model)
    with torch.no_grad():
        for example in capture_examples(examples, capture):
            image = _load_input(example.path, factor, capture)
            w, t, m = _crop_window(example, image, random.Random(0), classes, size, factor)
            x = torch.from_numpy(np.ascontiguousarray(w[None])).to(device)
            y = torch.from_numpy(np.ascontiguousarray(t[None])).to(device)
            if y.dim() == 3:
                y = y.unsqueeze(1)
            mask = torch.from_numpy(np.ascontiguousarray(m[None])).unsqueeze(1).to(device)
            losses.append(float(masked_focal_loss(model(x), y, mask).cpu()))
    model.train(was_training)
    return sum(losses) / len(losses) if losses else 0.0


def heatmap_for(model, path: Path, device: str | None = None) -> np.ndarray:
    """run the net over a whole capture and return its raw heatmap"""
    torch = _torch()
    if device is None:
        device = _device_of(model)
    image = _load_input(path, downscale_of(model), capture_width_of(model))
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
        device = _device_of(model)
    image = _load_input(path, downscale_of(model), capture_width_of(model))
    with torch.no_grad():
        model.eval()
        x = torch.from_numpy(image).unsqueeze(0).to(device)
        return torch.sigmoid(model(x))[0].cpu().numpy()


# how finely a score distribution is bucketed, 0..1. one bar per bucket under the threshold
# slider, so this is what decides how much shape the chart can show
SCORE_BUCKETS = 40


def heatmaps_of(model, image: np.ndarray, device=None) -> np.ndarray:
    """(classes, h / stride, w / stride) sigmoid heatmaps for an ALREADY PREPARED (3, h, w) float32
    input - what `frame_input` returns, after whatever masking the caller does - so a caller can
    decode off the model's thread. the one seam every runtime answers: `model` may be the torch
    module or a runner from `load(..., runtime=...)`"""
    torch = _torch()
    if device is None:
        device = _device_of(model)
    with torch.no_grad():
        model.eval()
        x = torch.from_numpy(np.ascontiguousarray(image, dtype=np.float32)).unsqueeze(0).to(device)
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

    width and height are the box in CAPTURE px (the model's recorded capture width). a frame of a
    different width is resampled to it, and every candidate is mapped back to that frame's own
    pixels; those from a resampled frame carry `resampled_from`, the frame's width.
    """
    by_channel = {index: label for label, index in classes.items()}
    if device is None:
        device = _device_of(model)
    factor = downscale_of(model)
    capture = capture_width_of(model)
    # decoded ahead of the model on two workers, see detect/prefetch.py for the measurement
    decoded = decode_ahead(frames, lambda f: _load_frame(f, factor, capture), ahead=3, workers=2)
    out, highest, histogram = _sweep_frames(
        model,
        decoded,
        len(frames),
        device,
        by_channel,
        factor,
        capture,
        width=width,
        height=height,
        min_score=min_score,
        max_per_frame=max_per_frame,
        on_progress=on_progress,
    )
    return out


def _sweep_frames(
    model,
    decoded,
    total,
    device,
    by_channel,
    downscale,
    capture,
    *,
    width,
    height,
    min_score,
    max_per_frame,
    on_progress,
):
    """`decoded` yields (path, (input, original width)) in order; `total` is how many"""
    out: list[dict] = []
    highest = 0.0
    # WHAT THE MODEL ANSWERED, over the frames actually swept - 2.5% buckets across 0..1. free,
    # because the heatmaps are already computed here, and it is the honest thing to choose a
    # threshold against: the response on THIS recording rather than on the training frames
    histogram = np.zeros(SCORE_BUCKETS, dtype=np.int64)
    for position, (path, (image, original)) in enumerate(decoded, start=1):
        maps = heatmaps_of(model, image, device)
        resampled = bool(capture) and original != capture
        highest = max(highest, float(maps.max()))
        counts, _ = np.histogram(maps.max(axis=0), bins=SCORE_BUCKETS, range=(0.0, 1.0))
        histogram += counts
        found = _decode_frame(
            maps,
            frame_ratio(model, original),
            by_channel,
            downscale,
            width=width,
            height=height,
            min_score=min_score,
            max_per_frame=max_per_frame,
        )
        for candidate in found:
            candidate["path"] = path.name
            if resampled:
                candidate["resampled_from"] = original
            out.append(candidate)
        if on_progress:
            on_progress(position, total, len(out), highest, histogram)
    return out, highest, histogram


def _decode_frame(
    maps: np.ndarray,
    ratio: float,
    by_channel: dict[int, str],
    downscale: int,
    *,
    width: int,
    height: int,
    min_score: float,
    max_per_frame: int,
) -> list[dict]:
    """candidates in frame px from one frame's heatmaps. `ratio` is frame px per capture px;
    boxes and peaks are decoded in capture px and mapped back"""
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
        for peak in decode_peaks(
            maps[channel], min_score=min_score, limit=max_per_frame, downscale=downscale
        )
    ]
    found.sort(key=lambda pair: pair[0].score, reverse=True)
    box_w, box_h = round(width * ratio), round(height * ratio)
    return [
        {
            # decode_peaks gives a CENTRE; a candidate box is its top-left corner
            "left": max(0, round(peak.x * ratio) - box_w // 2),
            "top": max(0, round(peak.y * ratio) - box_h // 2),
            "width": box_w,
            "height": box_h,
            "matched_template": by_channel.get(channel, f"channel {channel}"),
            "score": round(peak.score, 4),
        }
        for peak, channel in found[:max_per_frame]
    ]


def save(model, path: Path) -> Path:
    torch = _torch()
    path.parent.mkdir(parents=True, exist_ok=True)
    # temp beside it, then replace: a crash mid-write never leaves a checkpoint torch.load
    # refuses, and a lister never sees a half-written .pt
    temp = path.with_name(path.name + ".tmp")
    torch.save(model.state_dict(), temp)
    os.replace(temp, path)
    return path


def load(path: Path, device: str | None = None, classes: int | None = None, runtime: str = "torch"):
    """the head is sized FROM THE CHECKPOINT unless classes is given.

    `runtime` picks where the model runs: "torch" (this module, on `device`) or "coreml" (a
    runner over a core ml package beside the checkpoint, see smolsmort.detect.runtime). a runner
    answers the same call as the module and carries the same `downscale` and `capture_width`, so
    everything above this function is indifferent to the choice.

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
    # a checkpoint older than the stored factor was trained at the one that was hard-coded then
    state.setdefault("downscale", torch.tensor(LEGACY_DOWNSCALE))
    state.setdefault("capture_width", torch.tensor(0))  # 0: capture never recorded
    convs = [v for k, v in state.items() if k.endswith(".weight") and v.ndim == 4]
    if classes is None:
        # the last conv's weight is (classes, channels, 1, 1)
        classes = int(convs[-1].shape[0]) if convs else 1
    # the first conv's weight is (channels, 3, 3, 3): the width the net was trained at
    channels = int(convs[0].shape[0]) if convs else DEFAULT_CHANNELS
    model = build_model(
        classes=classes,
        downscale=int(state["downscale"]),
        channels=channels,
        capture_width=int(state["capture_width"]),
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    if runtime != "torch":
        from smolsmort.detect.runtime import build

        return build(runtime, model.to("cpu"), Path(path))
    return model
