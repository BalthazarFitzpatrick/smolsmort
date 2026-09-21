"""training loop and sweep for the box cnn - the size-aware sibling of detect/train.py.

THE SAME STRATEGY AS detect, deliberately: balanced windows cut from whole frames, most centred on a
real object, some on a hard negative, the rest random; ignore regions zeroed out of the loss so an
object nobody ruled on is never taught as background. What differs is only what the object IS here -
a box with its own size - and that every frame is first scaled to one working long side, so frames
captured at different resolutions train as one set.
"""

from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np

from smolsmort.boxes.model import (
    STRIDE,
    WORK_LONG_SIDE,
    box_loss,
    box_target,
    build_model,
    classes_in,
    decode_boxes,
    head_in,
    padded,
    receptive_field,
    scale_for,
    widths_in,
)
from smolsmort.detect.dataset import Example
from smolsmort.detect.model import _torch
from smolsmort.detect.train import Progress
from smolsmort.optim import build_optimizer

CROP = 256  # training window side, input px; a multiple of the net's 32
# how far a window may sit off the object it is centred on, as a fraction of the window
JITTER_FRACTION = 0.35
# the largest object must fit inside this share of the receptive field: a cell sizes what it sees,
# and at the very edge of its view it sees too little of the object to size it
REACH_MARGIN = 1.5


class BoxTrainError(Exception):
    pass


def load_input(path: Path, long_side: int = WORK_LONG_SIDE) -> tuple[np.ndarray, float]:
    """a frame as the net sees it - scaled to the working long side, rgb, channels-first, 0..1 -
    and the scale that maps capture px to input px"""
    from PIL import Image

    with Image.open(path) as handle:
        image = handle.convert("RGB")
        scale = scale_for(image.width, image.height, long_side)
        size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
        image = image.resize(size, Image.BILINEAR)
    return np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0, scale


def _sizes_of(example: Example) -> list[tuple[float, float]]:
    if len(example.sizes) != len(example.centres):
        raise BoxTrainError(
            f"{example.path.name} has {len(example.centres)} objects but {len(example.sizes)} sizes - "
            "the box backend learns each object's width and height, so every centre needs one"
        )
    return example.sizes


def _window(example: Example, image: np.ndarray, scale: float, rng: random.Random, classes, size):
    """one training window and its target, with the object anywhere in it rather than central"""
    _, height, width = image.shape
    side = min(size, height - height % 32, width - width % 32)
    roll = rng.random()
    if example.centres and roll < 0.6:
        cx, cy = rng.choice(example.centres)
        jitter = side * JITTER_FRACTION
        left = int(cx * scale - side / 2 + rng.uniform(-jitter, jitter))
        top = int(cy * scale - side / 2 + rng.uniform(-jitter, jitter))
    elif example.negatives and roll < 0.8:
        # a place the model fired and was told no - a random window almost never holds one
        cx, cy = rng.choice(example.negatives)
        left = int(cx * scale - side / 2 + rng.uniform(-side / 6, side / 6))
        top = int(cy * scale - side / 2 + rng.uniform(-side / 6, side / 6))
    else:
        left, top = rng.randrange(0, max(1, width - side)), rng.randrange(0, max(1, height - side))
    left, top = max(0, min(left, width - side)), max(0, min(top, height - side))

    boxes = []
    for (cx, cy), (box_w, box_h) in zip(example.centres, _sizes_of(example), strict=True):
        x, y, w, h = cx * scale - left, cy * scale - top, box_w * scale, box_h * scale
        boxes.append((x - w / 2, y - h / 2, x + w / 2, y + h / 2))
    labels = list(example.labels) + [None] * (len(example.centres) - len(example.labels))
    # an unlabelled object trains channel 0, what "an object, class unknown" means - as in detect
    channels = [classes.get(label, 0) if (classes and label) else 0 for label in labels]
    cells = side // STRIDE
    target = box_target((cells, cells), boxes, channels, len(classes) if classes else 1)

    for x0, y0, x1, y1 in example.ignore:
        a, b = int((x0 * scale - left) / STRIDE), int((y0 * scale - top) / STRIDE)
        c = math.ceil((x1 * scale - left) / STRIDE)
        d = math.ceil((y1 * scale - top) / STRIDE)
        a, b, c, d = max(0, a), max(0, b), min(cells, c), min(cells, d)
        if a < c and b < d:
            target.mask[b:d, a:c] = 0.0
    # an ignore region must never blank out a confirmed object inside it
    target.mask[target.heat.max(axis=0) > 0.3] = 1.0
    return image[:, top : top + side, left : left + side], target


def train(
    examples: list[Example],
    *,
    epochs: int = 30,
    batch: int = 8,
    learning_rate: float = 2e-3,
    seed: int = 0,
    device: str | None = None,
    on_progress=None,
    classes: dict[str, int] | None = None,
    crop: int = CROP,
    long_side: int = WORK_LONG_SIDE,
    steps_per_epoch: int | None = None,
    widths: tuple[int, int, int, int, int] | None = None,
    optimizer: str = "adamw",
    momentum: float = 0.9,
    weight_decay: float = 1e-4,
):
    """returns (model, history). on_progress is called once per epoch with a detect Progress.

    an epoch is one pass over the confirmed OBJECTS in batches, not over frames - a set of a few
    crowded frames and one of many sparse ones then train for comparable time

    optimizer/momentum/weight_decay go through smolsmort.optim.build_optimizer - the old hardcoded
    AdamW(wd=1e-4) is exactly "adamw" at momentum 0.9 (AdamW's own beta1 default) and weight_decay
    1e-4, so every default and existing checkpoint is unaffected. widths is build_model's own
    per-stage channel counts (16,32,64,96,128 by default); a smaller/larger tuple trades capacity
    for parameter count the same way `channels` does on the heatmap backend.
    """
    torch = _torch()
    usable = [e for e in examples if e.object_count or e.exhaustive]
    if not any(e.object_count for e in usable):
        raise BoxTrainError("no example has a confirmed object - nothing to learn from")
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    rng = random.Random(seed)
    torch.manual_seed(seed)

    model_kwargs = {"classes": len(classes) if classes else 1}
    if widths is not None:
        model_kwargs["widths"] = widths
    model = build_model(**model_kwargs)
    cache = {e.path: load_input(e.path, long_side) for e in usable}
    _refuse_out_of_reach(usable, cache, receptive_field(model))
    model = model.to(device)
    optimiser = build_optimizer(
        model.parameters(),
        optimizer=optimizer,
        learning_rate=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
    )
    objects = sum(e.object_count for e in usable)
    steps = steps_per_epoch or max(1, math.ceil(objects / batch))

    history = []
    seen = 0
    for epoch in range(1, epochs + 1):
        # a single step down late in training: cheap, and it settles the size head noticeably
        for group in optimiser.param_groups:
            group["lr"] = learning_rate if epoch <= 0.75 * epochs else learning_rate / 5
        model.train()
        total = 0.0
        for _ in range(steps):
            windows, targets = [], []
            for _ in range(batch):
                example = rng.choice(usable)
                image, scale = cache[example.path]
                window, target = _window(example, image, scale, rng, classes, crop)
                windows.append(window)
                targets.append(target)
            x = torch.from_numpy(np.ascontiguousarray(np.stack(windows))).to(device)

            def stacked(name, targets=targets):
                parts = np.ascontiguousarray(np.stack([getattr(t, name) for t in targets]))
                return torch.from_numpy(parts).to(device)

            loss, _ = box_loss(
                model(x),
                stacked("heat"),
                stacked("size"),
                stacked("offset"),
                stacked("centre"),
                stacked("mask"),
            )
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            total += float(loss.detach().cpu())
            seen += batch
        history.append(total / steps)
        if on_progress:
            on_progress(Progress(epoch=epoch, epochs=epochs, loss=history[-1], seen=seen))
    return model, history


def evaluate(
    model, examples: list[Example], *, classes=None, long_side: int = WORK_LONG_SIDE
) -> float:
    """mean box_loss of model over examples: eval mode, no gradient, and every example cut with
    its own fixed-seed rng so two calls agree"""
    torch = _torch()
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    losses = []
    with torch.no_grad():
        for example in examples:
            image, scale = load_input(example.path, long_side)
            window, target = _window(example, image, scale, random.Random(0), classes, CROP)

            def tensor(array):
                return torch.from_numpy(np.ascontiguousarray(array[None])).to(device)

            x = tensor(window)
            heat, size, offset, centre, mask = (
                tensor(getattr(target, name))
                for name in ("heat", "size", "offset", "centre", "mask")
            )
            loss, _ = box_loss(model(x), heat, size, offset, centre, mask)
            losses.append(float(loss.cpu()))
    model.train(was_training)
    return sum(losses) / len(losses) if losses else 0.0


def _refuse_out_of_reach(usable: list[Example], cache: dict, reach: int) -> None:
    """an object bigger than what one cell can see cannot be sized, so say so before training.

    this fails loudly on purpose: the alternative is a loss that plateaus and boxes that come out
    too small on exactly the largest objects, with nothing naming the cause.
    """
    limit = reach / REACH_MARGIN
    for example in usable:
        _, scale = cache[example.path]
        for box_w, box_h in _sizes_of(example):
            longest = max(box_w, box_h) * scale
            if longest > limit:
                raise BoxTrainError(
                    f"{example.path.name} holds an object {longest:.0f} input px long, but the net "
                    f"sizes objects up to {limit:.0f} (receptive field {reach} / {REACH_MARGIN}). "
                    "pass a smaller long_side so every frame is scaled down further"
                )


def _infer(model, image: np.ndarray, scale: float, device, min_score: float, limit: int):
    torch = _torch()
    with torch.no_grad():
        model.eval()
        x = torch.from_numpy(padded(image))[None].to(device)
        outputs = [o.cpu() for o in model(x)]
    return decode_boxes(outputs, scale, min_score=min_score, limit=limit)


def sweep(
    model,
    classes: dict[str, int],
    frames: list[Path],
    *,
    min_score: float = 0.3,
    max_per_frame: int = 50,
    long_side: int = WORK_LONG_SIDE,
    device: str | None = None,
    on_progress=None,
) -> list[dict]:
    """run trained weights over whole frames and return CANDIDATES in the schema the review tool
    reads - each with its OWN width and height, the one thing detect's sweep cannot give.

    frames are decoded one ahead on a worker thread, as in detect's sweep, so the model is not left
    waiting on pillow
    """
    from concurrent.futures import ThreadPoolExecutor

    if device is None:
        device = next(model.parameters()).device
    by_channel = {index: label for label, index in classes.items()}
    out: list[dict] = []
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(load_input, frames[0], long_side) if frames else None
        for position, path in enumerate(frames, start=1):
            image, scale = pending.result()
            if position < len(frames):
                pending = pool.submit(load_input, frames[position], long_side)
            for found in _infer(model, image, scale, device, min_score, max_per_frame):
                out.append(
                    {
                        "path": path.name,
                        "left": max(0, round(found.left)),
                        "top": max(0, round(found.top)),
                        "width": max(1, round(found.width)),
                        "height": max(1, round(found.height)),
                        "matched_template": by_channel.get(
                            found.channel, f"channel {found.channel}"
                        ),
                        "score": round(found.score, 4),
                    }
                )
            if on_progress:
                on_progress(position, len(frames), len(out))
    return out


def save(model, path: Path) -> Path:
    torch = _torch()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    return path


def load(path: Path, device: str | None = None):
    """class count, stage widths and head width are all read off the checkpoint, never assumed"""
    torch = _torch()
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    state = torch.load(path, map_location=device)
    model = build_model(classes=classes_in(state), widths=widths_in(state), head=head_in(state)).to(
        device
    )
    model.load_state_dict(state)
    model.eval()
    return model
