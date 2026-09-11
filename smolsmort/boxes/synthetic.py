"""synthetic fixed-camera frames with known boxes, for proving the box backend without real data.

WHAT IT IMITATES, and why each part is there. A STATIC background per frame size, because a fixed
camera is the case this backend is a starter for, and a moving background would test something
else. Elongated rotated ellipses, because an enclosing box around a rotated elongated object is the
case a centre-only model cannot express. Classes told apart by PATTERN with a random hue, so colour
alone cannot answer the class. About a third of objects in overlapping pairs, the later one drawn on
top, because a decoder that suppresses near neighbours fails exactly there.

NOT A BENCHMARK. It exists to prove the mechanics - that sizes are learned across a wide range, that
overlaps survive decoding, that two input resolutions train as one set. A number measured here says
nothing about real footage.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image

FRAME_SIZES = ((640, 480), (1280, 720))
# the major axis range in frame px - 6.7x, past the 4x this backend is asked to handle
MIN_LENGTH, MAX_LENGTH = 24, 160
CLASSES = 3
HUES = ((205, 185, 70), (95, 165, 215), (215, 95, 125), (140, 200, 120))


def background(width: int, height: int, seed: int) -> np.ndarray:
    """a smooth mottled field with horizontal bands, identical every time for one seed"""
    rng = np.random.default_rng(seed)
    coarse = rng.random((height // 32 + 2, width // 32 + 2, 3))
    smooth = Image.fromarray((coarse * 255).astype(np.uint8)).resize((width, height), Image.BICUBIC)
    base = np.asarray(smooth, dtype=np.float32) * 0.45 + 55
    bands = (np.sin(np.arange(height) / 23.0) * 14)[:, None, None]
    return np.clip(base + bands, 0, 255).astype(np.float32)


def extent(a: float, b: float, theta: float) -> tuple[float, float]:
    """half-width and half-height of the box enclosing an ellipse with semi-axes a, b turned by theta"""
    half_w = math.sqrt((a * math.cos(theta)) ** 2 + (b * math.sin(theta)) ** 2)
    half_h = math.sqrt((a * math.sin(theta)) ** 2 + (b * math.cos(theta)) ** 2)
    return half_w, half_h


def box_iou(p, q) -> float:
    """iou of two (x0, y0, x1, y1) boxes"""
    ix = max(0.0, min(p[2], q[2]) - max(p[0], q[0]))
    iy = max(0.0, min(p[3], q[3]) - max(p[1], q[1]))
    inter = ix * iy
    union = (p[2] - p[0]) * (p[3] - p[1]) + (q[2] - q[0]) * (q[3] - q[1]) - inter
    return inter / union if union > 0 else 0.0


def _draw(image, cx, cy, a, b, theta, cls, rng) -> None:
    height, width, _ = image.shape
    half_w, half_h = extent(a, b, theta)
    x0, x1 = max(0, int(cx - half_w) - 1), min(width, int(cx + half_w) + 2)
    y0, y1 = max(0, int(cy - half_h) - 1), min(height, int(cy + half_h) + 2)
    ys, xs = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    dx, dy = xs - cx, ys - cy
    along = dx * math.cos(theta) + dy * math.sin(theta)
    across = -dx * math.sin(theta) + dy * math.cos(theta)
    inside = (along / a) ** 2 + (across / b) ** 2 <= 1.0
    hue = np.array(HUES[rng.integers(len(HUES))], dtype=np.float32) + rng.normal(0, 18, 3)
    shade = np.ones_like(along)
    if cls == 1:  # bands across the body
        shade = np.where(np.sin(along / a * 3.0 * math.pi) > 0, 1.0, 0.45)
    elif cls == 2:  # one dark stripe along it
        shade = np.where(np.abs(across / b) < 0.35, 0.4, 1.0)
    colour = hue[None, None, :] * shade[..., None]
    patch = image[y0:y1, x0:x1]
    patch[inside] = colour[inside]


def make_frame(rng: np.random.Generator, backgrounds: dict, overlap_rate: float = 0.3) -> dict:
    """one frame: {image, boxes [(x0, y0, x1, y1)], classes, overlapping} in frame px.

    overlap_rate is the chance each placed object gets an overlapping partner; 0 keeps every object
    standing apart, which isolates the size and resolution questions from the crowding one
    """
    width, height = FRAME_SIZES[rng.integers(len(FRAME_SIZES))]
    image = backgrounds[(width, height)].copy()

    def shape():
        length = math.exp(rng.uniform(math.log(MIN_LENGTH), math.log(MAX_LENGTH)))
        aspect = rng.uniform(1.5, 5.0)
        return length / 2, max(3.0, length / aspect / 2), rng.uniform(-math.pi / 6, math.pi / 6)

    def box_at(cx, cy, a, b, theta):
        half_w, half_h = extent(a, b, theta)
        return (cx - half_w, cy - half_h, cx + half_w, cy + half_h)

    wanted = int(rng.integers(1, 6))
    shapes: list[list] = []
    for _ in range(200):
        if len(shapes) >= wanted:
            break
        a, b, theta = shape()
        half_w, half_h = extent(a, b, theta)
        cx = rng.uniform(half_w + 2, width - half_w - 2)
        cy = rng.uniform(half_h + 2, height - half_h - 2)
        box = box_at(cx, cy, a, b, theta)
        if any(box_iou(box, s[5]) > 0.02 for s in shapes):
            continue
        shapes.append([cx, cy, a, b, theta, box, int(rng.integers(CLASSES)), False])
        if rng.random() >= overlap_rate or len(shapes) >= wanted:
            continue
        # a partner overlapping this one by box iou 0.1-0.5, drawn on top of it
        for _ in range(60):
            a2, b2, t2 = shape()
            angle = rng.uniform(0, 2 * math.pi)
            reach = rng.uniform(0.3, 1.0) * (half_w + half_h) / 2
            cx2, cy2 = cx + math.cos(angle) * reach, cy + math.sin(angle) * reach
            box2 = box_at(cx2, cy2, a2, b2, t2)
            if box2[0] < 0 or box2[1] < 0 or box2[2] > width or box2[3] > height:
                continue
            if not 0.1 <= box_iou(box, box2) <= 0.5:
                continue
            if any(box_iou(box2, s[5]) > 0.02 for s in shapes[:-1]):
                continue
            shapes[-1][7] = True
            shapes.append([cx2, cy2, a2, b2, t2, box2, int(rng.integers(CLASSES)), True])
            break

    for cx, cy, a, b, theta, _, cls, _ in shapes:
        _draw(image, cx, cy, a, b, theta, cls, rng)
    noisy = np.clip(image + rng.normal(0, 5, image.shape), 0, 255).astype(np.uint8)
    return {
        "image": noisy,
        "boxes": [s[5] for s in shapes],
        "classes": [s[6] for s in shapes],
        "overlapping": [s[7] for s in shapes],
    }


def write_recording(
    directory: Path, count: int, seed: int = 0, overlap_rate: float = 0.3
) -> list[dict]:
    """`count` frames as png under `directory`, returned as their truth with a `path` added"""
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    # one background per size for the whole recording: the camera does not move
    backgrounds = {size: background(*size, seed=i) for i, size in enumerate(FRAME_SIZES)}
    frames = []
    for index in range(count):
        frame = make_frame(rng, backgrounds, overlap_rate)
        path = directory / f"{index:07d}.png"
        Image.fromarray(frame.pop("image")).save(path)
        frames.append({**frame, "path": path})
    return frames
