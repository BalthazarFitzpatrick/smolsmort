"""the fixed-size heatmap cnn behind the loop's ModelBackend seam.

the weights carry the box size the set was drawn at, because the model itself predicts only where -
so predict needs nothing but weights, frames and the class map, as the seam promises
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from smolsmort.backends import example_from, sidecar
from smolsmort.detect import train as heatmap_train

NAME = "heatmap"


class HeatmapBackendError(Exception):
    pass


@dataclass
class HeatmapWeights:
    model: object
    classes: dict[str, int]
    box: tuple[int, int] | None


def fitted_box(examples) -> tuple[int, int]:
    """the median drawn size - a hand-drawn box is a few px out each way, and the median is what
    the review tool fits for the same reason"""
    sizes = [size for e in examples for size in e.sizes]
    if not sizes:
        raise HeatmapBackendError("no example carries a box size to fit the fixed size from")
    widths = sorted(int(w) for w, _ in sizes)
    heights = sorted(int(h) for _, h in sizes)
    return widths[len(widths) // 2], heights[len(heights) // 2]


class HeatmapBackend:
    name = NAME

    def __init__(
        self,
        *,
        epochs: int = 30,
        device: str | None = None,
        min_score: float = 0.5,
        learning_rate: float = 3e-4,
        seed: int = 0,
        channels: int = 24,
        optimizer: str = "adamw",
        momentum: float = 0.9,
        weight_decay: float = 0.0,
    ):
        self.epochs = epochs
        self.device = device
        self.min_score = min_score
        self.learning_rate = learning_rate
        self.seed = seed
        self.channels = channels
        self.optimizer = optimizer
        self.momentum = momentum
        self.weight_decay = weight_decay

    def train(self, examples, *, classes, on_progress=None, window=None) -> HeatmapWeights:
        converted = [example_from(e) for e in examples]
        model, _ = heatmap_train.train(
            converted,
            epochs=self.epochs,
            learning_rate=self.learning_rate,
            seed=self.seed,
            crop=window,
            device=self.device,
            classes=dict(classes) or None,
            channels=self.channels,
            optimizer=self.optimizer,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
            on_progress=(lambda p: on_progress(p.epoch, p.epochs, p.loss)) if on_progress else None,
        )
        return HeatmapWeights(model=model, classes=dict(classes), box=fitted_box(converted))

    def predict(self, weights, frames, *, classes) -> list[dict]:
        if weights.box is None:
            raise HeatmapBackendError(
                "these weights do not say what box size they were trained at - a checkpoint saved "
                "before backends named themselves; retrain, or save it again through this backend"
            )
        width, height = weights.box
        return heatmap_train.sweep(
            weights.model,
            dict(classes) or {"object": 0},
            [Path(f) for f in frames],
            width=width,
            height=height,
            min_score=self.min_score,
        )

    def save(self, weights, path: Path) -> Path:
        heatmap_train.save(weights.model, path)
        meta = {"backend": NAME, "classes": weights.classes, "box": weights.box}
        sidecar(path).write_text(json.dumps(meta, indent=2))
        return path

    def load(self, path: Path) -> HeatmapWeights:
        # a checkpoint older than the sidecar still loads; it just cannot predict until sized
        meta = json.loads(sidecar(path).read_text()) if sidecar(path).is_file() else {}
        box = meta.get("box")
        return HeatmapWeights(
            model=heatmap_train.load(path, device=self.device),
            classes=meta.get("classes", {}),
            box=tuple(box) if box else None,
        )
