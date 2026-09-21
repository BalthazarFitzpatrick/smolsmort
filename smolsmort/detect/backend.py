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
from smolsmort.detect.model import DEFAULT_DOWNSCALE, capture_width_of, downscale_of

NAME = "heatmap"


class HeatmapBackendError(Exception):
    pass


@dataclass
class HeatmapWeights:
    model: object
    classes: dict[str, int]
    box: tuple[int, int] | None
    capture_width: int | None = None
    downscale: int | None = None


def fitted_box(examples) -> tuple[int, int]:
    """the median drawn size (in capture px once the examples are expressed in them) - a hand-drawn box is a few px out each way, and the median is what
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
        downscale: int = DEFAULT_DOWNSCALE,
        capture_width: int | None = None,
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
        self.downscale = downscale
        self.capture_width = capture_width

    def train(
        self, examples, *, classes, on_progress=None, window=None, init=None
    ) -> HeatmapWeights:
        """`init` is earlier weights to continue from: see heatmap_train.train for what is refused"""
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
            downscale=self.downscale,
            capture_width=self.capture_width,
            init_model=init.model if init is not None else None,
            on_progress=(lambda p: on_progress(p.epoch, p.epochs, p.loss)) if on_progress else None,
        )
        # the box is kept in capture px, the size the net saw, and mapped per frame on predict
        capture = capture_width_of(model)
        return HeatmapWeights(
            model=model,
            classes=dict(classes),
            box=fitted_box(heatmap_train.capture_examples(converted, capture)),
            capture_width=capture,
            downscale=downscale_of(model),
        )

    def evaluate(self, weights, examples, *, classes) -> float:
        """mean training loss of the weights over examples, no augmentation"""
        return heatmap_train.evaluate(
            weights.model, [example_from(e) for e in examples], classes=dict(classes) or None
        )

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
        meta = {
            "backend": NAME,
            "classes": weights.classes,
            "box": weights.box,
            "capture_width": capture_width_of(weights.model),
            "downscale": downscale_of(weights.model),
        }
        sidecar(path).write_text(json.dumps(meta, indent=2))
        return path

    def load(self, path: Path) -> HeatmapWeights:
        # a checkpoint older than the sidecar still loads; it just cannot predict until sized
        meta = json.loads(sidecar(path).read_text()) if sidecar(path).is_file() else {}
        box = meta.get("box")
        model = heatmap_train.load(path, device=self.device)
        return HeatmapWeights(
            model=model,
            classes=meta.get("classes", {}),
            box=tuple(box) if box else None,
            capture_width=capture_width_of(model),
            downscale=downscale_of(model),
        )
