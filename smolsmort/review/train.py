"""training a model behind the ModelBackend seam: bind a training set or load weights, train, sweep,
look at the heatmap channel by channel, save the result under a name.

THE REFERENCE DRIVER OF SEAM 4 (see docs/REVIEW_TOOL_DESIGN.md). `smolsmort/detect` - the heatmap cnn
- is the first `ModelBackend` and the one every method below is proven against, but training, sweeping
and saving all go through the seam's four methods (`train`, `predict`, `save`, `load`), never through a
backend's own module directly, so a different backend picked by name (`smolsmort/boxes`, an outside
one registered with `smolsmort.backends.register`) drives identically. Only per-channel heatmap
inspection reaches past the seam, because looking at the raw response before any threshold is cut is
not something the seam's `predict` can answer - see `channel_heatmap` below for why that is backend
-shaped rather than seam-shaped.

DELIBERATELY WITHOUT the web tool's threading, `paths` module or `ReviewState` - those belong to later
port cards (see the package layout in the design doc). This holds a backend and the state of one run;
it touches the filesystem only at the checkpoint path it is given.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from smolsmort.backends import get_backend
from smolsmort.review.backends import load_provenance, save_named

# the heatmap backend's own default learning rate (smolsmort/detect/train.py's `train`) - the backend
# adapter does not expose it as an option, so every run uses it silently. named here so a saved run's
# provenance can say what was actually used rather than leaving it unstated
DEFAULT_LEARNING_RATE = 3e-4
# the heatmap and box trainers' own default seed (smolsmort/detect/train.py, smolsmort/boxes/train.py)
DEFAULT_SEED = 0


class TrainStateError(Exception):
    pass


class TrainState:
    """drives one named `ModelBackend` through bind -> train -> sweep -> inspect -> save.

    BINDING A SET AND LOADING WEIGHTS ARE MUTUALLY EXCLUSIVE, as they were in the parent project's
    trainer: a training set says what to train on, weights say what is already trained, and each one
    carries its own channel map - keeping both would leave the channel picker offering classes that
    belong to whichever one is stale.
    """

    def __init__(self, backend_name: str = "heatmap", **backend_options: Any):
        self.backend_name = backend_name
        self.backend_options = dict(backend_options)
        self._backend = get_backend(backend_name, **self.backend_options)
        self.examples: list = []
        self.classes: dict[str, int] = {}
        self.training_set: str | None = None
        self.weights: Any = None
        self.loaded_weights: Path | None = None

    # ---- binding ------------------------------------------------------------------------
    def bind(self, examples: list, classes: Mapping[str, int], *, training_set: str | None = None):
        """which examples the next run trains on.

        BINDING DROPS ANY LOADED WEIGHTS. a checkpoint carries the channel map it was trained under;
        keeping one loaded against newly bound data would go on offering that checkpoint's classes for
        data they were never trained on, which is silently wrong rather than loudly so.
        """
        self.examples = list(examples)
        self.classes = dict(classes)
        self.training_set = training_set
        self.weights = None
        self.loaded_weights = None

    def load_weights(self, path: Path) -> dict:
        """load a checkpoint AND the class map it was trained under, through the seam's own `load`.

        LOADING DROPS ANY BOUND SET, for the same reason binding drops loaded weights - the training
        set that produced these weights is whatever the provenance beside them says, never whichever
        set happened to be bound before.
        """
        path = Path(path)
        self.weights = self._backend.load(path)
        meta = load_provenance(path)
        # the backend's own weights carry their classes (read from `smolsmort.backends`' sidecar);
        # the provenance file beside a named save is only consulted when that is somehow empty
        self.classes = dict(getattr(self.weights, "classes", None) or meta.get("classes") or {})
        self.training_set = meta.get("training_set")
        self.examples = []
        self.loaded_weights = path
        return self.class_names()

    def class_names(self) -> dict:
        """the channel order of whatever is bound or loaded - derived from the classes present,
        never a fixed list, so a picker over the heatmap only ever offers channels that exist"""
        pairs = sorted(self.classes.items(), key=lambda kv: kv[1])
        return {
            "training_set": self.training_set,
            "weights": self.loaded_weights.name if self.loaded_weights else None,
            "classes": [name for name, _ in pairs],
        }

    # ---- training -----------------------------------------------------------------------
    def train(self, *, on_progress: Callable[[int, int], None] | None = None) -> dict:
        """learn from the bound examples, entirely through `ModelBackend.train` - this never imports
        a backend's own training module, which is what makes it work identically against the heatmap
        cnn, the box cnn, or a fake registered for a test."""
        if not self.examples:
            raise TrainStateError("nothing bound to train on - bind a training set first")
        self.weights = self._backend.train(
            self.examples, classes=self.classes, on_progress=on_progress
        )
        self.loaded_weights = None
        return self.class_names()

    # ---- sweeping (predict) ---------------------------------------------------------------
    def sweep(self, frames: list) -> list[dict]:
        """propose candidates on frames nobody has judged - the seam's `predict`, which is also the
        loop's return edge: what comes back is in the same schema `find` speaks, ready to re-enter
        judging."""
        if self.weights is None:
            raise TrainStateError(
                "no trained or loaded weights to sweep with - train or load first"
            )
        return self._backend.predict(self.weights, [str(f) for f in frames], classes=self.classes)

    # ---- inspecting -------------------------------------------------------------------
    def channel_heatmap(self, frame, channel: int = -1):
        """the model's raw response for one frame, one channel at a time (-1: the loudest of all).

        THIS READS THE MODEL DIRECTLY, past the seam. `predict` only ever returns candidates already
        cut at a threshold, and choosing a threshold at all means looking at the response BEFORE one is
        applied - see the parent project's `peak_distribution` and `overlay_bytes`, which this ports.
        `smolsmort/detect` (`heatmap`) is the reference backend this is proven against: its model
        answers one sigmoid heatmap per class. A box-shaped backend's model answers several heads
        (heat, size, offset, centre) that are not a stack of per-class heatmaps, so this refuses rather
        than reading one of them as if it were.
        """
        if self.weights is None:
            raise TrainStateError("no trained or loaded weights to inspect")
        # by the NAME asked for, not an attribute on the backend instance - the seam (ModelBackend)
        # promises only train/predict/save/load, so a conforming backend need not carry a `.name` at all
        if self.backend_name != "heatmap":
            raise TrainStateError(
                f"channel inspection needs a heatmap-shaped backend; {self.backend_name!r} does not "
                "expose one heatmap per class"
            )
        from smolsmort.detect.train import heatmaps_for

        stack = heatmaps_for(self.weights.model, Path(frame))
        if 0 <= channel < len(stack):
            return stack[channel]
        return stack.max(axis=0)

    # ---- saving -------------------------------------------------------------------------
    def save(self, path: Path) -> dict:
        """snapshot the current weights under a name, with provenance beside them.

        `backend.save` writes the seam's own sidecar (backend name, classes, box size - see
        `smolsmort.backends.sidecar`); `save_named` adds what the seam does not know - the training
        set and the options this run actually used, falling back to the two knobs every backend uses
        even when it does not expose them (`DEFAULT_LEARNING_RATE`, `DEFAULT_SEED`) - so a checkpoint
        picked by name later can be told apart from another run on the same data.
        """
        if self.weights is None:
            raise TrainStateError("nothing trained or loaded yet - train a model first")
        options = {
            "learning_rate": DEFAULT_LEARNING_RATE,
            "seed": DEFAULT_SEED,
            **self.backend_options,
        }
        return save_named(
            self._backend,
            self.weights,
            Path(path),
            backend_name=self.backend_name,
            training_set=self.training_set,
            classes=[name for name, _ in sorted(self.classes.items(), key=lambda kv: kv[1])],
            options=options,
        )
