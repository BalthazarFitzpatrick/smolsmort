"""training a model behind the ModelBackend seam: bind a training set or load weights, train (in the
foreground or on a background thread, with abort), sweep, look at the heatmap channel by channel,
save the result under a name, browse where checkpoints live.

THE REFERENCE DRIVER OF SEAM 4 (see docs/REVIEW_TOOL_DESIGN.md). `smolsmort/detect` - the heatmap cnn
- is the first `ModelBackend` and the one every method below is proven against, but training, sweeping
and saving all go through the seam's four methods (`train`, `predict`, `save`, `load`), never through a
backend's own module directly, so a different backend picked by name (`smolsmort/boxes`, an outside
one registered with `smolsmort.backends.register`) drives identically. Only per-channel heatmap
inspection reaches past the seam, because looking at the raw response before any threshold is cut is
not something the seam's `predict` can answer - see `channel_heatmap` below for why that is backend
-shaped rather than seam-shaped.

STILL WITHOUT `ReviewState` or a `paths` module - those belong to later port cards (see the package
layout in the design doc). This holds a backend and the state of one run; the checkpoint-browsing
methods below take the checkpoints folder as an argument for the same reason, rather than importing
`smolsmort.review.paths.CHECKPOINTS_DIR`, which does not exist on this branch yet.
"""

from __future__ import annotations

import json
import math
import threading
from collections.abc import Callable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from smolsmort.backends import example_from, get_backend, sidecar
from smolsmort.detect.dataset import frame_width
from smolsmort.review.backends import class_map, class_names, load_provenance, save_named

# the heatmap backend's own default learning rate (smolsmort/detect/train.py's `train`) - the backend
# adapter does not expose it as an option, so every run uses it silently. named here so a saved run's
# provenance can say what was actually used rather than leaving it unstated
DEFAULT_LEARNING_RATE = 3e-4
# the heatmap and box trainers' own default seed (smolsmort/detect/train.py, smolsmort/boxes/train.py)
DEFAULT_SEED = 0


class TrainStateError(Exception):
    pass


def _idle_job() -> dict[str, Any]:
    """the job dict of a run that has not started. every key `status` promises exists from the
    first poll, so a poller never has to guess whether a missing key means "not yet" or "never"."""
    return {
        "running": False,
        "state": "idle",
        "epoch": 0,
        "epochs": 0,
        "error": None,
        "finished": False,
        "aborted": None,
        "loss": None,
        "history": [],
        "train_loss": None,
        "val_loss": None,
        "test_loss": None,
        "checkpoint_count": 0,
        "weights": None,
        "window": None,
        "learning_rate": None,
        "seed": None,
    }


def window_floor(box_width: int, box_height: int, *, downscale: int | None = None) -> int:
    """the smallest training window (input px) that holds a box of this size whole, snapped up to a
    whole number of network cells. heatmap-shaped: it needs that backend's downscale, stride and
    jitter, imported lazily so this module still loads without torch.

    REPORTED, NOT JUST ENFORCED: a bare "283" means nothing without the box that produced it.
    """
    from smolsmort.detect.model import DEFAULT_DOWNSCALE as DOWNSCALE
    from smolsmort.detect.train import JITTER_FRACTION, snapped_window

    scale = DOWNSCALE if downscale is None else downscale
    longest = max(box_width, box_height) / scale
    return snapped_window(int(math.ceil(longest / (1.0 - 2.0 * JITTER_FRACTION))))


class TrainingAborted(Exception):
    """raised from inside the `on_progress` callback `start` wraps around `train` - a backend only
    ever calls back once per epoch, so that is the only place an abort can land without cutting a
    backend's own training loop open (ported from the parent project's `TrainingAborted`)."""

    def __init__(self, epoch: int):
        super().__init__(f"aborted at epoch {epoch}")
        self.epoch = epoch


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
        # held-out frames, never trained on: `val_examples` is judged for tuning, `test_examples`
        # once at the end of a run. build both with `smolsmort.review.splits.partition`
        self.val_examples: list = []
        self.test_examples: list = []
        self.classes: dict[str, int] = {}
        self.training_set: str | None = None
        self.weights: Any = None
        self.loaded_weights: Path | None = None
        # BACKGROUND TRAINING STATE. the parent project's own reason still holds: this is meant to be
        # driven from a web server, and a blocked server has no progress bar and no way to look at
        # anything while a run works - so `start` below trains on a worker thread instead of the
        # caller's own, and `job` is what a poller reads back.
        self._job_lock = threading.Lock()
        self._worker: threading.Thread | None = None
        # set by `abort`, read from the wrapped progress callback, cleared by every `start`
        self._abort = threading.Event()
        self.job: dict[str, Any] = _idle_job()

    # ---- binding ------------------------------------------------------------------------
    def bind(
        self,
        examples: list,
        classes: Mapping[str, int],
        *,
        training_set: str | None = None,
        val_examples: list | None = None,
        test_examples: list | None = None,
    ):
        """which examples the next run trains on.

        BINDING DROPS ANY LOADED WEIGHTS. a checkpoint carries the channel map it was trained under;
        keeping one loaded against newly bound data would go on offering that checkpoint's classes for
        data they were never trained on, which is silently wrong rather than loudly so.
        """
        self.examples = list(examples)
        self.val_examples = list(val_examples or [])
        self.test_examples = list(test_examples or [])
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
        self.classes = dict(getattr(self.weights, "classes", None) or {}) or class_map(
            meta.get("classes")
        )
        self.training_set = meta.get("training_set")
        self.examples = []
        self.val_examples = []
        self.test_examples = []
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
    def train(
        self,
        *,
        on_progress: Callable[..., None] | None = None,
        window: int | None = None,
        detailed: bool = False,
    ) -> dict:
        """learn from the bound examples, entirely through `ModelBackend.train` - this never imports
        a backend's own training module, which is what makes it work identically against the heatmap
        cnn, the box cnn, or a fake registered for a test.

        `on_progress` gets (epoch, epochs); `start` passes `detailed=True` to also receive the
        optional (loss, weights) a backend may add."""
        if on_progress is not None and not detailed:
            plain = on_progress

            def on_progress(epoch: int, epochs: int, *_extra: Any) -> None:  # noqa: F811
                plain(epoch, epochs)

        if not self.examples:
            raise TrainStateError("nothing bound to train on - bind a training set first")
        # `window` only reaches a backend that was asked for one, so a backend without the knob
        # never sees an unknown keyword
        extra = {} if window is None else {"window": window}
        self.weights = self._backend.train(
            self.examples, classes=self.classes, on_progress=on_progress, **extra
        )
        self.loaded_weights = None
        return self.class_names()

    # ---- background training + abort -----------------------------------------------------
    def start(
        self,
        *,
        on_progress: Callable[[int, int], None] | None = None,
        name: str | None = None,
        root: Path | None = None,
        folder: str = "",
        window: int | None = None,
        checkpoint_every: int | None = None,
    ) -> dict:
        """train on a BACKGROUND THREAD, so a caller (the http server this eventually sits behind)
        stays responsive.

        This drives the same `train` above from a worker, wrapping `on_progress` to update `job` and
        to raise `TrainingAborted` once `abort` has been called - a backend calls back once per
        epoch, so that is the only boundary an abort can land on. Precondition failures surface
        through `job['error']` after the thread runs, like any other training failure.

        A NAMED RUN (`name` plus `root`, the checkpoints folder) writes its own weights when it
        finishes, marked `final`, and is refused up front when the name is taken - a long run is
        never wasted on a name it cannot be saved under. `checkpoint_every=n` also saves
        `<name>-e<epoch>` every n epochs, but only for a backend that hands its weights to the
        progress callback as a fourth argument: `on_progress(epoch, epochs, loss, weights)`. `loss`
        and `weights` are optional, so a two-argument backend works unchanged.

        A `window` (training crop, input px) below `window_floor` is refused, naming both numbers.
        """
        if window is not None and self.backend_name == "heatmap":
            box = self._fitted_box()
            if box is not None:
                floor = window_floor(*box, downscale=self.backend_options.get("downscale"))
                if int(window) < floor:
                    return {
                        "error": (
                            f"a {box[0]}x{box[1]} box needs a window of at least {floor}; "
                            f"{int(window)} would clip it at the jitter extremes"
                        )
                    }
        target: tuple[Path, str] | None = None
        if name is not None and name.strip():
            if root is None:
                return {"error": "a named run needs the checkpoints folder to save into"}
            picked = self.checkpoint_target(Path(root), name, folder)
            if isinstance(picked, dict):
                return picked
            target = picked
        with self._job_lock:
            if self._worker is not None and self._worker.is_alive():
                return {"error": "a training run is already going"}
            self._abort.clear()
            self.job = _idle_job()
            options = {"learning_rate": DEFAULT_LEARNING_RATE, "seed": DEFAULT_SEED}
            options.update(self.backend_options)
            self.job.update(
                running=True,
                state="running",
                weights=target[1] if target else None,
                window=window,
                learning_rate=options["learning_rate"],
                seed=options["seed"],
            )

        def progress(epoch: int, epochs: int, loss: float | None = None, weights: Any = None):
            with self._job_lock:
                self.job["epoch"] = epoch
                self.job["epochs"] = epochs
                if loss is not None:
                    self.job["loss"] = loss
                    self.job["history"].append(loss)
            if self._abort.is_set():
                raise TrainingAborted(epoch)
            interim = checkpoint_every and epoch % checkpoint_every == 0 and epoch < epochs
            if target and weights is not None and interim:
                stem = target[0].with_suffix("")
                self._write_checkpoint(weights, stem.with_name(f"{stem.name}-e{epoch}.pt"))
            if on_progress:
                on_progress(epoch, epochs)

        def run() -> None:
            try:
                self.train(on_progress=progress, window=window, detailed=True)
                losses = self._evaluate_losses()
                if target:
                    self._write_checkpoint(self.weights, target[0], final=True, **losses)
                    self.loaded_weights = target[0]
                with self._job_lock:
                    self.job.update(running=False, finished=True, state="finished", **losses)
            except TrainingAborted as stop:
                # NOTHING FINAL IS SAVED. `train` only assigns `self.weights` after the backend's own
                # `train` returns, so whatever was trained or loaded before stays as it was.
                with self._job_lock:
                    self.job.update(running=False, aborted=stop.epoch, state="aborted")
            except Exception as exc:  # noqa: BLE001
                # deliberately broad: this runs on a worker thread, so anything not caught here
                # leaves `job` pinned at running=True and a poller reading a dead run forever
                with self._job_lock:
                    self.job.update(
                        running=False, state="error", error=f"{type(exc).__name__}: {exc}"
                    )

        self._worker = threading.Thread(target=run, daemon=True)
        self._worker.start()
        return {"ok": True}

    def _fitted_box(self) -> tuple[int, int] | None:
        """the median drawn box of the bound examples, or None when none carries a size. drawn in
        frame px; scaled to the run's `capture_width` option when that differs from the frames"""
        examples = list(map(example_from, self.examples))
        sizes = [size for e in examples for size in e.sizes]
        if not sizes:
            return None
        widths = sorted(int(w) for w, _ in sizes)
        heights = sorted(int(h) for _, h in sizes)
        box = widths[len(widths) // 2], heights[len(heights) // 2]
        capture = self.backend_options.get("capture_width")
        if capture and examples:
            try:
                ratio = capture / frame_width(examples[0].path)
            except OSError:
                return box
            return round(box[0] * ratio), round(box[1] * ratio)
        return box

    def window_floor(self) -> dict | None:
        """the floor for the bound set's own box, with the box and downscale that produced it"""
        from smolsmort.detect.model import DEFAULT_DOWNSCALE

        box = self._fitted_box()
        if box is None:
            return None
        downscale = self.backend_options.get("downscale") or DEFAULT_DOWNSCALE
        return {
            "floor": window_floor(*box, downscale=downscale),
            "box": list(box),
            "downscale": downscale,
        }

    def _evaluate_losses(self) -> dict[str, float | None]:
        """train, val and test loss of the freshly trained weights, through an optional backend
        method `evaluate(weights, examples, *, classes) -> float`.

        TEST LOSS IS COMPUTED ONCE, HERE AT THE END OF A RUN, never per epoch or per checkpoint:
        a number read while tuning stops being held out. A split with no examples, or a backend with
        no `evaluate`, gives None. `train_loss` is the last loss the backend reported, else the
        loss on the train examples.
        """
        evaluate = getattr(self._backend, "evaluate", None)

        def loss_on(examples: list) -> float | None:
            if evaluate is None or not examples:
                return None
            return float(evaluate(self.weights, examples, classes=self.classes))

        with self._job_lock:
            reported = self.job["loss"]
        return {
            "train_loss": reported if reported is not None else loss_on(self.examples),
            "val_loss": loss_on(self.val_examples),
            "test_loss": loss_on(self.test_examples),
        }

    def abort(self) -> dict:
        """stop the running background training at its next epoch boundary. nothing is saved, so
        whatever was trained or loaded before this run stays in use."""
        with self._job_lock:
            running = self.job["running"] and self._worker is not None and self._worker.is_alive()
        if not running:
            return {"error": "no training run is going"}
        self._abort.set()
        return {"ok": True}

    def status(self) -> dict:
        """a snapshot of the background run - safe to poll from another thread while `start` runs.

        keys: running, state (idle/running/finished/aborted/error), epoch, epochs, error, finished,
        aborted, loss, history, train_loss, val_loss, test_loss (float or None until computed; the
        last two only at the end of a run), checkpoint_count (checkpoints this run has saved so
        far), weights (the run's checkpoint name or None), window, learning_rate, seed.
        """
        with self._job_lock:
            snapshot = dict(self.job)
            snapshot["history"] = list(self.job["history"])
            return snapshot

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
    def save(self, path: Path, *, final: bool = False) -> dict:
        """snapshot the current weights under a name, with provenance beside them.

        `backend.save` writes the seam's own sidecar (backend name, classes, box size - see
        `smolsmort.backends.sidecar`); `save_named` adds what the seam does not know - the training
        set and the options this run actually used, falling back to the two knobs every backend uses
        even when it does not expose them (`DEFAULT_LEARNING_RATE`, `DEFAULT_SEED`) - so a checkpoint
        picked by name later can be told apart from another run on the same data.
        """
        if self.weights is None:
            raise TrainStateError("nothing trained or loaded yet - train a model first")
        return self._write_checkpoint(self.weights, Path(path), final=final, count=False)

    def _write_checkpoint(
        self, weights: Any, path: Path, *, final: bool = False, count=True, **extra
    ):
        """one checkpoint through `save_named`; a run's own writes also bump `checkpoint_count`"""
        options = {
            "learning_rate": DEFAULT_LEARNING_RATE,
            "seed": DEFAULT_SEED,
            **self.backend_options,
        }
        saved = save_named(
            self._backend,
            weights,
            path,
            backend_name=self.backend_name,
            training_set=self.training_set,
            # the same {label: index} dict the backend sidecar carries; an index is what a
            # channel means, and a name list threw it away
            classes=dict(sorted(self.classes.items(), key=lambda kv: kv[1])),
            options=options,
            final=final,
            **extra,
        )
        if count:
            with self._job_lock:
                self.job["checkpoint_count"] += 1
        return saved

    def checkpoint_target(self, root: Path, name: str, folder: str = "") -> tuple[Path, str] | dict:
        """(weights path, its name under `root`) for a typed name, or an error dict.

        shared by a named run and a manual save so both refuse the same things: a folder outside
        `root`, and a name already taken. the name is rebuilt from an allowlist so it cannot carry a
        path or a dot that would confuse the provenance file beside it.
        """
        here = _confine(Path(root), folder or "")
        if here is None:
            return {"error": "that folder is outside the checkpoints folder"}
        raw = name.strip().removesuffix(".pt")
        typed = "".join(c for c in raw if c.isalnum() or c in "-_")
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        stem = typed or f"{self.training_set or 'unbound'}-{stamp}"
        weights = here / f"{stem}.pt"
        relative = weights.relative_to(Path(root).resolve()).as_posix()
        if weights.exists():
            return {"error": f"{relative} already exists - pick another name"}
        weights.parent.mkdir(parents=True, exist_ok=True)
        return weights, relative

    # ---- checkpoint-folder browsing ---------------------------------------------------------
    # `root` IS THE CHECKPOINTS FOLDER, taken as an argument everywhere below rather than read off a
    # `smolsmort.review.paths` module - that module does not exist on this branch yet (a later port
    # card adds it and the routes that call these with `paths.CHECKPOINTS_DIR`). Ported from the
    # parent project's `checkpoint_folders` / `saved_checkpoints`, which read it off `review.paths`.
    def checkpoint_folders(self, root: Path, under: str = "") -> dict:
        """the folders under `root` a checkpoint can be saved into, one level at a time, with a
        suggested name for the next save.

        CONFINED TO `root`: the picker only ever lists what is under it, so a checkpoint saved
        anywhere else could never be found again by browsing from the page.
        """
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        here = _confine(root, under or "")
        if here is None or not here.is_dir():
            here = root.resolve()
        relative = here.relative_to(root.resolve()).as_posix()
        relative = "" if relative == "." else relative
        parent = None
        if relative:
            parent = Path(relative).parent.as_posix()
            parent = "" if parent == "." else parent
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        return {
            "here": relative,
            "parent": parent,
            "folders": sorted(p.name for p in here.iterdir() if p.is_dir()),
            "name": f"{self.training_set or 'unbound'}-{stamp}",
        }

    def resolve_checkpoint(self, root: Path, name: str) -> Path | None:
        """a checkpoint's path under `root` by the name `saved_checkpoints` listed it under, or
        None when `name` would leave `root` - the same confinement `checkpoint_folders` applies to
        where a save may go applies to where a load may come from."""
        path = _confine(root, name)
        return path if path is not None and path.is_file() else None


def saved_checkpoints(root: Path) -> dict:
    """the named checkpoints under `root`, newest first, each with the classes it carries and a `final`
    flag (True for the weights a finished run wrote, False for interim and manual saves).

    the classes come from the review-level provenance `save_named` writes beside a checkpoint first;
    a checkpoint saved by a bare `backend.save()` (no review-level provenance at all) falls back to
    the seam's own sidecar (`smolsmort.backends.sidecar`) the same way `TrainState.load_weights` does
    - a checkpoint predating this is still listable, just without a training set to show for it.
    """
    root = Path(root)
    found = []
    saved = sorted(root.rglob("*.pt"), key=_recency, reverse=True) if root.is_dir() else []
    for path in saved:
        meta = load_provenance(path)
        classes = class_names(meta.get("classes"))
        if not classes:
            backend_meta = sidecar(path)
            if backend_meta.is_file():
                try:
                    classes = class_names(json.loads(backend_meta.read_text()).get("classes"))
                except json.JSONDecodeError:
                    classes = []
        found.append(
            {
                "name": path.relative_to(root).as_posix(),
                "training_set": meta.get("training_set"),
                "classes": classes or [],
                "kb": round(path.stat().st_size / 1024),
                "final": bool(meta.get("final", False)),
            }
        )
    return {"weights": found}


def _recency(path: Path) -> tuple[int, bool]:
    """sort key for newest first: modified time, and on a tie a run's final weights come first"""
    return path.stat().st_mtime_ns, bool(load_provenance(path).get("final", False))


def _confine(root: Path, relative: str) -> Path | None:
    """a path inside `root`, or None when `relative` would leave it"""
    root = Path(root).resolve()
    target = (root / relative).resolve()
    try:
        target.relative_to(root)
    except ValueError:
        return None
    return target
