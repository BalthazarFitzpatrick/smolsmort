"""where a heatmap model runs: torch, or core ml on the apple neural engine.

`load(path, runtime="coreml")` hands back a runner with the torch module's call contract - a
(1, 3, H, W) float32 batch in, (1, classes, h, w) logits out - carrying `downscale` and
`capture_width` like the module does, so everything above `load` (frame_input, heatmaps_for_frame,
predict_frame, sweep, decode_peaks) never learns which runtime it holds.

THE .pt STAYS THE SOURCE OF TRUTH. the core ml package is a compiled copy beside it,
`<name>.pt.mlpackage`, converted once at fp16 for one fixed input shape and reused while it is
newer than the checkpoint; the class-map sidecars are not touched. a core ml model wants its input
shape at conversion, and the first frame is where that shape is known, so conversion happens on
the first call unless `smolsmort.detect.export_coreml` did it ahead of time.

MEASURED 2026-09-22 on an apple m4 with the first consumer's checkpoint (100,602 parameters,
18 classes, 1440x936 frames -> (1, 3, 468, 720) input): see the spike test's printout and the
release notes for the ms/frame of torch cpu against the neural engine.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from smolsmort.detect.model import capture_width_of, downscale_of

RUNTIMES: dict[str, Callable] = {}


class RuntimeError_(Exception):
    """a runtime that cannot be built: a missing extra, a shape the package was not converted for"""


def register(name: str, factory: Callable) -> None:
    RUNTIMES[name] = factory


def names() -> list[str]:
    return sorted(RUNTIMES)


def build(name: str, model, path: Path):
    """the runner for `name` over the torch `model` loaded from `path`"""
    try:
        factory = RUNTIMES[name]
    except KeyError:
        raise RuntimeError_(f"no runtime called {name!r} - known: {', '.join(names())}") from None
    return factory(model, Path(path))


# ---------------------------------------------------------------- torch


def _torch_runtime(model, path: Path):
    return model


register("torch", _torch_runtime)


# ---------------------------------------------------------------- core ml


def package_path(weights: Path) -> Path:
    """the compiled copy beside a checkpoint: weights/a.pt -> weights/a.pt.mlpackage"""
    return weights.with_name(weights.name + ".mlpackage")


def _coremltools():
    try:
        import coremltools
    except ImportError as exc:
        raise RuntimeError_(
            'the "coreml" runtime needs coremltools - install smolsmort[coreml]'
        ) from exc
    return coremltools


def convert(model, weights: Path, input_shape: tuple[int, int, int, int]) -> Path:
    """convert the torch `model` to a core ml package beside `weights` for one input shape,
    fp16, every compute unit allowed (the neural engine takes it when it can). returns the path"""
    import torch

    ct = _coremltools()
    model = model.eval()
    example = torch.zeros(*input_shape, dtype=torch.float32)
    with torch.no_grad():
        traced = torch.jit.trace(model, example)
    package = ct.convert(
        traced,
        inputs=[ct.TensorType(name="frame", shape=input_shape, dtype=np.float32)],
        outputs=[ct.TensorType(name="logits")],
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.ALL,
    )
    target = package_path(weights)
    package.save(str(target))
    return target


def _input_shape_of(package) -> tuple[int, ...]:
    spec = package.get_spec()
    return tuple(int(d) for d in spec.description.input[0].type.multiArrayType.shape)


class CoreMLRunner:
    """the torch module's call contract over a core ml package.

    `__call__` takes a (1, 3, H, W) float32 torch tensor or numpy array and returns logits as a
    torch tensor on cpu, so `torch.sigmoid(model(x))[0].cpu().numpy()` reads the same. the package
    is converted on the first call when none is beside the checkpoint or it is older than the
    checkpoint; a later call with another shape is refused rather than silently resampled.
    """

    def __init__(self, model, weights: Path):
        self._model = model
        self._weights = Path(weights)
        self._package = None
        self.downscale = downscale_of(model)
        self.capture_width = capture_width_of(model)
        self.runtime = "coreml"
        self.device = "cpu"
        ct = _coremltools()
        target = package_path(self._weights)
        fresh = target.exists() and os.path.getmtime(target) >= os.path.getmtime(self._weights)
        if fresh:
            self._package = ct.models.MLModel(str(target), compute_units=ct.ComputeUnit.ALL)

    @property
    def input_shape(self) -> tuple[int, ...] | None:
        return _input_shape_of(self._package) if self._package is not None else None

    def eval(self):
        return self

    def to(self, device):
        return self

    def __call__(self, x):
        import torch

        array = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
        array = np.ascontiguousarray(array, dtype=np.float32)
        if array.ndim != 4:
            raise RuntimeError_(f"a (1, 3, h, w) batch is expected, got shape {array.shape}")
        if self._package is None:
            ct = _coremltools()
            started = time.perf_counter()
            target = convert(self._model, self._weights, tuple(array.shape))
            self._package = ct.models.MLModel(str(target), compute_units=ct.ComputeUnit.ALL)
            self.converted_in = time.perf_counter() - started
        elif tuple(array.shape) != self.input_shape:
            raise RuntimeError_(
                f"{package_path(self._weights).name} was converted for input {self.input_shape}, "
                f"this frame is {tuple(array.shape)} - re-export it for this size "
                "(python -m smolsmort.detect.export_coreml)"
            )
        logits = self._package.predict({"frame": array})["logits"]
        return torch.from_numpy(np.ascontiguousarray(logits, dtype=np.float32))


def _coreml_runtime(model, path: Path):
    return CoreMLRunner(model, path)


register("coreml", _coreml_runtime)
