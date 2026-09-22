"""where a heatmap model runs: torch, or core ml on the apple neural engine.

`load(path, runtime="coreml")` hands back a runner with the torch module's call contract - a
(1, 3, H, W) float32 batch in, (1, classes, h, w) logits out - carrying `downscale` and
`capture_width` like the module does, so everything above `load` (frame_input, heatmaps_for_frame,
predict_frame, sweep, decode_peaks) never learns which runtime it holds.

THE .pt STAYS THE SOURCE OF TRUTH. a core ml package is a compiled copy beside it, converted at
fp16 for ONE input shape and named for it - `<name>.pt.468x720.mlpackage` - and reused while it is
newer than the checkpoint; the class-map sidecars are not touched. a core ml model wants its input
shape at conversion, and the first frame is where that shape is known, so a shape the runner has no
package for is converted on first sight (about half a second) rather than refused: a capture that
changes resolution costs one conversion, not a restart. `smolsmort.detect.export_coreml` does the
same ahead of time for a process that must not pay it.

MEASURED 2026-09-22 on an apple m4 (10 cores), python 3.13, torch 2.14, coremltools 9.0, with the
first consumer's checkpoint (100,602 parameters, 18 classes, 1440x936 frames -> (1, 3, 468, 720)
input), the three interleaved through `heatmaps_of` under a load average of 3.4: torch cpu
30.4 ms per frame (p90 39.4; 43/35/33 ms alone at 1/4/8 threads), torch mps 12.1 ms (p90 13.5),
core ml 2.1 ms (p90 2.6; the bare runner call 1.06 ms). fp16 drift against the fp32 module at
most 0.040 in logits, 0.0035 after the sigmoid. python 3.14 cannot run it until coremltools ships
wheels for it - see _coremltools.
"""

from __future__ import annotations

import importlib.util
import os
import sys
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


def package_path(weights: Path, input_shape: tuple[int, ...]) -> Path:
    """the compiled copy beside a checkpoint, one per input shape:
    weights/a.pt at (1, 3, 468, 720) -> weights/a.pt.468x720.mlpackage"""
    height, width = int(input_shape[-2]), int(input_shape[-1])
    return weights.with_name(f"{weights.name}.{height}x{width}.mlpackage")


def _coremltools():
    """coremltools with its native libraries, or a named refusal. on a python coremltools has
    no wheels for (3.14 with coremltools 9.0) the package still installs and imports, as a
    pure-python shell whose first model load dies of "BlobWriter not loaded" - so the check is
    for the compiled module, not the import"""
    try:
        import coremltools
    except ImportError as exc:
        raise RuntimeError_(
            'the "coreml" runtime needs coremltools - install smolsmort[coreml]'
        ) from exc
    if not native_libraries_present(coremltools):
        version = ".".join(str(part) for part in sys.version_info[:2])
        raise RuntimeError_(
            f"coremltools {coremltools.__version__} on python {version} has no native libraries "
            "(no wheels for this interpreter) - core ml needs a python it ships them for, "
            "3.11 to 3.13 as of coremltools 9.0"
        )
    return coremltools


def native_libraries_present(coremltools) -> bool:
    """whether the compiled half of coremltools is there: the model loader and the blob writer"""
    return all(
        importlib.util.find_spec(f"coremltools.{name}") is not None
        for name in ("libcoremlpython", "libmilstoragepython")
    )


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
    target = package_path(weights, input_shape)
    package.save(str(target))
    return target


def _input_shape_of(package) -> tuple[int, ...]:
    spec = package.get_spec()
    return tuple(int(d) for d in spec.description.input[0].type.multiArrayType.shape)


class CoreMLRunner:
    """the torch module's call contract over core ml packages, one per input shape.

    `__call__` takes a (1, 3, H, W) float32 torch tensor or numpy array and returns logits as a
    torch tensor on cpu, so `torch.sigmoid(model(x))[0].cpu().numpy()` reads the same. the package
    for a shape is loaded on the first call at that shape, converted first when it is missing or
    older than the checkpoint, and kept for the runner's life; nothing is ever resampled.
    """

    def __init__(self, model, weights: Path):
        self._model = model
        self._weights = Path(weights)
        self._packages: dict[tuple[int, ...], object] = {}
        self.downscale = downscale_of(model)
        self.capture_width = capture_width_of(model)
        self.runtime = "coreml"
        self.device = "cpu"
        _coremltools()  # refuse at load time, not on the first frame, when the extra is missing

    @property
    def shapes(self) -> tuple[tuple[int, ...], ...]:
        """the input shapes this runner holds a package for, in the order they were first seen"""
        return tuple(self._packages)

    def _package_for(self, shape: tuple[int, ...]):
        package = self._packages.get(shape)
        if package is not None:
            return package
        ct = _coremltools()
        target = package_path(self._weights, shape)
        fresh = target.exists() and os.path.getmtime(target) >= os.path.getmtime(self._weights)
        if not fresh:
            started = time.perf_counter()
            convert(self._model, self._weights, shape)
            self.converted_in = time.perf_counter() - started
        package = ct.models.MLModel(str(target), compute_units=ct.ComputeUnit.ALL)
        if _input_shape_of(package) != shape:
            raise RuntimeError_(
                f"{target.name} carries input {_input_shape_of(package)}, not {shape} - "
                "delete it and let the runner convert again"
            )
        self._packages[shape] = package
        return package

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
        package = self._package_for(tuple(int(n) for n in array.shape))
        logits = package.predict({"frame": array})["logits"]
        return torch.from_numpy(np.ascontiguousarray(logits, dtype=np.float32))


def _coreml_runtime(model, path: Path):
    return CoreMLRunner(model, path)


register("coreml", _coreml_runtime)
