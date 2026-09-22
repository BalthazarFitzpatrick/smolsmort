"""the core ml runtime: a runner with the torch module's contract, converted once beside the .pt.

the numbers a run prints (ms/frame for torch cpu against the neural engine) are the point of the
spike; the assertions only pin that the package answers like the module. skipped wherever
coremltools is missing, so ci on linux and a plain `smolsmort[vision]` install never see it.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pytest

from smolsmort.detect.model import (
    STRIDE,
    build_model,
    capture_width_of,
    downscale_of,
    set_capture_width,
)
from smolsmort.detect.runtime import RuntimeError_, names, package_path
from smolsmort.detect.train import frame_input, heatmaps_for_frame, heatmaps_of, load, save

torch = pytest.importorskip("torch")
pytest.importorskip("coremltools")

FRAME = (936, 1440, 3)  # the first consumer's capture, downscaled by 2 to (468, 720)
INPUT = (1, 3, 468, 720)
HEAT = (468 // STRIDE, 720 // STRIDE)  # the net's own stride on top of the input downscale


@pytest.fixture
def weights(tmp_path):
    torch.manual_seed(0)
    model = build_model(classes=4, downscale=2, channels=8)
    set_capture_width(model, 1440)
    return save(model, tmp_path / "small.pt")


def test_registry_knows_both_runtimes():
    assert names() == ["coreml", "torch"]


def test_coreml_runner_answers_like_the_module(weights):
    module = load(weights, device="cpu")
    runner = load(weights, runtime="coreml")
    assert downscale_of(runner) == 2 and capture_width_of(runner) == 1440
    frame = np.random.default_rng(1).integers(0, 255, FRAME, dtype=np.uint8)
    x = torch.from_numpy(np.ascontiguousarray(frame_input(module, frame)[0])[None])
    with torch.no_grad():
        ref = torch.sigmoid(module(x)).numpy()
    got = torch.sigmoid(runner(x)).numpy()
    assert got.shape == ref.shape == (1, 4, *HEAT)
    assert np.abs(got - ref).max() < 0.02, "fp16 on the neural engine drifted past 0.02 in sigmoid"
    # the package now sits beside the checkpoint, named for its shape, and is picked up without
    # converting again
    assert package_path(weights, INPUT).name == "small.pt.468x720.mlpackage"
    assert package_path(weights, INPUT).exists()
    again = load(weights, runtime="coreml")
    again(x)
    assert again.shapes == (INPUT,)
    assert not hasattr(again, "converted_in")


def test_stale_package_is_rebuilt_and_another_shape_gets_its_own(weights):
    runner = load(weights, runtime="coreml")
    runner(torch.zeros(*INPUT))
    package = package_path(weights, INPUT)
    # the checkpoint is newer than the package: the runner must convert again on first use
    save(load(weights, device="cpu"), weights)
    stamp = package.stat().st_mtime - 10
    os.utime(package, (stamp, stamp))
    fresh = load(weights, runtime="coreml")
    assert fresh.shapes == ()
    fresh(torch.zeros(*INPUT))
    assert fresh.shapes == (INPUT,) and fresh.converted_in > 0
    # a capture that changes resolution gets a second package beside the first, no refusal
    other = (1, 3, 232, 360)
    out = fresh(torch.zeros(*other))
    assert out.shape[-2:] == (232 // STRIDE, 360 // STRIDE)
    assert fresh.shapes == (INPUT, other)
    assert package_path(weights, other).name == "small.pt.232x360.mlpackage"
    assert package_path(weights, other).exists() and package.exists()


def test_heatmaps_for_frame_takes_the_runner(weights):
    runner = load(weights, runtime="coreml")
    frame = np.random.default_rng(2).integers(0, 255, FRAME, dtype=np.uint8)
    maps, ratio = heatmaps_for_frame(runner, frame)
    assert maps.shape == (4, *HEAT) and ratio == 1.0


def test_heatmaps_of_takes_a_prepared_array_on_both_runtimes(weights):
    """the seam a caller that masks its own input uses: (3, h, w) float32 in, sigmoid maps out"""
    module = load(weights, device="cpu")
    runner = load(weights, runtime="coreml")
    frame = np.random.default_rng(3).integers(0, 255, FRAME, dtype=np.uint8)
    image, _ = frame_input(module, frame)
    image[:, :40, :] = 0.0  # a masked strip, as a consumer zeroes its hud before inference
    ref = heatmaps_of(module, image)
    got = heatmaps_of(runner, image)
    assert got.shape == ref.shape == (4, *HEAT)
    assert np.abs(got - ref).max() < 0.02
    # a numpy batch goes straight into the runner too, and comes back a cpu float32 tensor
    out = runner(image[None])
    assert isinstance(out, torch.Tensor) and out.dtype == torch.float32 and out.device.type == "cpu"


def test_unknown_runtime_is_named(weights):
    with pytest.raises(RuntimeError_, match="no runtime called 'metal'"):
        load(weights, runtime="metal")


def test_spike_prints_ms_per_frame(weights, capsys):
    """not an assertion on speed - hardware and load vary - but the figures the runtime shipped
    on: 2026-09-22, apple m4, python 3.13, 18-class 100,602-parameter checkpoint at
    (1, 3, 468, 720), interleaved under load 3.4: torch cpu 30.4 ms, torch mps 12.1 ms, core ml
    2.1 ms through heatmaps_of (bare call 1.06 ms)"""
    module = load(weights, device="cpu")
    runner = load(weights, runtime="coreml")
    x = torch.rand(*INPUT)

    def bench(fn, n=20):
        fn()
        fn()
        laps = []
        for _ in range(n):
            started = time.perf_counter()
            fn()
            laps.append((time.perf_counter() - started) * 1000)
        return float(np.median(laps))

    torch.set_num_threads(4)
    with torch.no_grad():
        cpu = bench(lambda: module(x))
    ane = bench(lambda: runner(x))
    with capsys.disabled():
        print(f"\ncoreml spike: torch cpu {cpu:.1f} ms/frame, coreml all units {ane:.1f} ms/frame")
    assert ane > 0 and cpu > 0
