"""convert a heatmap checkpoint to core ml ahead of time, for one frame size; the package is
named for its input shape (`weights.pt.468x720.mlpackage`) so several sizes can sit side by side.

    uv run --extra coreml python -m smolsmort.detect.export_coreml weights.pt --height 936

the input width follows the checkpoint's capture width and downscale (1440 / 2 = 720); the height
is the frame height at that capture width, downscaled the same way. `load(path, runtime="coreml")`
finds the package beside the checkpoint and skips its own first-call conversion.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from smolsmort.detect.model import capture_width_of, downscale_of
from smolsmort.detect.runtime import convert


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("weights", type=Path)
    parser.add_argument(
        "--height", type=int, required=True, help="frame height in px at the capture width"
    )
    parser.add_argument(
        "--width", type=int, help="frame width in px; default: the checkpoint's capture width"
    )
    args = parser.parse_args(argv)
    from smolsmort.detect.train import load

    model = load(args.weights, device="cpu")
    capture = args.width or capture_width_of(model)
    if not capture:
        parser.error("the checkpoint recorded no capture width - pass --width")
    factor = downscale_of(model)
    shape = (1, 3, args.height // factor, capture // factor)
    target = convert(model, args.weights, shape)
    print(f"wrote {target} for input {shape} (capture {capture}x{args.height}, downscale {factor})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
