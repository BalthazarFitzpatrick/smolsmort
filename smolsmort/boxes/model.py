"""a small centre-and-size cnn: finds objects of VARYING size, and says how big each one is.

WHY THIS EXISTS BESIDE detect/model.py rather than replacing it. The heatmap cnn there is built on
one assumption - the object is a fixed known size - and it is right to stay that way for a consumer
whose objects are. This backend drops the assumption: each object's width and height are regressed
at its centre, so one set can hold objects 4x apart in size, in frames of different resolutions.

THE SHAPE, centernet-lite. Three heads on one stride-4 feature map: a heatmap per class (where and
which, the same idea as detect), log(width, height) in input px, and a sub-cell offset. Log space so
a 10 px error on a 20 px object costs what a 50 px error on a 100 px one does. An encoder down to
stride 32 and back up with skip additions, because the stride-4 map has to SEE the whole of its
largest object to size it: detect's net sees ~27x51 input px, which is why it could never grow a
size head - see receptive_field.

INPUTS ARE RESIZED, NOT CROPPED TO ONE SIZE. every frame is scaled so its long side is
WORK_LONG_SIDE, whatever it was captured at; boxes scale with it and are scaled back on decode. That
is what lets two capture resolutions train as one set, which detect.dataset refuses for its own
backend.

ENCLOSING BOXES ONLY. a turned object gets the axis-aligned box around it. No angle is predicted, so
the candidate schema - left, top, width, height - is unchanged for every seam downstream.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from smolsmort.detect.model import _torch

STRIDE = 4  # heatmap cell : input pixel
# the input is padded to a multiple of this so every stride-2 stage halves exactly
PAD_MULTIPLE = 32
# long side of the working input, in px. provenance: the synthetic spike (u0), where a 6.7x size
# range at 640 and 1280 wide trained inside 8 cpu minutes; a caller with smaller objects raises it
WORK_LONG_SIDE = 768
PEAK_MIN_SCORE = 0.3
# centernet's prior: every heatmap cell starts near sigmoid 0.1, so early focal loss is not swamped
# by thousands of confident wrong background cells
HEAT_PRIOR = -2.19


class BoxModelError(Exception):
    pass


def build_model(classes: int = 1, widths=(16, 32, 64, 96, 128), head: int = 48):
    """the net: rgb in, (heatmap per class, log size, offset) out, all at stride 4"""
    torch = _torch()
    nn = torch.nn
    functional = torch.nn.functional

    def conv(cin, cout, stride=1, dilation=1):
        return nn.Sequential(
            nn.Conv2d(cin, cout, 3, stride=stride, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

    def branch(out):
        return nn.Sequential(conv(head, head), nn.Conv2d(head, out, 1))

    w1, w4, w8, w16, w32 = widths

    class BoxNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = nn.Sequential(conv(3, w1, 2), conv(w1, w1))
            self.s4 = nn.Sequential(conv(w1, w4, 2), conv(w4, w4))
            self.s8 = nn.Sequential(conv(w4, w8, 2), conv(w8, w8))
            self.s16 = nn.Sequential(conv(w8, w16, 2), conv(w16, w16))
            # dilated at the coarsest level: reach for the largest objects without more depth
            self.s32 = nn.Sequential(
                conv(w16, w32, 2), conv(w32, w32, dilation=2), conv(w32, w32, dilation=2)
            )
            self.lat16 = nn.Conv2d(w32, w16, 1)
            self.up16 = conv(w16, w16)
            self.lat8 = nn.Conv2d(w16, w8, 1)
            self.up8 = conv(w8, w8)
            self.lat4 = nn.Conv2d(w8, head, 1)
            self.skip4 = nn.Conv2d(w4, head, 1)
            self.up4 = conv(head, head)
            self.heat = branch(classes)
            self.size = branch(2)
            self.offset = branch(2)
            self.heat[-1].bias.data.fill_(HEAT_PRIOR)

        def forward(self, x):
            f4 = self.s4(self.stem(x))
            f8 = self.s8(f4)
            f16 = self.s16(f8)
            f32 = self.s32(f16)

            def up(small, like):
                return functional.interpolate(small, size=like.shape[-2:], mode="nearest")

            p16 = self.up16(f16 + up(self.lat16(f32), f16))
            p8 = self.up8(f8 + up(self.lat8(p16), f8))
            p4 = self.up4(self.skip4(f4) + up(self.lat4(p8), f4))
            return self.heat(p4), self.size(p4), self.offset(p4)

    return BoxNet()


def classes_in(state: dict) -> int:
    """the class count a checkpoint was trained with, read off its heatmap head"""
    weight = state.get("heat.1.weight")
    if weight is None:
        raise BoxModelError("not a box-model checkpoint: it has no heat.1.weight")
    return int(weight.shape[0])


def widths_in(state: dict) -> tuple[int, int, int, int, int]:
    """the five stage widths a checkpoint was built with, read off each stage's first conv.
    a net built narrower or wider than the default could not be reloaded before this: `load`
    rebuilt the default and load_state_dict refused the shapes"""
    found = []
    for stage in ("stem", "s4", "s8", "s16", "s32"):
        weight = state.get(f"{stage}.0.0.weight")
        if weight is None:
            raise BoxModelError(f"not a box-model checkpoint: it has no {stage}.0.0.weight")
        found.append(int(weight.shape[0]))
    return tuple(found)


def head_in(state: dict) -> int:
    """the head width, read off the lateral conv that feeds it"""
    weight = state.get("lat4.weight")
    if weight is None:
        raise BoxModelError("not a box-model checkpoint: it has no lat4.weight")
    return int(weight.shape[0])


def receptive_field(model, side: int = 1024) -> int:
    """how far one centre cell can see, in input px, measured rather than derived.

    the extent of nonzero gradient from one heatmap cell back to the input. random input, not zeros:
    a zero input leaves every relu at its kink and no gradient flows at all.
    """
    torch = _torch()
    was_training = model.training
    model.eval()
    parameter = next(model.parameters())
    x = torch.rand(1, 3, side, side, device=parameter.device, requires_grad=True)
    heat, _, _ = model(x)
    heat[0, 0, heat.shape[2] // 2, heat.shape[3] // 2].backward()
    reach = x.grad[0].abs().sum(0)
    rows = torch.nonzero(reach.sum(1)).flatten()
    cols = torch.nonzero(reach.sum(0)).flatten()
    model.train(was_training)
    model.zero_grad(set_to_none=True)
    return int(min(rows[-1] - rows[0] + 1, cols[-1] - cols[0] + 1))


def scale_for(width: int, height: int, long_side: int = WORK_LONG_SIDE) -> float:
    return long_side / max(width, height)


def padded(image: np.ndarray) -> np.ndarray:
    """(3, h, w) zero-padded up to a multiple of PAD_MULTIPLE, top-left anchored so coordinates hold"""
    _, height, width = image.shape
    out = np.zeros(
        (
            3,
            math.ceil(height / PAD_MULTIPLE) * PAD_MULTIPLE,
            math.ceil(width / PAD_MULTIPLE) * PAD_MULTIPLE,
        ),
        dtype=np.float32,
    )
    out[:, :height, :width] = image
    return out


def _gaussian(channel: np.ndarray, cx: int, cy: int, sigma_x: float, sigma_y: float) -> None:
    height, width = channel.shape
    rx, ry = int(3 * sigma_x) + 1, int(3 * sigma_y) + 1
    x0, x1 = max(0, cx - rx), min(width, cx + rx + 1)
    y0, y1 = max(0, cy - ry), min(height, cy + ry + 1)
    if x0 >= x1 or y0 >= y1:
        return
    ys = np.arange(y0, y1, dtype=np.float32)[:, None]
    xs = np.arange(x0, x1, dtype=np.float32)[None, :]
    blob = np.exp(-((xs - cx) ** 2 / (2 * sigma_x**2) + (ys - cy) ** 2 / (2 * sigma_y**2)))
    channel[y0:y1, x0:x1] = np.maximum(channel[y0:y1, x0:x1], blob)


@dataclass
class Target:
    """everything the loss compares against, for one window, at stride-4 cells"""

    heat: np.ndarray  # (classes, h, w)
    size: np.ndarray  # (2, h, w) log width, log height in input px, set at centre cells only
    offset: np.ndarray  # (2, h, w) sub-cell position of the centre, set at centre cells only
    centre: np.ndarray  # (h, w) 1 where a size/offset target is set
    mask: np.ndarray  # (h, w) 0 where the loss must not look


def box_target(cells: tuple[int, int], boxes, channels: list[int], classes: int) -> Target:
    """targets for boxes given as (x0, y0, x1, y1) in WINDOW input px.

    a box whose centre falls outside the window is masked where its body shows: it is an object, so
    it must not train as background, but its centre is not here to learn from.

    THE BLOB IS ELLIPTICAL, sized from the box (ttfnet's 0.54 of the box over 6 sigma): an elongated
    object gets an elongated blob. a round one would either spill past a thin object or starve a
    long one of positives along its length.
    """
    height, width = cells
    heat = np.zeros((classes, height, width), dtype=np.float32)
    size = np.zeros((2, height, width), dtype=np.float32)
    offset = np.zeros((2, height, width), dtype=np.float32)
    centre = np.zeros((height, width), dtype=np.float32)
    mask = np.ones((height, width), dtype=np.float32)
    for (x0, y0, x1, y1), channel in zip(boxes, channels, strict=True):
        box_w, box_h = max(1.0, x1 - x0), max(1.0, y1 - y0)
        cx, cy = (x0 + x1) / 2 / STRIDE, (y0 + y1) / 2 / STRIDE
        if not (0 <= cx < width and 0 <= cy < height):
            a, b = max(0, int(x0 / STRIDE)), max(0, int(y0 / STRIDE))
            c, d = max(0, math.ceil(x1 / STRIDE)), max(0, math.ceil(y1 / STRIDE))
            mask[b:d, a:c] = 0.0
            continue
        # the FLOOR cell carries the peak and the offset carries the rest - so the peak is exactly
        # 1.0 and counts as a positive, the trap detect/train.py documents for fractional centres
        ix, iy = int(cx), int(cy)
        sigma_x = max(0.8, 0.54 * box_w / STRIDE / 6)
        sigma_y = max(0.8, 0.54 * box_h / STRIDE / 6)
        if 0 <= channel < classes:
            _gaussian(heat[channel], ix, iy, sigma_x, sigma_y)
        size[:, iy, ix] = (math.log(box_w), math.log(box_h))
        offset[:, iy, ix] = (cx - ix, cy - iy)
        centre[iy, ix] = 1.0
    # a masked region must never hide a centre that is here
    mask[heat.max(axis=0) > 0.3] = 1.0
    return Target(heat=heat, size=size, offset=offset, centre=centre, mask=mask)


def box_loss(outputs, heat, size, offset, centre, mask):
    """focal loss on the heatmap (detect's, masked) plus l1 on size and offset at centre cells"""
    torch = _torch()
    from smolsmort.detect.train import masked_focal_loss

    heat_logits, size_out, offset_out = outputs
    count = centre.sum().clamp(min=1.0)
    at_centre = centre.unsqueeze(1)
    size_loss = ((size_out - size).abs() * at_centre).sum() / count
    offset_loss = ((offset_out - offset).abs() * at_centre).sum() / count
    focal = masked_focal_loss(heat_logits, heat, mask.unsqueeze(1))
    return focal + size_loss + offset_loss, torch.stack([focal, size_loss, offset_loss]).detach()


@dataclass(frozen=True)
class Detection:
    """one found object, in CAPTURE px"""

    left: float
    top: float
    width: float
    height: float
    score: float
    channel: int


def decode_boxes(
    outputs, scale: float, min_score: float = PEAK_MIN_SCORE, limit: int | None = 100
) -> list[Detection]:
    """3x3 local maxima above min_score, strongest first, as boxes in capture px.

    ONLY A 3x3 LOCAL MAXIMUM, no wider suppression. detect's decode refuses peaks within 3 cells of a
    stronger one, which is right for objects that never overlap and wrong here: two overlapping
    objects have centres closer than that, and the second one is exactly what must survive.
    """
    torch = _torch()
    heat_logits, size, offset = outputs
    heat = torch.sigmoid(heat_logits)[0]
    peak = torch.nn.functional.max_pool2d(heat[None], 3, stride=1, padding=1)[0] == heat
    channels, ys, xs = torch.nonzero(peak & (heat >= min_score), as_tuple=True)
    scores = heat[channels, ys, xs]
    order = scores.argsort(descending=True)
    if limit is not None:
        order = order[:limit]
    found = []
    for i in order.tolist():
        channel, y, x = int(channels[i]), int(ys[i]), int(xs[i])
        cx = (x + float(offset[0, 0, y, x])) * STRIDE
        cy = (y + float(offset[0, 1, y, x])) * STRIDE
        width, height = math.exp(float(size[0, 0, y, x])), math.exp(float(size[0, 1, y, x]))
        found.append(
            Detection(
                left=(cx - width / 2) / scale,
                top=(cy - height / 2) / scale,
                width=width / scale,
                height=height / scale,
                score=float(scores[i]),
                channel=channel,
            )
        )
    return found
