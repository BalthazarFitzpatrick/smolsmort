"""pixel-perfect nameplate templates, extracted from Balthazar Fitzpatrick's tight crops and matched against
real frames as an alternative teacher to the border-pair heuristic (spike_nameplate_border.py).

THE IDEA: the classical detector fails on warm, textured terrain (chat text, dirt, foliage - all
measured false positives, task 53) because it only tests local structure (a warm run of the right
width). A real per-colour template, cross-correlated against a frame, tests actual appearance
instead - it should reject terrain far more reliably, at the cost of needing one template per
(hostility colour, target/non-target state) combo it is meant to find.

TRIM RULE, Balthazar Fitzpatrick's, measured against his own crops (screenshots/nameplate *.png):
  1. locate the bar's true rectangle within the tight crop, using the crop's own corners as the
     background reference (grass, in every sample so far)
  2. inset that rectangle 1px on all four sides - the boundary row/column is an anti-aliased
     blend with the background, not reliable border colour
  3. mask off a 3px triangular wedge at each of the four corners of the inset rectangle - the
     rounded end-caps mean the true corners are not square gold, they are a blend, and forcing a
     match there would fight the correlation rather than help it
  4. keep only the LEFT HALF as the template - the right half carries the HP number and level
     box, which differ per mob and per moment, while the left cap + border + fill carries none of
     that and repeats across every mob sharing the same (colour, state)

FILL COLOUR IS PART OF THE TEMPLATE, deliberately - confirmed static across the HP range on a red
sample (measured (214,71,58) on a full-health crop against (208,71,55) on another red sample), so
it is signal, not noise, for at least that colour. Unconfirmed for the others yet.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# how far (euclidean, 0-255 RGB) a pixel has to sit from the crop's own background corners to
# count as "bar", not backdrop. background varies by scene (grass so far); this is generous
# enough to hold across different terrain colours without also catching JPEG noise
BACKGROUND_DISTANCE = 28

# the corner wedge Balthazar Fitzpatrick specified, in pixels of the inset rectangle
CORNER_WEDGE = 3


class TemplateError(Exception):
    pass


@dataclass(frozen=True)
class PlateTemplate:
    name: str
    rgb: np.ndarray  # HxWx3 uint8
    mask: np.ndarray  # HxW bool, True = counts toward the match

    @property
    def height(self) -> int:
        return self.rgb.shape[0]

    @property
    def width(self) -> int:
        return self.rgb.shape[1]


def _background_colour(image: np.ndarray, sample: int = 3) -> np.ndarray:
    """the crop's own corners, averaged - a tight crop's corners are backdrop by construction"""
    h, w, _ = image.shape
    sample = min(sample, h, w)
    corners = np.concatenate(
        [
            image[:sample, :sample].reshape(-1, 3),
            image[:sample, -sample:].reshape(-1, 3),
            image[-sample:, :sample].reshape(-1, 3),
            image[-sample:, -sample:].reshape(-1, 3),
        ]
    )
    return corners.astype(float).mean(axis=0)


# a row belongs to the bar only if a MAJORITY of it differs from background - "any one pixel"
# is what let a name-text row above the bar pull background into the box in the first place
ROW_MAJORITY = 0.5


def locate_bar(image: np.ndarray) -> tuple[int, int, int, int]:
    """(top, left, bottom, right) bounding box of the bar within a tight crop, exclusive.

    two passes, deliberately not one: a full-width bar row is unambiguous, so find ROWS first by
    requiring a majority of the row to differ from background. only THEN look for columns, and
    only within those rows - a lone bright pixel in the name-text row above the bar must not be
    allowed to widen the box, and restricting to real bar rows first is what stops it.
    """
    background = _background_colour(image)
    distance = np.linalg.norm(image.astype(float) - background, axis=2)
    bar = distance > BACKGROUND_DISTANCE

    row_share = bar.mean(axis=1)
    rows = np.where(row_share > ROW_MAJORITY)[0]
    if len(rows) == 0:
        raise TemplateError("no row has a majority of pixels differing from background")
    top, bottom = int(rows[0]), int(rows[-1]) + 1

    cols = np.where(bar[top:bottom].any(axis=0))[0]
    if len(cols) == 0:
        raise TemplateError("no column differs enough from background within the bar rows")
    return top, int(cols[0]), bottom, int(cols[-1]) + 1


def _corner_wedge_mask(height: int, width: int, wedge: int = CORNER_WEDGE) -> np.ndarray:
    """True where a pixel is INSIDE one of the four corner wedges - i.e. should be ignored"""
    ii, jj = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    top_left = ii + jj < wedge
    top_right = ii + (width - 1 - jj) < wedge
    bottom_left = (height - 1 - ii) + jj < wedge
    bottom_right = (height - 1 - ii) + (width - 1 - jj) < wedge
    return top_left | top_right | bottom_left | bottom_right


# how much of the bar's width survives the cut, left-to-right. NOT half any more - Balthazar Fitzpatrick's own
# measurement: at 50% the HP number's digits already start bleeding into the kept region on some
# mobs, and it only gets worse on higher-level ones (more digits, wider level box, both pushing
# that text further left). 40% buys margin against that without losing anything real, since the
# discarded strip was already redundant fill colour - "no gain in more of the same pixels"
KEPT_WIDTH_FRACTION = 0.4


def extract_template(image: np.ndarray, name: str) -> PlateTemplate:
    """crop -> trimmed, corner-masked, left-portion template, per Balthazar Fitzpatrick's rule"""
    top, left, bottom, right = locate_bar(image)
    # inset 1px on all sides, dropping the anti-aliased boundary
    top, left, bottom, right = top + 1, left + 1, bottom - 1, right - 1
    if bottom - top < 3 or right - left < 6:
        raise TemplateError(f"{name}: bar too small after inset ({bottom - top}x{right - left})")

    inset = image[top:bottom, left:right]
    height, width = inset.shape[:2]
    keep = ~_corner_wedge_mask(height, width)

    kept_width = round(width * KEPT_WIDTH_FRACTION)
    return PlateTemplate(name=name, rgb=inset[:, :kept_width], mask=keep[:, :kept_width])


def crop_at_rect(
    image: np.ndarray, name: str, top: int, left: int, height: int, width: int
) -> PlateTemplate:
    """a fixed-size crop at a known-good position - no locate_bar re-detection.

    for the drag-to-align workflow specifically: once a human has lined a candidate up against a
    fixed-size guide, the guide's own dimensions ARE the answer. re-running locate_bar's adaptive
    row/column detection on every slightly-shifted crop instead introduces real noise - a 1px
    difference in exactly where the drag landed changes which rows/columns pass the background-
    distance test, which is what made the kernel preview's size visibly jump during a drag, and
    on a genuine misalignment produced an asymmetric border (more rows kept on one side than the
    other) with a sliver of real background surviving into the "kept" region. a fixed geometric
    crop can't do either, by construction - it always returns exactly (height, width).
    """
    sub = image[top : top + height, left : left + width]
    if sub.shape[0] != height or sub.shape[1] != width:
        raise TemplateError(f"{name}: {width}x{height} at ({left},{top}) falls outside the frame")
    return PlateTemplate(name=name, rgb=sub, mask=~_corner_wedge_mask(height, width))


def mirrored_guide_mask(template: PlateTemplate) -> np.ndarray:
    """a full-bar alignment guide, mirroring the template's left-half mask to make it symmetric.

    used to show a fixed silhouette to drag a candidate image into alignment with - the template
    itself is deliberately only ever the left half (the right carries HP number/level box), but a
    human aligning by eye wants to see the WHOLE bar's expected shape, not half of it.
    """
    return np.concatenate([template.mask, template.mask[:, ::-1]], axis=1)


def save_template(template: PlateTemplate, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    out = directory / f"{template.name}.npz"
    np.savez(out, rgb=template.rgb, mask=template.mask)
    return out


def load_template(path: Path) -> PlateTemplate:
    data = np.load(path)
    return PlateTemplate(name=path.stem, rgb=data["rgb"], mask=data["mask"])


def load_templates(directory: Path) -> list[PlateTemplate]:
    if not directory.is_dir():
        raise TemplateError(f"no template directory at {directory}")
    templates = [load_template(p) for p in sorted(directory.glob("*.npz"))]
    if not templates:
        raise TemplateError(f"{directory} holds no templates")
    return templates


def _normalised_cross_correlation(frame: np.ndarray, template: PlateTemplate) -> np.ndarray:
    """masked NCC via FFT - no opencv dependency, and the frame is only ever a few MB.

    scored per top-left placement: for each position, correlate the template against the
    UNMASKED pixels only, normalised so a perfect match anywhere scores 1.0 regardless of local
    brightness. this is the textbook masked-template-matching formula (Lewis 1995), computed with
    FFT convolutions instead of a python-level sliding window because a 2560x1440 frame makes the
    naive loop too slow to be worth writing.
    """
    frame_f = frame.astype(np.float64)
    template_f = template.rgb.astype(np.float64)
    mask_f = template.mask.astype(np.float64)

    fh, fw, _ = frame_f.shape
    th, tw, _ = template_f.shape
    out_h, out_w = fh - th + 1, fw - tw + 1
    if out_h <= 0 or out_w <= 0:
        raise TemplateError("template is larger than the frame")

    n = mask_f.sum()
    if n < 1:
        raise TemplateError("template mask is empty")

    fft_shape = (fh + th, fw + tw)  # generous padding, avoids wraparound

    def conv(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """valid-region cross-correlation of a (frame-sized) against b (template-sized)"""
        fa = np.fft.rfft2(a, fft_shape)
        fb = np.fft.rfft2(b[::-1, ::-1], fft_shape)  # flip for correlation, not convolution
        full = np.fft.irfft2(fa * fb, fft_shape)
        return full[th - 1 : th - 1 + out_h, tw - 1 : tw - 1 + out_w]

    template_mean = (template_f * mask_f[..., None]).sum(axis=(0, 1)) / n
    template_centred = (template_f - template_mean) * mask_f[..., None]
    template_energy = float((template_centred**2).sum())
    if template_energy < 1e-9:
        raise TemplateError("template has no variance to match against")

    score = np.zeros((out_h, out_w))
    frame_energy = np.zeros((out_h, out_w))
    frame_sum = np.zeros((out_h, out_w))
    ones = np.ones((fh, fw))
    window_n = conv(ones, mask_f)

    for channel in range(3):
        score += conv(frame_f[:, :, channel], template_centred[:, :, channel])
        frame_sum_c = conv(frame_f[:, :, channel], mask_f)
        frame_sq_c = conv(frame_f[:, :, channel] ** 2, mask_f)
        frame_mean_c = frame_sum_c / np.maximum(window_n, 1e-9)
        frame_energy += frame_sq_c - 2 * frame_mean_c * frame_sum_c + frame_mean_c**2 * window_n
        frame_sum += frame_sum_c

    denom = np.sqrt(np.maximum(frame_energy, 0) * template_energy)
    return np.divide(score, denom, out=np.zeros_like(score), where=denom > 1e-6)


# a full-resolution FFT over the whole 2560x1440 frame is what made this slow (~2s/template
# measured, ~15s/frame across 7 templates - unusable over thousands of frames). COARSE_FACTOR
# runs the FFT search on a block-averaged frame/template instead, then REFINE_RADIUS does an
# exact-resolution brute-force search only in a small window around each coarse peak. compute for
# the FFT pass drops by roughly COARSE_FACTOR**2; the brute-force refine is cheap because it only
# ever touches a handful of small windows, not the whole frame
COARSE_FACTOR = 4
REFINE_RADIUS = COARSE_FACTOR + 2


def _downsample(image: np.ndarray, factor: int) -> np.ndarray:
    """block-average downsample, cropping any remainder so every block is full-size"""
    h, w = image.shape[:2]
    h, w = h - h % factor, w - w % factor
    cropped = image[:h, :w]
    if cropped.ndim == 2:
        return cropped.reshape(h // factor, factor, w // factor, factor).mean(axis=(1, 3))
    channels = cropped.shape[2]
    return cropped.reshape(h // factor, factor, w // factor, factor, channels).mean(axis=(1, 3))


def _coarse_template(template: PlateTemplate, factor: int) -> PlateTemplate:
    """downsampled template, keeping only blocks that were mostly real (unmasked) pixels"""
    mask_share = _downsample(template.mask.astype(np.float64), factor)
    masked_rgb = template.rgb.astype(np.float64) * template.mask[..., None]
    rgb_share = _downsample(masked_rgb, factor)
    with np.errstate(invalid="ignore"):
        rgb = np.divide(
            rgb_share,
            mask_share[..., None],
            out=np.zeros_like(rgb_share),
            where=mask_share[..., None] > 0,
        )
    mask = mask_share > 0.5
    if not mask.any():
        raise TemplateError(f"{template.name}: nothing survives downsampling by {factor}")
    return PlateTemplate(name=template.name, rgb=rgb.astype(np.uint8), mask=mask)


def _score_at(frame: np.ndarray, template: PlateTemplate, top: int, left: int) -> float:
    """exact-resolution NCC score for the template placed at one specific position"""
    th, tw = template.height, template.width
    window = frame[top : top + th, left : left + tw].astype(np.float64)
    mask = template.mask
    n = mask.sum()
    if window.shape[:2] != (th, tw) or n < 1:
        return 0.0

    tmpl = template.rgb.astype(np.float64)
    tmpl_mean = (tmpl * mask[..., None]).sum(axis=(0, 1)) / n
    tmpl_centred = (tmpl - tmpl_mean) * mask[..., None]
    tmpl_energy = float((tmpl_centred**2).sum())

    window_mean = (window * mask[..., None]).sum(axis=(0, 1)) / n
    window_centred = (window - window_mean) * mask[..., None]
    window_energy = float((window_centred**2).sum())

    denom = math.sqrt(max(window_energy, 0.0) * tmpl_energy)
    if denom < 1e-6:
        return 0.0
    return float((window_centred * tmpl_centred).sum()) / denom


def _refine(
    frame: np.ndarray, template: PlateTemplate, approx_top: int, approx_left: int
) -> tuple[float, int, int]:
    """brute-force exact search in a small window around a coarse-pass peak"""
    fh, fw, _ = frame.shape
    th, tw = template.height, template.width
    top0 = max(0, approx_top - REFINE_RADIUS)
    left0 = max(0, approx_left - REFINE_RADIUS)
    top1 = min(fh - th, approx_top + REFINE_RADIUS)
    left1 = min(fw - tw, approx_left + REFINE_RADIUS)

    best = (-1.0, approx_top, approx_left)
    for top in range(top0, top1 + 1):
        for left in range(left0, left1 + 1):
            score = _score_at(frame, template, top, left)
            if score > best[0]:
                best = (score, top, left)
    return best


@dataclass(frozen=True)
class TemplateGuess:
    name: str
    score: float
    top: int
    left: int


def guess_best_template(
    frame: np.ndarray, top: int, left: int, templates: list[PlateTemplate]
) -> TemplateGuess | None:
    """which library template this position looks most like, for prefilling a label.

    a plain brute-force refine at the KNOWN position - no coarse search needed, since the caller
    already has an approximate box (from the border-pair teacher) to check against. cheap enough
    to run per-candidate-per-page rather than precomputing the whole session up front.

    carries the REFINED position too, not just the name - the border-pair teacher's box is only
    approximate, and the refine step already finds the exact spot the winning template matched
    best. a caller positioning a crop rectangle should use that, not the raw teacher box.
    """
    if not templates:
        return None
    best: TemplateGuess | None = None
    for template in templates:
        score, refined_top, refined_left = _refine(frame, template, top, left)
        if best is None or score > best.score:
            best = TemplateGuess(
                name=template.name, score=score, top=refined_top, left=refined_left
            )
    return best


def delete_template(directory: Path, name: str) -> None:
    path = directory / f"{name}.npz"
    path.unlink(missing_ok=True)


def find_matches(
    frame: np.ndarray, template: PlateTemplate, threshold: float = 0.6, max_matches: int = 12
) -> list[dict]:
    """non-max-suppressed match locations for one template against one frame.

    coarse-to-fine: an FFT pass on a block-averaged frame/template finds APPROXIMATE peaks fast,
    then each is refined to an exact position/score with a small brute-force search at full
    resolution (see COARSE_FACTOR/REFINE_RADIUS above). the coarse pass uses a looser threshold
    than the caller's, since block-averaging blurs scores down - the real threshold is applied
    after refinement, against the exact score.

    returns capture-pixel boxes in the ORIGINAL (pre-crop) sense: left/top/width/height describe
    where the LEFT-HALF template landed, not the full plate - callers that want the full plate
    width should double it, since the template is deliberately only half the bar.
    """
    coarse_frame = _downsample(frame, COARSE_FACTOR)
    coarse_template = _coarse_template(template, COARSE_FACTOR)
    coarse_scores = _normalised_cross_correlation(coarse_frame, coarse_template)

    coarse_threshold = threshold * 0.7  # block-averaging blurs true peaks down; refine sorts it out
    working = coarse_scores.copy()
    half_h, half_w = coarse_template.height // 2, coarse_template.width // 2
    approx_peaks = []
    for _ in range(max_matches * 3):  # a generous coarse net; refinement and threshold trim it
        peak = np.unravel_index(np.argmax(working), working.shape)
        if working[peak] < coarse_threshold:
            break
        top, left = int(peak[0]), int(peak[1])
        approx_peaks.append((top * COARSE_FACTOR, left * COARSE_FACTOR))
        r0, r1 = max(0, peak[0] - half_h), min(working.shape[0], peak[0] + half_h + 1)
        c0, c1 = max(0, peak[1] - half_w), min(working.shape[1], peak[1] + half_w + 1)
        working[r0:r1, c0:c1] = -math.inf

    refined = [_refine(frame, template, top, left) for top, left in approx_peaks]
    refined = [r for r in refined if r[0] >= threshold]
    refined.sort(key=lambda r: -r[0])

    matches = []
    half_h, half_w = template.height // 2, template.width // 2
    for value, top, left in refined:
        if any(abs(top - m["top"]) < half_h and abs(left - m["left"]) < half_w for m in matches):
            continue  # the same plate, found from two nearby coarse peaks
        matches.append(
            {
                "left": left,
                "top": top,
                "width": template.width,
                "height": template.height,
                "score": round(float(value), 3),
                "template": template.name,
            }
        )
        if len(matches) >= max_matches:
            break
    return matches
