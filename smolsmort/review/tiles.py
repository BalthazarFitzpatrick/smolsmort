"""a tile: the small crop cut out of a frame around one box, stored one npz per tile.

the on-disk layout is fixed - `rgb` (HxWx3 uint8) and `mask` (HxW bool, True = the pixel counts) in
`<name>.npz` - so tiles written before this module existed open unchanged. a tile is named
"<tag>_k<index>", which is how it resolves back to the box, frame and recording it came from
(see naming.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image


class TileError(Exception):
    pass


@dataclass(frozen=True)
class Tile:
    name: str
    rgb: np.ndarray
    mask: np.ndarray

    @property
    def height(self) -> int:
        return self.rgb.shape[0]

    @property
    def width(self) -> int:
        return self.rgb.shape[1]


def cut_tile(frame: np.ndarray, name: str, *, top: int, left: int, height: int, width: int) -> Tile:
    """a fixed-size crop at a known position, or TileError when it runs off the frame.

    negative starts are refused outright: numpy would wrap them into a short slice that only the
    shape check below happens to catch.
    """
    if top < 0 or left < 0:
        raise TileError(f"{name}: {width}x{height} at ({left},{top}) falls outside the frame")
    sub = frame[top : top + height, left : left + width]
    if sub.shape[0] != height or sub.shape[1] != width:
        raise TileError(f"{name}: {width}x{height} at ({left},{top}) falls outside the frame")
    return Tile(name=name, rgb=sub, mask=np.ones((height, width), dtype=bool))


def save_tile(tile: Tile, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    out = directory / f"{tile.name}.npz"
    np.savez(out, rgb=tile.rgb, mask=tile.mask)
    return out


def load_tile(path: Path) -> Tile:
    with np.load(path) as data:
        return Tile(name=path.stem, rgb=data["rgb"], mask=data["mask"])


def tile_png(tile: Tile) -> bytes:
    """the tile as a png, transparent where the mask says the pixel does not count"""
    rgba = np.dstack([tile.rgb, np.full(tile.mask.shape, 255, dtype=np.uint8)])
    rgba[~tile.mask, 3] = 0
    return png_bytes(Image.fromarray(rgba, mode="RGBA"))


def png_bytes(image: Image.Image) -> bytes:
    buf = BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()
