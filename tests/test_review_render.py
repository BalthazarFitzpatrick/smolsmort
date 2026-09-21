"""smolsmort/review/render.py: every frame read is confined to the recording's frames folder"""

from __future__ import annotations

import pytest
import review_world

from smolsmort.review.render import ImageRenderer, frame_file


def _frames(tmp_path):
    frames = tmp_path / "rec" / "frames"
    frames.mkdir(parents=True)
    review_world.write_frame(frames / "f00.png")
    (tmp_path / "outside.png").write_bytes((frames / "f00.png").read_bytes())
    return frames


def test_a_bare_name_resolves_and_a_directory_part_is_cut(tmp_path):
    frames = _frames(tmp_path)
    assert frame_file(frames, "f00.png") == (frames / "f00.png").resolve()
    # a stored path with a folder in it still means the file of that name in THIS folder
    assert frame_file(frames, "elsewhere/f00.png") == (frames / "f00.png").resolve()


def test_every_renderer_method_refuses_a_path_outside_the_frames_folder(tmp_path):
    frames = _frames(tmp_path)
    renderer = ImageRenderer()
    with pytest.raises(FileNotFoundError):
        renderer.frame("../outside.png", frames_dir=frames)
    box = {"path": "../outside.png", "left": 0, "top": 0, "width": 8, "height": 8}
    with pytest.raises(FileNotFoundError):
        renderer.thumb(box, frames_dir=frames)
    with pytest.raises(FileNotFoundError):
        renderer.crop({**box, "cx": 4, "cy": 4}, frames_dir=frames)
    with pytest.raises(FileNotFoundError):
        renderer.frame("f00.png", frames_dir=None)
