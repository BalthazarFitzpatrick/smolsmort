"""where browser tests write their screenshots: a pytest temp dir, never the repo"""

from pathlib import Path

_root: Path | None = None


def set_shot_root(root: Path) -> None:
    global _root
    _root = root


def shot_dir(*parts: str) -> Path:
    if _root is None:
        raise RuntimeError("screenshot root not set; conftest sets it per session")
    folder = _root.joinpath(*parts)
    folder.mkdir(parents=True, exist_ok=True)
    return folder
