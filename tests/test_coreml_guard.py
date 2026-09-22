"""the "coreml" runtime refuses by name when coremltools is absent or is a shell without its
native libraries - the state a python without coremltools wheels (3.14 with 9.0) is left in"""

from __future__ import annotations

import sys
import types

import pytest

from smolsmort.detect.model import build_model
from smolsmort.detect.runtime import RuntimeError_, native_libraries_present
from smolsmort.detect.train import load, save

torch = pytest.importorskip("torch")


@pytest.fixture
def weights(tmp_path):
    return save(build_model(classes=2, downscale=2, channels=8), tmp_path / "w.pt")


def test_missing_coremltools_is_named(weights, monkeypatch):
    monkeypatch.setitem(sys.modules, "coremltools", None)  # import raises ImportError
    with pytest.raises(RuntimeError_, match="needs coremltools"):
        load(weights, runtime="coreml")


def test_a_shell_without_native_libraries_names_the_interpreter(weights, monkeypatch):
    shell = types.ModuleType("coremltools")
    shell.__version__ = "9.0"
    shell.__path__ = []  # a package with nothing compiled under it
    # a real coremltools imported earlier in this process would answer find_spec from
    # sys.modules; forget its submodules so the check has to look under the shell's path
    for name in [n for n in sys.modules if n.startswith("coremltools.")]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(sys.modules, "coremltools", shell)
    assert not native_libraries_present(shell)
    version = f"{sys.version_info[0]}.{sys.version_info[1]}"
    with pytest.raises(RuntimeError_, match=f"python {version} has no native libraries"):
        load(weights, runtime="coreml")
