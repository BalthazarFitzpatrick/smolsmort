"""the device choice: cuda first, then mps, then cpu"""

from types import SimpleNamespace

import pytest

from smolsmort.torch_device import default_device


def fake_torch(cuda: bool, mps: bool):
    return SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: cuda),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps)),
    )


@pytest.mark.parametrize(
    ("cuda", "mps", "device"),
    [(True, False, "cuda"), (True, True, "cuda"), (False, True, "mps"), (False, False, "cpu")],
)
def test_default_device_prefers_cuda_then_mps(cuda, mps, device):
    assert default_device(fake_torch(cuda, mps)) == device
