"""machine probes remain optional across supported operating systems"""

from __future__ import annotations

import subprocess
from types import SimpleNamespace

import pytest

from smolsmort.forecast import machine


@pytest.fixture
def mock_platform(monkeypatch):
    monkeypatch.setattr(machine.platform, "platform", lambda: "test-platform")
    monkeypatch.setattr(machine.platform, "machine", lambda: "test-arch")
    monkeypatch.setattr(machine.os, "cpu_count", lambda: 12)
    monkeypatch.setattr(machine.metadata, "version", lambda package: f"{package}-version")

    def select_system(system):
        monkeypatch.setattr(machine.platform, "system", lambda: system)

    return select_system


def test_darwin_machine_records_hardware_threads_and_versions(mock_platform, monkeypatch):
    mock_platform("Darwin")
    values = {
        "machdep.cpu.brand_string": "test apple cpu",
        "hw.physicalcpu": "12",
        "hw.perflevel0.physicalcpu": "8",
        "hw.perflevel1.physicalcpu": "4",
        "hw.memsize": str(32 * 1024**3),
    }

    def read_sysctl(command, **kwargs):
        assert command[:2] == ["/usr/sbin/sysctl", "-n"]
        assert kwargs == {"text": True, "stderr": subprocess.DEVNULL, "timeout": 1}
        return values[command[2]] + "\n"

    monkeypatch.setattr(machine.subprocess, "check_output", read_sysctl)
    assert machine.describe_machine(10, 1) == {
        "platform": "test-platform",
        "machine_arch": "test-arch",
        "logical_cores": 12,
        "cpu_brand": "test apple cpu",
        "physical_cores": 12,
        "performance_cores": 8,
        "efficiency_cores": 4,
        "memory_bytes": 32 * 1024**3,
        "nthread": 10,
        "max_concurrent_families": 1,
        "versions": {
            package: f"{package}-version"
            for package in ("xgboost", "lightgbm", "statsforecast", "numpy")
        },
    }


def test_linux_machine_counts_unique_cores_across_sockets(mock_platform, monkeypatch):
    mock_platform("Linux")
    processors = [("0", "0"), ("0", "0"), ("0", "1"), ("1", "0")]
    cpuinfo = "\n\n".join(
        f"model name : test cpu\nphysical id : {socket}\ncore id : {core}"
        for socket, core in processors
    )
    monkeypatch.setattr(machine.Path, "read_text", lambda path: cpuinfo)
    monkeypatch.setattr(
        machine.os,
        "sysconf",
        lambda key: {"SC_PHYS_PAGES": 1000, "SC_PAGE_SIZE": 4096}[key],
        raising=False,
    )
    result = machine.describe_machine(12, 4)
    assert result["cpu_brand"] == "test cpu"
    assert result["physical_cores"] == 3
    assert result["memory_bytes"] == 4096000
    assert result["performance_cores"] is None
    assert result["efficiency_cores"] is None
    assert result["max_concurrent_families"] == 4


def test_linux_missing_topology_stays_unknown(mock_platform, monkeypatch):
    mock_platform("Linux")
    monkeypatch.setattr(machine.Path, "read_text", lambda path: "model name : test cpu\n")
    monkeypatch.setattr(machine.os, "sysconf", lambda key: 1, raising=False)
    assert machine.describe_machine(2, 1)["physical_cores"] is None


def test_windows_machine_records_brand_and_memory(mock_platform, monkeypatch):
    mock_platform("Windows")
    monkeypatch.setattr(machine.platform, "processor", lambda: "test windows cpu")

    def read_memory(pointer):
        status = pointer._obj
        assert status.length == machine.ctypes.sizeof(status) == 64
        status.total_physical = 16 * 1024**3
        return 1

    monkeypatch.setattr(
        machine.ctypes,
        "WinDLL",
        lambda name, **kwargs: SimpleNamespace(GlobalMemoryStatusEx=read_memory),
        raising=False,
    )
    result = machine.describe_machine(8, 1)
    assert result["cpu_brand"] == "test windows cpu"
    assert result["memory_bytes"] == 16 * 1024**3
    assert result["physical_cores"] is None


@pytest.mark.parametrize(
    "error",
    [
        OSError("unavailable"),
        subprocess.CalledProcessError(1, "sysctl"),
        subprocess.TimeoutExpired("sysctl", 1),
        UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid"),
    ],
)
def test_darwin_probe_failures_do_not_fail_run(mock_platform, monkeypatch, error):
    mock_platform("Darwin")

    def fail_probe(*args, **kwargs):
        raise error

    monkeypatch.setattr(machine.subprocess, "check_output", fail_probe)
    result = machine.describe_machine(12, 1)
    for field in (
        "cpu_brand",
        "physical_cores",
        "performance_cores",
        "efficiency_cores",
        "memory_bytes",
    ):
        assert result[field] is None
    assert result["nthread"] == 12


def test_one_darwin_probe_failure_preserves_other_fields(mock_platform, monkeypatch):
    mock_platform("Darwin")

    def read_sysctl(command, **kwargs):
        return "unknown" if command[-1] == "hw.perflevel1.physicalcpu" else "4"

    monkeypatch.setattr(machine.subprocess, "check_output", read_sysctl)
    result = machine.describe_machine(4, 1)
    assert result["physical_cores"] == 4
    assert result["performance_cores"] == 4
    assert result["efficiency_cores"] is None


def test_linux_probe_failures_do_not_fail_run(mock_platform, monkeypatch):
    mock_platform("Linux")

    def fail_probe(*args):
        raise OSError("unavailable")

    monkeypatch.setattr(machine.Path, "read_text", fail_probe)
    monkeypatch.setattr(machine.os, "sysconf", fail_probe, raising=False)
    result = machine.describe_machine(1, 1)
    assert result["cpu_brand"] is None
    assert result["physical_cores"] is None
    assert result["memory_bytes"] is None


def test_windows_probe_failure_does_not_fail_run(mock_platform, monkeypatch):
    mock_platform("Windows")
    monkeypatch.setattr(machine.platform, "processor", lambda: "test windows cpu")
    monkeypatch.setattr(
        machine.ctypes,
        "WinDLL",
        lambda name, **kwargs: SimpleNamespace(GlobalMemoryStatusEx=lambda pointer: 0),
        raising=False,
    )
    result = machine.describe_machine(1, 1)
    assert result["cpu_brand"] == "test windows cpu"
    assert result["memory_bytes"] is None


def test_unknown_platform_and_missing_packages_do_not_fail_run(mock_platform, monkeypatch):
    mock_platform("unknown")

    def missing_package(package):
        raise machine.metadata.PackageNotFoundError(package)

    monkeypatch.setattr(machine.metadata, "version", missing_package)
    result = machine.describe_machine(1, 1)
    assert result["cpu_brand"] is None
    assert result["memory_bytes"] is None
    assert all(version is None for version in result["versions"].values())


def test_common_probe_failures_do_not_fail_run(mock_platform, monkeypatch):
    mock_platform("unknown")

    def fail_probe():
        raise OSError("unavailable")

    for name in ("platform", "machine", "system"):
        monkeypatch.setattr(machine.platform, name, fail_probe)
    monkeypatch.setattr(machine.os, "cpu_count", fail_probe)
    result = machine.describe_machine(2, 1)
    assert result["platform"] is None
    assert result["machine_arch"] is None
    assert result["logical_cores"] is None
    assert result["nthread"] == 2
