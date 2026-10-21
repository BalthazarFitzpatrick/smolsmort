"""best-effort machine metadata for interpreting run wall-clock measurements"""

from __future__ import annotations

import ctypes
import os
import platform
import subprocess
from importlib import metadata
from pathlib import Path


def _read_probe(probe):
    try:
        return probe()
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def _probe_platform(system):
    """keep optional operating-system probes behind one seam"""
    details = {
        "cpu_brand": None,
        "physical_cores": None,
        "performance_cores": None,
        "efficiency_cores": None,
        "memory_bytes": None,
    }
    if system == "Darwin":

        def read_sysctl(key):
            return subprocess.check_output(
                ["/usr/sbin/sysctl", "-n", key], text=True, stderr=subprocess.DEVNULL, timeout=1
            ).strip()

        details["cpu_brand"] = _read_probe(lambda: read_sysctl("machdep.cpu.brand_string"))
        for field, key in (
            ("physical_cores", "hw.physicalcpu"),
            ("performance_cores", "hw.perflevel0.physicalcpu"),
            ("efficiency_cores", "hw.perflevel1.physicalcpu"),
            ("memory_bytes", "hw.memsize"),
        ):
            details[field] = _read_probe(lambda key=key: int(read_sysctl(key)))
    elif system == "Linux":
        cpuinfo = _read_probe(lambda: Path("/proc/cpuinfo").read_text())
        if cpuinfo:
            cores = set()
            for processor in cpuinfo.split("\n\n"):
                fields = {
                    key.strip(): value.strip()
                    for line in processor.splitlines()
                    if ":" in line
                    for key, value in [line.split(":", 1)]
                }
                details["cpu_brand"] = details["cpu_brand"] or fields.get("model name")
                if fields.get("physical id") and fields.get("core id"):
                    cores.add((fields["physical id"], fields["core id"]))
            details["physical_cores"] = len(cores) or None
        details["memory_bytes"] = _read_probe(
            lambda: os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
        )
    elif system == "Windows":
        details["cpu_brand"] = _read_probe(platform.processor)

        def read_memory():
            class MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("length", ctypes.c_uint32),
                    ("load", ctypes.c_uint32),
                    ("total_physical", ctypes.c_uint64),
                    ("available_physical", ctypes.c_uint64),
                    ("total_pagefile", ctypes.c_uint64),
                    ("available_pagefile", ctypes.c_uint64),
                    ("total_virtual", ctypes.c_uint64),
                    ("available_virtual", ctypes.c_uint64),
                    ("available_extended_virtual", ctypes.c_uint64),
                ]

            status = MemoryStatus()
            status.length = ctypes.sizeof(status)
            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            if not kernel.GlobalMemoryStatusEx(ctypes.byref(status)):
                raise OSError("could not read windows memory")
            return status.total_physical

        details["memory_bytes"] = _read_probe(read_memory)
    return details


def describe_machine(nthread, max_concurrent_families):
    """record hardware and installed versions without importing model libraries"""
    versions = {}
    for package in ("xgboost", "lightgbm", "statsforecast", "numpy"):
        try:
            versions[package] = metadata.version(package)
        except (metadata.PackageNotFoundError, OSError, ValueError):
            versions[package] = None
    return {
        "platform": _read_probe(platform.platform),
        "machine_arch": _read_probe(platform.machine),
        "logical_cores": _read_probe(os.cpu_count),
        **_probe_platform(_read_probe(platform.system)),
        "nthread": nthread,
        "max_concurrent_families": max_concurrent_families,
        "versions": versions,
    }
