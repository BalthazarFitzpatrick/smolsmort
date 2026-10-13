"""portable parent supervision and bounded cleanup of owned processes"""

from __future__ import annotations

import json
import multiprocessing
import os
import signal
import subprocess
import sys
import threading
import time
from contextlib import suppress
from pathlib import Path


def is_process_alive(pid):
    if sys.platform == "win32":
        from smolsmort.forecast.runs import _windows_alive

        return _windows_alive(pid)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def terminate_process(pid, *, force=False):
    """windows os.kill cannot query or gracefully stop a reattached process"""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.OpenProcess(0x0001, False, pid)
        if handle:
            try:
                kernel.TerminateProcess(handle, 1)
            finally:
                kernel.CloseHandle(handle)
    else:
        with suppress(ProcessLookupError):
            os.kill(pid, signal.SIGKILL if force else signal.SIGTERM)


def terminate_run_tree(pid, *, force=False):
    """the launcher's new session owns native refit descendants without recorded pids"""
    if sys.platform == "win32":
        command = ["taskkill", "/PID", str(pid), "/T", "/F"]
        subprocess.run(command, capture_output=True, timeout=5, check=False)
    else:
        with suppress(ProcessLookupError):
            os.killpg(pid, signal.SIGKILL if force else signal.SIGTERM)


def is_run_tree_alive(pid):
    if sys.platform == "win32":
        return is_process_alive(pid)
    try:
        os.killpg(pid, 0)
    except ProcessLookupError:
        return False
    return True


def read_run_pids(folder):
    pids = set()
    for path in (
        folder / "coordinator.json",
        *folder.glob("families/*/status.json"),
        *folder.glob("families/*/candidate_pid.json"),
    ):
        try:
            data = json.loads(path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            continue
        values = [data.get("pid"), data.get("finish_pid")]
        values.extend(state.get("pid") for state in data.get("families", {}).values())
        pids.update(pid for pid in values if type(pid) is int and pid > 0)
    return pids


def stop_run_processes(folder, *, alive):
    """prefer coordinator cleanup; escalate owned pids when a native fit blocks it"""
    from smolsmort.forecast.runs import RunError

    pids = set()
    coordinator_pid = int((folder / "pid").read_text())

    def pending():
        pids.update(read_run_pids(folder))
        coordinator_alive = alive()
        return (
            coordinator_alive
            or is_run_tree_alive(coordinator_pid)
            or any(is_process_alive(pid) for pid in pids)
        )

    def wait(seconds):
        until = time.monotonic() + seconds
        while pending() and time.monotonic() < until:
            time.sleep(0.02)
        return pending()

    if not wait(2):
        return
    for force in (False, True):
        for pid in pids:
            if is_process_alive(pid):
                terminate_process(pid, force=force)
        # children supervise their own descendants; let the parent reap family workers first
        wait(0.5)
        if is_run_tree_alive(coordinator_pid):
            terminate_run_tree(coordinator_pid, force=force)
        if not wait(1):
            return
    raise RunError("run workers did not exit after cancellation")


def stop_children():
    children = multiprocessing.active_children()
    for child in children:
        child.terminate()
    until = time.monotonic() + 0.2
    for child in children:
        child.join(timeout=max(0, until - time.monotonic()))
        if child.is_alive():
            child.kill()
    until = time.monotonic() + 0.2
    for child in children:
        child.join(timeout=max(0, until - time.monotonic()))


def watch_parent(parent_pid, *, stop_path=None):
    """the watchdog runs while a native fit blocks the scoring thread"""

    def watch():
        from smolsmort.forecast.runs import _windows_alive

        while True:
            alive = os.getppid() == parent_pid
            if alive and sys.platform == "win32":
                alive = _windows_alive(parent_pid)
            if not alive or (stop_path and Path(stop_path).exists()):
                stop_children()
                os._exit(1)
            time.sleep(0.02)

    threading.Thread(target=watch, daemon=True).start()
