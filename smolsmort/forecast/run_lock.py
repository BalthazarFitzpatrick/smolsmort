"""one forecast run at a time across processes, released by the os on exit"""

from __future__ import annotations

import errno
import os
import sys
from pathlib import Path


class RunLock:
    def __init__(self):
        configured = os.environ.get("SMOLSMORT_RUN_LOCK")
        path = (
            Path(configured)
            if configured
            else Path.home() / ".cache" / "smolsmort" / "forecast.lock"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        self.fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
        os.set_inheritable(self.fd, False)
        if hasattr(os, "register_at_fork"):
            os.register_at_fork(after_in_child=self.close)

    def try_acquire(self) -> bool:
        try:
            if sys.platform == "win32":
                import msvcrt

                os.lseek(self.fd, 0, os.SEEK_SET)
                msvcrt.locking(self.fd, msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno not in (errno.EACCES, errno.EAGAIN):
                raise
            return False
        return True

    def close(self):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
