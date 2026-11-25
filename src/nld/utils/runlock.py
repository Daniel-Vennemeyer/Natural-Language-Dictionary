"""Prevent concurrent writes to the same output and check available GPU memory."""
from __future__ import annotations

import atexit
import os
import signal
import sys
import time


def _forced():
    return os.environ.get("RUNLOCK_FORCE", "") not in ("", "0")


def acquire(name: str, lockdir: str = "outputs/locks") -> str | None:
    if _forced():
        print("[lock] RUNLOCK_FORCE set -- skipping single-instance lock", flush=True)
        return None
    os.makedirs(lockdir, exist_ok=True)
    path = os.path.join(lockdir, f"{name}.lock")
    while True:
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"{os.getpid()}\n{' '.join(sys.argv)}\n{time.ctime()}\n".encode())
            os.close(fd)
            break
        except FileExistsError:
            try:
                pid = int(open(path).readline().strip())
            except (ValueError, OSError):
                os.remove(path)                                   # unreadable -> stale
                continue
            try:
                os.kill(pid, 0)                                   # alive?
            except ProcessLookupError:
                print(f"[lock] removing stale lock (dead PID {pid})", flush=True)
                os.remove(path)
                continue
            except PermissionError:
                pass                                              # someone else's live pid -> treat as alive
            info = open(path).read().rstrip()
            sys.exit(f"[lock] Another '{name}' run is already active:\n"
                     f"{info}")
    print(f"[lock] acquired {path} (pid {os.getpid()})", flush=True)

    def _release(*_a):
        try:
            os.remove(path)
        except OSError:
            pass

    atexit.register(_release)
    for sg in (signal.SIGINT, signal.SIGTERM):
        prev = signal.getsignal(sg)

        def _h(s, f, prev=prev):
            _release()
            if callable(prev):
                prev(s, f)
            else:
                sys.exit(128 + s)
        signal.signal(sg, _h)
    return path


def gpu_guard(min_free_gb: float = 18.0):
    """Refuse to start unless the visible GPU has headroom for the judge + training."""
    if _forced():
        return
    import torch
    if not torch.cuda.is_available():
        return
    free, total = torch.cuda.mem_get_info()
    if free / 1e9 < min_free_gb:
        sys.exit(f"[lock] Visible GPU has {free / 1e9:.1f} GB free of "
                 f"{total / 1e9:.0f} GB (requires at least {min_free_gb:.0f} GB).")
    print(f"[lock] GPU headroom OK: {free / 1e9:.1f} GB free", flush=True)
