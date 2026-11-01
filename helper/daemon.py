"""Running HikariLM in the background, so it keeps running after the terminal is closed.

hikarilm start starts it and returns, hikarilm stop stops it, and Update in Settings (or
hikarilm update) pulls the new code and restarts it. What it prints goes to ~/.hikarilm/hikarilm.log, which keeps only
its last MAX_LINES lines.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from helper import store

ROOT = Path(__file__).resolve().parent.parent
MAX_LINES = 2000
STOP_SECONDS = 20


def log_path() -> Path:
    return store.home() / "hikarilm.log"


def pid_path() -> Path:
    return store.home() / "hikarilm.pid"


class CappedLog:
    """A log file that never holds more than max_lines lines: past that, the oldest half goes."""

    def __init__(self, path: Path, max_lines: int = MAX_LINES) -> None:
        self.path = path
        self.max_lines = max_lines
        self._lock = threading.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.lines = path.read_bytes().count(b"\n")
        except FileNotFoundError:
            self.lines = 0
        self._handle = open(path, "ab")
        if self.lines > max_lines:
            self._trim()

    def write(self, data: bytes) -> None:
        with self._lock:
            self._handle.write(data)
            self._handle.flush()
            self.lines += data.count(b"\n")
            if self.lines > self.max_lines:
                self._trim()

    def close(self) -> None:
        with self._lock:
            self._handle.close()

    def _trim(self) -> None:
        self._handle.close()
        kept = self.path.read_bytes().splitlines(keepends=True)[-(self.max_lines // 2):]
        spare = self.path.with_name(self.path.name + ".new")
        spare.write_bytes(b"".join(kept))
        os.replace(spare, self.path)
        self.lines = len(kept)
        self._handle = open(self.path, "ab")


def capture_output(log: CappedLog) -> None:
    """Send everything this process prints into the log, at the file-descriptor level, so the
    Python warnings and anything a child process inherits land there too."""
    read, write = os.pipe()
    sys.stdout.flush()
    sys.stderr.flush()
    os.dup2(write, 1)
    os.dup2(write, 2)
    os.close(write)

    def pump() -> None:
        with open(read, "rb", buffering=0) as source:
            while True:
                chunk = source.read(65536)
                if not chunk:
                    return
                try:
                    log.write(chunk)
                except OSError:
                    # A full disk must not stop the helper: a pipe nobody reads would.
                    pass

    threading.Thread(target=pump, name="log", daemon=True).start()


def write_pid() -> None:
    pid_path().write_text(str(os.getpid()), encoding="utf-8")


def forget_pid() -> None:
    try:
        if pid_path().read_text(encoding="utf-8").strip() == str(os.getpid()):
            pid_path().unlink()
    except OSError:
        pass


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def is_hikarilm(pid: int) -> bool:
    """Whether this process is HikariLM's start.py, so a pid reused by another program is left alone."""
    try:
        command = subprocess.run(["ps", "-p", str(pid), "-o", "command="], capture_output=True,
                                 text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return False
    return "start.py" in command


def saved_pid() -> int | None:
    try:
        pid = int(pid_path().read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None
    return pid if alive(pid) and is_hikarilm(pid) else None


def wait_gone(pid: int, seconds: float) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if not alive(pid):
            return True
        time.sleep(0.2)
    return not alive(pid)


def stop(pid: int, seconds: float = STOP_SECONDS) -> bool:
    """Ask HikariLM to stop (it unloads its models first) and wait for it. True once it has."""
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return True
    return wait_gone(pid, seconds)


def spawn(arguments: list[str], output: Path | None = None) -> subprocess.Popen:
    """Start start.py in a session of its own, so closing the terminal does not stop it."""
    sink = open(output, "ab") if output else subprocess.DEVNULL
    try:
        return subprocess.Popen([sys.executable, str(ROOT / "start.py"), *arguments], cwd=ROOT,
                                stdin=subprocess.DEVNULL, stdout=sink, stderr=sink, start_new_session=True)
    finally:
        if output:
            sink.close()


def restart_later(host: str, port: int) -> None:
    """Start a copy that waits for this one to end, then end this one. The new copy runs the
    first-start steps again, in case the update needs new packages."""
    spawn(["--no-browser", "--host", host, "--port", str(port), "--after", str(os.getpid())], output=log_path())
    threading.Timer(0.5, os.kill, (os.getpid(), signal.SIGTERM)).start()
