"""Pictures from stable-diffusion.cpp's sd-server, or another server with the same API.

sd-server keeps its model in its own memory, apart from Ollama's. So the helper
starts it only when a picture is asked for, after the chat model has left
memory, and stops it before the next chat model loads, or when the keepalive
runs out. That keeps one large model in memory, the same rule as Ollama's.

A server the user started themselves (sd-server, AUTOMATIC1111, Forge, Draw
Things) is reached by its address with the same /sdapi/v1 calls. Its memory is
its own, so the helper cannot free it.
"""

from __future__ import annotations

import json
import os
import platform
import shlex
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from helper import keepalive, store

PROGRAM = "sd-server"
LOAD_TIMEOUT = 900          # seconds a model may take to load before giving up
POLL = 0.7                  # seconds between progress checks
NAME_CHARS = set("abcdefghijklmnopqrstuvwxyz0123456789-_.")


class PictureError(Exception):
    """Worded for the user."""


# --- settings ---------------------------------------------------------------------

def program_dir() -> Path:
    """Where the setup steps say to unzip sd-server, so no path has to be typed."""
    return store.home() / "sd-server"


def find_program(configured: str = "") -> str | None:
    if configured:
        path = Path(configured).expanduser()
        return str(path) if path.is_file() else None
    found = shutil.which(PROGRAM)
    if found:
        return found
    name = PROGRAM + (".exe" if os.name == "nt" else "")
    folder = program_dir()
    # The zip may unpack straight into the folder or into one folder inside it.
    for path in [folder / name, *sorted(folder.glob("*/" + name))]:
        if path.is_file():
            _let_run(path)
            return str(path)
    return None


def _let_run(path: Path) -> None:
    """A program unzipped from a download is marked by macOS, which then refuses to run it
    (and the library beside it), and a zip can lose the run permission."""
    if path in _let_run.done:
        return
    _let_run.done.add(path)
    try:
        if not os.access(path, os.X_OK):
            path.chmod(path.stat().st_mode | 0o111)
        if sys.platform == "darwin":
            subprocess.run(["xattr", "-dr", "com.apple.quarantine", str(path.parent)],
                           capture_output=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        pass


_let_run.done = set()


def release_hint() -> str:
    """Which file on stable-diffusion.cpp's releases page fits this computer."""
    machine = platform.machine().lower()
    if sys.platform == "darwin":
        return "the file with Darwin-macOS and arm64 in its name" if machine == "arm64" else \
            "none: the releases are for Apple Silicon Macs, so on an Intel Mac sd-server has to be built from source"
    if os.name == "nt":
        return "win-vulkan-x64 (any graphics card), win-cuda12-x64 (NVIDIA), or win-cpu-x64 (no graphics card)"
    return "Linux…x86_64-vulkan (any graphics card), …x86_64-rocm (AMD), or Linux…x86_64 (CPU only)"


def parse_models(text: str) -> tuple[list[dict], list[str]]:
    """One model per line: a name, a colon, then sd-server's model arguments.
    Returns the models and a plain note for each line that could not be read."""
    models, problems = [], []
    seen = set()
    for number, line in enumerate((text or "").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, colon, rest = line.partition(":")
        name = name.strip().lower()
        if not colon or not name or not set(name) <= NAME_CHARS:
            problems.append(f"Line {number} needs a name (letters, digits, - _ .), a colon, then the arguments.")
            continue
        try:
            args = shlex.split(rest, posix=os.name != "nt")
        except ValueError:
            problems.append(f"Line {number} has an unclosed quote.")
            continue
        if not args:
            problems.append(f"Line {number} has no model arguments after the colon.")
            continue
        if any(arg in ("--listen-ip", "--listen-port") for arg in args):
            problems.append(f"Line {number}: leave out --listen-ip and --listen-port; the helper sets them.")
            continue
        if name in seen:
            problems.append(f"Line {number}: the name {name} is used twice.")
            continue
        seen.add(name)
        # Windows splitting keeps the quotes around a path with spaces.
        args = [arg[1:-1] if len(arg) > 1 and arg[0] == arg[-1] == '"' else arg for arg in args]
        models.append({"name": name, "args": [os.path.expanduser(arg) if arg.startswith("~") else arg for arg in args]})
    return models, problems


def model_files(args: list[str]) -> list[Path]:
    return [Path(arg) for arg in args if not arg.startswith("-") and Path(arg).is_file()]


def model_size(args: list[str]) -> int:
    """Bytes of the model files named in the arguments: about what they take in memory."""
    return sum(path.stat().st_size for path in model_files(args))


def keepalive_seconds(value: str) -> float:
    # sd-server holds its model in memory, so "never unload" is not honoured here.
    found = keepalive.seconds(value)
    return 300.0 if found == float("inf") else found


# --- the /sdapi/v1 calls ---------------------------------------------------------------

def _call(base: str, path: str, payload: dict | None = None, timeout: float | None = 10):
    request = urllib.request.Request(base.rstrip("/") + path, method="POST" if payload is not None else "GET",
                                     data=json.dumps(payload).encode() if payload is not None else None,
                                     headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode() or "{}")
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace").strip()
        try:
            detail = json.loads(detail).get("error") or json.loads(detail).get("detail") or detail
        except (ValueError, AttributeError):
            pass
        raise PictureError(f"The picture server refused the request: {str(detail)[:300]}") from error
    except (urllib.error.URLError, OSError, ValueError) as error:
        raise PictureError(f"The picture server at {base} is not answering.") from error


def _answers(base: str) -> bool:
    """Whether a server is listening. Any HTTP reply counts, even an error, since not
    every build has every /sdapi/v1 call."""
    try:
        with urllib.request.urlopen(base + "/sdapi/v1/sd-models", timeout=2):
            return True
    except urllib.error.HTTPError:
        return True
    except (urllib.error.URLError, OSError):
        return False


def current_model(base: str) -> str:
    """The model an outside server has loaded, for the picker."""
    options = _call(base, "/sdapi/v1/options", timeout=5)
    name = str(options.get("sd_model_checkpoint") or "").strip()
    return name or "picture server"


def txt2img(base: str, request: dict, on_progress, stopped) -> str:
    """Make one picture and return it as base64. on_progress(step, total) is called while
    it draws; stopped() returning True ends the wait (the caller cancels the work)."""
    result: dict = {}

    def run():
        try:
            result["body"] = _call(base, "/sdapi/v1/txt2img", {
                "prompt": request["prompt"], "width": request["width"], "height": request["height"],
                "seed": request["seed"], "batch_size": 1, "n_iter": 1,
                **({"steps": request["steps"]} if request["steps"] else {}),
            }, timeout=None)
        except PictureError as error:
            result["error"] = error

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    while worker.is_alive():
        worker.join(POLL)
        if stopped():
            return ""
        try:
            state = _call(base, "/sdapi/v1/progress?skip_current_image=true", timeout=3).get("state") or {}
            on_progress(int(state.get("sampling_step") or 0), int(state.get("sampling_steps") or 0))
        except PictureError:
            pass
    if "error" in result:
        raise result["error"]
    images = (result.get("body") or {}).get("images") or []
    if not images or not isinstance(images[0], str):
        raise PictureError("The picture server finished without sending a picture.")
    return images[0].split(",", 1)[-1]


def interrupt(base: str) -> None:
    try:
        _call(base, "/sdapi/v1/interrupt", {}, timeout=5)
    except PictureError:
        pass


# --- the sd-server the helper runs -----------------------------------------------------------

def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


class Managed:
    """At most one sd-server, holding one model, started and stopped by the helper."""

    def __init__(self, keep_alive: str = "5m") -> None:
        self._lock = threading.RLock()
        self._process: subprocess.Popen | None = None
        self._timer: threading.Timer | None = None
        self.model: str | None = None
        self.base = ""
        self.keep_alive = keepalive_seconds(keep_alive)
        self.stops_at: float | None = None

    def log_path(self) -> Path:
        return store.home() / "sd-server.log"

    def running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def ensure(self, program: str, model: dict) -> str:
        """Start sd-server with this model unless it already runs it. Returns its address."""
        with self._lock:
            self._cancel_timer()
            if self.running() and self.model == model["name"]:
                return self.base
            self.stop()
            port = _free_port()
            folder = store.home()
            folder.mkdir(parents=True, exist_ok=True)
            with open(self.log_path(), "wb") as log:
                try:
                    self._process = subprocess.Popen(
                        [program, *model["args"], "--listen-ip", "127.0.0.1", "--listen-port", str(port)],
                        stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                    )
                except OSError as error:
                    raise PictureError(f"sd-server could not be started: {error.strerror or error}") from error
            self.model = model["name"]
            self.base = f"http://127.0.0.1:{port}"
        deadline = time.monotonic() + LOAD_TIMEOUT
        while time.monotonic() < deadline:
            if not self.running():
                tail = self._log_tail()
                self.stop()
                raise PictureError("sd-server stopped while loading the model." + (f" It said: {tail}" if tail else ""))
            if _answers(self.base):
                return self.base
            time.sleep(0.5)
        self.stop()
        raise PictureError("sd-server took too long to load the model.")

    def touch(self) -> None:
        """Keep it for the keepalive after a picture, then stop it, like Ollama does."""
        with self._lock:
            self._cancel_timer()
            if self.running():
                self._timer = threading.Timer(self.keep_alive, self.stop)
                self._timer.daemon = True
                self._timer.start()
                self.stops_at = time.time() + self.keep_alive

    def stop(self) -> None:
        with self._lock:
            self._cancel_timer()
            process, self._process = self._process, None
            self.model = None
            self.base = ""
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(10)

    def _cancel_timer(self) -> None:
        self.stops_at = None
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None

    def _log_tail(self) -> str:
        try:
            lines = self.log_path().read_text(encoding="utf-8", errors="replace").strip().splitlines()
        except OSError:
            return ""
        wanted = [line for line in lines if "error" in line.lower() or "fail" in line.lower()] or lines
        return " ".join(wanted[-2:])[:300]
