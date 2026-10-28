"""One-time picture setup: download sd-server and FLUX.2 klein, then add the model.

The page's setup window starts this and watches it. Everything goes into the app
folder (sd-server/ and models/), so nothing else on the computer changes. Files
already there are kept, so a stopped or failed setup picks up where it left
off, file by file. Downloads stream to disk in small pieces and never sit in memory.
"""

from __future__ import annotations

import ctypes.util
import json
import os
import platform
import shutil
import sys
import threading
import time
import urllib.request
import zipfile
from pathlib import Path

from helper import context, imagegen, sdserver

RELEASES = "https://api.github.com/repos/leejet/stable-diffusion.cpp/releases/latest"
CHUNK = 1 << 20
SPARE = 1 << 30           # disk space left over after everything is downloaded
KLEIN_BYTES = 5_000_000_000  # about what the three FLUX.2 klein files take
NAME = "flux2-klein"


class SetupError(Exception):
    """Worded for the user."""


def _release_asset(assets: list[dict]) -> dict:
    """The sd-server zip that runs on this computer."""
    machine = platform.machine().lower()
    if sys.platform == "darwin":
        if machine != "arm64":
            raise SetupError("sd-server has no ready-made download for Intel Macs, so it has to be built from "
                             "github.com/leejet/stable-diffusion.cpp.")
        wanted = [("darwin", "arm64")]
    elif os.name == "nt":
        wanted = [("win-vulkan-x64",), ("win-cpu-x64",)]
    elif machine in ("x86_64", "amd64"):
        # The Vulkan build uses any graphics card, but needs Vulkan to be installed.
        cpu = ("linux", "x86_64.zip")
        wanted = [("linux", "x86_64-vulkan"), cpu] if ctypes.util.find_library("vulkan") else [cpu]
    else:
        raise SetupError(f"sd-server has no ready-made download for {platform.system()} on {machine}.")
    for parts in wanted:
        for asset in assets:
            name = asset.get("name", "").lower()
            if name.endswith(".zip") and all(part in name for part in parts):
                return asset
    raise SetupError("The latest sd-server release has no download for this computer.")


class Setup:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.state = {"running": False, "file": "", "done": 0, "total": 0, "error": "", "finished": False}

    def status(self) -> dict:
        program = sdserver.find_program(context.load_settings()["pictures"]["program"])
        with self._lock:
            out = dict(self.state)
        out.update({"program": program, "program_dir": str(sdserver.program_dir()), "klein": imagegen.klein_setup()})
        return out

    def start(self) -> None:
        with self._lock:
            if self.state["running"]:
                return
            self._stop.clear()
            self.state = {"running": True, "file": "", "done": 0, "total": 0, "error": "", "finished": False}
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _set(self, **changes) -> None:
        with self._lock:
            self.state.update(changes)

    def _run(self) -> None:
        try:
            self._setup()
            if not self._stop.is_set():
                add_model()
                self._set(finished=True)
        except SetupError as error:
            self._set(error=str(error))
        except OSError as error:
            self._set(error=f"The download failed: {getattr(error, 'reason', None) or error.strerror or error}. "
                            "Press Set up again to carry on from where it stopped.")
        finally:
            self._set(running=False, file="", done=0, total=0)

    def _setup(self) -> None:
        settings = context.load_settings()
        need_program = sdserver.find_program(settings["pictures"]["program"]) is None
        files = [item for item in imagegen.klein_setup()["files"] if not item["found"]]
        folder = imagegen.models_dir()
        asset = None
        if need_program:
            with urllib.request.urlopen(urllib.request.Request(RELEASES, headers={"User-Agent": "LightUsagi"}),
                                        timeout=30) as response:
                asset = _release_asset(json.loads(response.read().decode()).get("assets") or [])
        needed = (asset["size"] if asset else 0) + (KLEIN_BYTES if files else 0) + SPARE
        if shutil.disk_usage(folder).free < needed:
            raise SetupError(f"There is not enough free disk space. Setting up needs about "
                             f"{needed / 1e9:.0f} GB free in {folder}.")
        if asset:
            target = sdserver.program_dir()
            target.mkdir(parents=True, exist_ok=True)
            archive = target / asset["name"]
            if not self._download(asset["browser_download_url"], archive, "sd-server"):
                return
            with zipfile.ZipFile(archive) as unzipped:
                unzipped.extractall(target)
            archive.unlink()
            if sdserver.find_program("") is None:
                raise SetupError(f"The sd-server download did not contain the program. Look in {target}.")
        for item in files:
            if not self._download(item["url"], folder / item["name"], item["name"]):
                return

    def _download(self, url: str, path: Path, label: str) -> bool:
        """Stream url to path. False if stopped. A half-finished file is never left under its real name."""
        partial = path.with_name(path.name + ".part")
        request = urllib.request.Request(url, headers={"User-Agent": "LightUsagi"})
        with urllib.request.urlopen(request, timeout=60) as response, open(partial, "wb") as handle:
            total = int(response.headers.get("Content-Length") or 0)
            self._set(file=label, done=0, total=total)
            done, shown = 0, 0.0
            while True:
                if self._stop.is_set():
                    break
                piece = response.read(CHUNK)
                if not piece:
                    break
                handle.write(piece)
                done += len(piece)
                if time.monotonic() - shown > 0.5:
                    self._set(done=done)
                    shown = time.monotonic()
        if self._stop.is_set():
            partial.unlink(missing_ok=True)
            return False
        if total and done != total:
            partial.unlink(missing_ok=True)
            raise SetupError(f"The download of {label} was cut short. Press Set up again to try once more.")
        os.replace(partial, path)
        return True


def add_model() -> None:
    """Put the flux2-klein line in Settings, in place of any earlier one."""
    settings = context.load_settings()
    lines = [line for line in settings["pictures"]["models"].splitlines()
             if line.strip() and not line.strip().lower().startswith(NAME + ":")]
    settings["pictures"]["models"] = "\n".join(lines + [imagegen.klein_setup()["line"]])
    context.save_settings(settings)


SETUP = Setup()
