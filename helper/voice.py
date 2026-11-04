"""Answers read aloud with Kokoro, run by helper/voice_worker.py in ~/.hikarilm/voice-venv.

mlx-audio runs Kokoro, and pins its own MLX (and wants transformers 5.14 or newer), so it has
a venv of its own, made by `hikarilm restart` once Kokoro is downloaded, the way mflux has
pictures-venv. Kokoro turns English into sounds with misaki, which needs Python 3.12 or older,
and scipy that macOS 27 can load needs 3.11 or newer: the venv is made with 3.11 or 3.12.
misaki's English extra would bring torch (for a transformer parser it does not use here), so
its parts are installed one by one, with spaCy's small English model, which misaki would
otherwise download the first time it reads.

Kokoro (82M parameters, 0.4 GB) stays loaded beside the chat model, like the notes model: it
never counts as the one large model and never makes it leave. The page sends the answer as
the words to say, one sentence or line to a line; each comes back as 16-bit PCM as soon as it
is made, so reading starts before the whole answer is spoken.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time
from pathlib import Path

from helper import engine, store

REPO = "mlx-community/Kokoro-82M-bf16"
NAME = "Kokoro"
WEIGHTS = "kokoro-v1_0.safetensors"
GET = "~/.hikarilm/venv/bin/hf download " + REPO
SPACY_MODEL = ("https://github.com/explosion/spacy-models/releases/download/"
               "en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl")
PACKAGES = ("mlx-audio", "misaki", "num2words", "spacy", "phonemizer-fork", "espeakng-loader", SPACY_MODEL,
            "scipy>=1.16.3")
OLDEST, NEWEST = 11, 12
WORKER = Path(__file__).with_name("voice_worker.py")
# American (a) and British (b) English voices: misaki reads English. The others need
# misaki's Japanese or Chinese parts, or espeak-ng's rules for their language.
LANGUAGES = ("a", "b")
DEFAULT_VOICE = "af_heart"
SPEEDS = (0.5, 2.0)
TEXT_CHARS = 100_000
LINE_SECONDS = 120  # a long line, on the first read after loading
SETUP = ("Reading aloud is not set up yet. Run  hikarilm restart  in Terminal; it installs it once "
         "Kokoro is downloaded.")
# What the venv must import, outside any sandbox, before it counts as ready.
LOADS = ("import mlx_audio.tts.models.kokoro, misaki.en, misaki.espeak, spacy; "
         "spacy.load('en_core_web_sm', enable=['tok2vec', 'tagger'])")


def venv() -> Path:
    return store.home() / "voice-venv"


def python() -> Path:
    return Path(os.environ.get("HIKARILM_VOICE_PYTHON") or venv() / "bin" / "python")


def worker() -> Path:
    return Path(os.environ.get("HIKARILM_VOICE_WORKER") or WORKER)


def folder() -> Path | None:
    """Kokoro's files, once all of them are downloaded."""
    repo = engine.hub() / ("models--" + REPO.replace("/", "--"))
    if engine.downloading(repo):
        return None
    for copy in engine._snapshots(repo):
        if (copy / "config.json").is_file() and (copy / WEIGHTS).is_file() and voices_in(copy):
            return copy
    return None


def voices_in(where: Path) -> list[str]:
    return sorted(path.stem for path in (where / "voices").glob("*.safetensors")
                  if path.stem[:1] in LANGUAGES and path.is_file())


def voices() -> list[str]:
    where = folder()
    return voices_in(where) if where else []


def ready() -> bool:
    if os.environ.get("HIKARILM_VOICE_PYTHON"):
        return python().exists()
    try:
        return (venv() / engine.READY_MARK).read_text(encoding="utf-8").split() == list(PACKAGES)
    except OSError:
        return False


def prepare(say, base_python: str | None = None) -> bool:
    """Make ~/.hikarilm/voice-venv once Kokoro is downloaded. True when it is ready."""
    if os.environ.get("HIKARILM_VOICE_PYTHON"):
        return True
    if folder() is None:
        return False
    base_python = base_python or engine.find_python(OLDEST, NEWEST)
    if not base_python:
        say(f"  Reading aloud needs Python 3.{OLDEST} or 3.{NEWEST} (Kokoro's misaki does not run on newer "
            f"ones, and older ones get a scipy macOS cannot load). Install one with  brew install python@3.12  "
            "or from python.org, then run  hikarilm restart.")
        return False
    made = engine.venv_minor(venv())
    if made is not None and not OLDEST <= made <= NEWEST:
        say(f"Reading aloud was set up with Python 3.{made}; setting it up again with 3.{OLDEST} or 3.{NEWEST}…")
        shutil.rmtree(venv(), ignore_errors=True)
    if not engine._prepare_venv(venv(), python(), PACKAGES, "Kokoro (for reading aloud)", say, base_python):
        return False
    try:
        loads = subprocess.run([str(python()), "-c", LOADS], capture_output=True, text=True, timeout=300,
                               env={**os.environ, "HF_HUB_OFFLINE": "1"})
    except (OSError, subprocess.SubprocessError) as error:
        loads = subprocess.CompletedProcess([], 1, "", str(error))
    if loads.returncode:
        lines = (loads.stderr or loads.stdout).strip().splitlines()
        say(f"  Kokoro is installed but does not load: {lines[-1] if lines else loads.returncode}")
        (venv() / engine.READY_MARK).unlink(missing_ok=True)
        return False
    return True


# --- the worker ----------------------------------------------------------------

_worker: engine._Worker | None = None
_lock = threading.Lock()        # one reading at a time
_turn = threading.Event()        # set to end the reading under way
_turn.set()


def _ready() -> engine._Worker:
    global _worker
    if _worker is not None and _worker.alive():
        return _worker
    where = folder()
    if where is None:
        raise engine.ModelError(f"{NAME} is not downloaded. Get it with:  {GET}")
    if not python().exists() or not ready():
        raise engine.ModelError(SETUP)
    started = engine._Worker(NAME, where, [], "voice-worker.log", python(), worker())
    event = started.next(engine.LOAD_SECONDS)
    if not event or not event.get("ready"):
        started.end()
        raise engine.ModelError((event or {}).get("error") or
                                f"{NAME} could not start. What it said is in {store.home() / 'voice-worker.log'}.")
    _worker = started
    return started


def _forget(process: engine._Worker) -> None:
    global _worker
    process.end()
    if _worker is process:
        _worker = None


def speak(lines: list[str], voice: str, speed: float):
    """Yields {"audio": base64 16-bit PCM, "rate": 24000, "line": n} for each line as it is
    made. Starting another reading ends this one; so does closing the generator."""
    global _turn
    mine = threading.Event()
    _turn.set()
    _turn = mine
    with _lock:
        if mine.is_set():
            return
        process = _ready()
        process.send({"op": "speak", "lines": lines, "voice": voice, "speed": speed})
        done = stopped = False
        try:
            while True:
                if mine.is_set() and not stopped:
                    process.send({"op": "stop"})
                    stopped = True
                event = process.next(LINE_SECONDS)
                if event is None or "error" in event:
                    done = True
                    if event is None or not process.alive():
                        _forget(process)
                    raise engine.ModelError((event or {}).get("error") or
                                            f"{NAME} stopped. What it said is in {store.home() / 'voice-worker.log'}.")
                if event.get("done"):
                    done = True
                    return
                if "audio" in event and not mine.is_set():
                    yield event
        finally:
            if not done:
                _finish(process, stopped)


def _finish(process: engine._Worker, stopped: bool) -> None:
    """Stop a reading the page left, and wait for its done line, so the next one starts clean."""
    deadline = time.monotonic() + engine.STOP_SECONDS
    try:
        if not stopped:
            process.send({"op": "stop"})
        while time.monotonic() < deadline:
            event = process.next(max(0.1, deadline - time.monotonic()))
            if event is None or "error" in event or event.get("done"):
                break
    except engine.ModelError:
        pass
    if not process.alive() or time.monotonic() >= deadline:
        _forget(process)


def running() -> list[dict]:
    current = _worker
    if current is None or not current.alive():
        return []
    return [{"name": NAME, "size": current.size, "expires": None}]


def unload() -> None:
    _turn.set()
    with _lock:
        if _worker is not None:
            _forget(_worker)


def check_request(body: dict) -> tuple[list[str], str, float]:
    """The lines, voice and speed of a /api/speak request, or ValueError, worded for the page."""
    lines = body.get("lines")
    if not isinstance(lines, list) or not all(isinstance(line, str) for line in lines):
        raise ValueError("lines must be a list of strings")
    lines = [" ".join(line.split()) for line in lines]
    lines = [line for line in lines if line]
    if not lines:
        raise ValueError("there is nothing to read aloud")
    if sum(len(line) for line in lines) > TEXT_CHARS:
        raise ValueError(f"that is too long to read aloud (more than {TEXT_CHARS:,} characters)")
    voice = str(body.get("voice") or DEFAULT_VOICE)
    known = voices()
    if known and voice not in known:
        voice = DEFAULT_VOICE if DEFAULT_VOICE in known else known[0]
    try:
        speed = float(body.get("speed", 1.0))
    except (TypeError, ValueError):
        speed = 1.0
    speed = min(SPEEDS[1], max(SPEEDS[0], speed)) if speed == speed else 1.0
    return lines, voice, speed


def status() -> dict:
    return {"downloaded": folder() is not None, "ready": ready(), "voices": voices(), "default": DEFAULT_VOICE,
            "get": GET, "speeds": list(SPEEDS)}
