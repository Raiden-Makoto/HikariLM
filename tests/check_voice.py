"""Reading aloud on the Mac, with the real Kokoro: python3 tests/check_voice.py

Gets Kokoro if it is not downloaded (0.4 GB) and deletes it after; one that was there before
is left alone. Sets up ~/.hikarilm/voice-venv the way hikarilm restart does, then reads what
the page sends for answers with code, math, tables, links and symbols, through the real
worker. Each line must come back as sound (not silence, and about as long as the words), a
faster speed must be shorter, a stopped reading must leave the next one clean, and every
English voice must speak. What it said is written to WAV files to listen to.
"""

from __future__ import annotations

import base64
import math
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from helper import engine, voice  # noqa: E402
from helper.engine import ModelError  # noqa: E402

OUT = Path(tempfile.gettempdir()) / "hikarilm-voice-check"
# What spoken.js makes of answers with math, code, tables, links and symbols.
LINES = [
    "There is a plot above.",
    "The graph of y equals x squared from x equals 0 to x equals 4 is a parabola that opens upward.",
    "There is a Python code block here, 3 lines long.",
    "the integral from 0 to 4 of x squared d x equals 64 over 3.",
    "the fraction negative b plus or minus the square root of b squared minus 4 a c, over 2 a.",
    "A table with 2 columns: Name, Age.",
    "Name, Ann; Age, 30.",
    "See the docs or a link to example.com.",
    "It costs $5 and $10, about 30 degrees C, 50 percent done.",
    "sine squared of x plus cosine squared of x equals 1.",
    "Run pip install numpy first.",
]
failures = 0


def report(ok: bool, what: str, detail: str = "") -> None:
    global failures
    failures += not ok
    print(("ok    " if ok else "FAIL  ") + what + (f": {detail}" if detail else ""), flush=True)


def samples(parts: list[dict]) -> list[int]:
    out: list[int] = []
    for part in parts:
        raw = base64.b64decode(part["audio"])
        out.extend(struct.unpack(f"<{len(raw) // 2}h", raw))
    return out


def save(name: str, values: list[int], rate: int = 24000) -> Path:
    OUT.mkdir(exist_ok=True)
    path = OUT / f"{name}.wav"
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes(struct.pack(f"<{len(values)}h", *values))
    return path


def loudness(values: list[int]) -> float:
    return math.sqrt(sum(v * v for v in values) / len(values)) / 32768 if values else 0.0


def get_kokoro() -> bool:
    """Download Kokoro if it is not there. True if this check downloaded it."""
    if voice.folder() is not None:
        print(f"      Kokoro is already downloaded ({voice.folder()}); it is left alone")
        return False
    print("      Kokoro is not downloaded; getting it for the check (0.4 GB)…", flush=True)
    script = f"from huggingface_hub import snapshot_download; snapshot_download({voice.REPO!r})"
    subprocess.run([str(engine.python()), "-c", script], check=True)
    return True


def main() -> int:
    downloaded = False
    try:
        downloaded = get_kokoro()
        report(voice.folder() is not None, "Kokoro's files are there", str(voice.folder()))
        started = time.monotonic()
        report(voice.prepare(lambda text: print("      " + text, flush=True)), "the voice venv is set up and loads",
               f"{time.monotonic() - started:.0f} s")
        if failures:
            return 1
        voices = voice.voices()
        report(len(voices) >= 10, "English voices", f"{len(voices)}: {', '.join(voices)}")

        started = time.monotonic()
        first = None
        parts = []
        for part in voice.speak(LINES, voice.DEFAULT_VOICE, 1.0):
            first = first or time.monotonic() - started
            parts.append(part)
        took = time.monotonic() - started
        report({part["line"] for part in parts} == set(range(len(LINES))), "every line came back as sound",
               f"first after {first:.1f} s (loading included), all in {took:.1f} s")
        for number, line in enumerate(LINES):
            values = samples([part for part in parts if part["line"] == number])
            seconds = len(values) / 24000
            words = len(line.split())
            fits = words * 0.15 < seconds < words * 0.8 + 1.5
            report(loudness(values) > 0.01 and fits, f"line {number + 1} sounds right",
                   f"{seconds:.1f} s for {words} words, loudness {loudness(values):.3f}  ({line[:50]})")
        path = save("answer", samples(parts))
        print(f"      listen: {path}")

        normal = len(samples(list(voice.speak([LINES[1]], voice.DEFAULT_VOICE, 1.0))))
        quick = len(samples(list(voice.speak([LINES[1]], voice.DEFAULT_VOICE, 1.6))))
        report(quick < normal * 0.85, "a faster speed is shorter", f"{normal / 24000:.1f} s at 1.0, {quick / 24000:.1f} s at 1.6")

        reading = voice.speak(LINES, voice.DEFAULT_VOICE, 1.0)
        next(reading)
        reading.close()
        after = list(voice.speak(["Stopped, and on to the next one."], voice.DEFAULT_VOICE, 1.0))
        report(len(after) >= 1 and all(part["line"] == 0 for part in after), "a stopped reading leaves the next one clean")

        quiet = []
        for name in voices:
            values = samples(list(voice.speak(["Hello, this is " + name.split("_")[1] + "."], name, 1.0)))
            if loudness(values) <= 0.01:
                quiet.append(name)
            save("voice-" + name, values)
        report(not quiet, "every English voice speaks", ", ".join(quiet) or f"samples in {OUT}")

        before = voice._worker.process.pid if voice._worker else None
        for _ in voice.speak(["Still here."], voice.DEFAULT_VOICE, 1.0):
            pass
        report(voice._worker is not None and voice._worker.process.pid == before, "Kokoro stays loaded between readings")
    except (ModelError, subprocess.CalledProcessError, StopIteration) as error:
        report(False, "reading aloud", str(error) or type(error).__name__)
    finally:
        voice.unload()
        if downloaded:
            repo = engine.hub() / ("models--" + voice.REPO.replace("/", "--"))
            shutil.rmtree(repo, ignore_errors=True)
            print("      deleted the Kokoro this check downloaded")
    print("All OK." if not failures else f"{failures} FAILED.")
    return 1 if failures else 0


if __name__ == "__main__":
    os.environ.pop("HIKARILM_VOICE_PYTHON", None)
    sys.exit(main())
