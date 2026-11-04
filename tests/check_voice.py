"""Reading aloud on the Mac, with the real Kokoro, checked by ear by Whisper:
python3 tests/check_voice.py

Gets what it needs if it is not there, and deletes it after (what was there before is left
alone): Kokoro (0.4 GB) and Whisper large-v3 turbo, 4-bit (0.47 GB). Sets up
~/.hikarilm/voice-venv the way hikarilm restart does, then has the real worker read what the
page sends for answers with math, code, tables, links and symbols, and Whisper write down what
it heard. Each line must be heard as it was written (few words wrong, and the ones that carry
the meaning all there). So must a sentence in every English voice, and the next reading after
one is stopped. A faster speed must be shorter. The check also makes sure it would notice
wrong speech: one line's sound against another line's words must fail.
"""

from __future__ import annotations

import base64
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from helper import engine, voice  # noqa: E402
from helper.engine import ModelError  # noqa: E402

WHISPER = "mlx-community/whisper-large-v3-turbo-asr-4bit"
MOST_WRONG = 0.2  # of the words in a line, or one word, whichever is more
# What spoken.js makes of answers (tests/spoken.test.mjs pins these), and the words in each
# that must be heard.
LINES = [
    ("There is a plot above.", ["plot"]),
    ("The graph of y equals x squared from x equals 0 to x equals 4 is a parabola that opens upward.",
     ["equals", "squared", "parabola"]),
    ("There is a Python code block here, 3 lines long.", ["python", "code", "block", "lines"]),
    ("the integral from 0 to 4 of x squared d x equals 64 over 3.", ["integral", "squared", "sixty", "over"]),
    ("the fraction negative b plus or minus the square root of b squared minus 4 a c, over 2 a.",
     ["fraction", "negative", "plus", "minus", "square", "root"]),
    ("A table with 2 columns: Name, Age.", ["table", "columns", "name", "age"]),
    ("Name, Ann; Age, 30.", ["name", "age", "thirty"]),
    ("See the docs or a link to example.com.", ["docs", "link", "example"]),
    ("It costs $5 and $10, about 30 degrees C, 50 percent done.", ["five", "ten", "dollars", "degrees", "percent"]),
    ("sine squared of x plus cosine squared of x equals 1.", ["sine", "cosine", "squared", "one"]),
    ("Run pip install numpy first.", ["pip", "install"]),
]
VOICE_LINE = "The quick brown fox jumps over the lazy dog."
failures = 0


def report(ok: bool, what: str, detail: str = "") -> None:
    global failures
    failures += not ok
    print(("ok    " if ok else "FAIL  ") + what + (f": {detail}" if detail else ""), flush=True)


# --- comparing what was said with what was heard -------------------------------------

ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen " \
       "seventeen eighteen nineteen".split()
TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()
# Words Whisper may write for the same sound.
SAME = {"ex": "x", "why": "y", "two": "2", "too": "2", "to": "2", "for": "4", "four": "4", "won": "one",
        "sign": "sine", "dx": "d x"}
# How Whisper may write the math it hears.
SIGNS = [("²", " squared"), ("³", " cubed"), ("±", " plus or minus "), ("=", " equals "), ("+", " plus "),
         ("√", " square root of "), ("×", " times "), ("÷", " divided by ")]


def number_words(n: int) -> str:
    if n < 20:
        return ONES[n]
    if n < 100:
        return TENS[n // 10] + ("" if n % 10 == 0 else " " + ONES[n % 10])
    if n < 1000:
        return ONES[n // 100] + " hundred" + ("" if n % 100 == 0 else " " + number_words(n % 100))
    if n < 1_000_000:
        return number_words(n // 1000) + " thousand" + ("" if n % 1000 == 0 else " " + number_words(n % 1000))
    return str(n)


def words(text: str) -> list[str]:
    """Lowercase words with numbers spelled out, so "$5" and "five dollars" are the same."""
    text = text.lower().replace("%", " percent").replace("°", " degrees")
    for sign, said in SIGNS:
        text = text.replace(sign, said)
    text = re.sub(r"(\w)\s*/\s*(\w)", r"\1 over \2", text)
    text = re.sub(r"\s[-−]\s", " minus ", text)
    text = re.sub(r"(\w)\^2\b", r"\1 squared", text)
    text = re.sub(r"\$(\d+)", r"\1 dollars", text)
    text = re.sub(r"(\d),(\d{3})", r"\1\2", text)
    text = re.sub(r"\d+", lambda m: " " + number_words(int(m.group())) + " ", text)
    return " ".join(SAME.get(word, word) for word in re.findall(r"[a-z]+", text)).split()


def errors(said: str, heard: str) -> tuple[int, int]:
    """Words changed, added or left out, and how many words were said."""
    a, b = words(said), words(heard)
    if not a:
        return len(b), 0
    row = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        previous, row[0] = row[0], i
        for j, y in enumerate(b, 1):
            previous, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, previous + (x != y))
    return row[len(b)], len(a)


def heard_right(said: str, heard: str) -> bool:
    count, total = errors(said, heard)
    return count <= max(1, MOST_WRONG * total)


def rate(said: str, heard: str) -> str:
    count, total = errors(said, heard)
    return f"{count} of {total} words wrong"


def missing(keys: list[str], heard: str) -> list[str]:
    got = set(words(heard))
    return [key for key in keys if not set(words(key)) <= got]


# --- running it --------------------------------------------------------------------

def repo_folder(repo: str) -> Path:
    return engine.hub() / ("models--" + repo.replace("/", "--"))


def download(repo: str, what: str) -> None:
    print(f"      {what} is not downloaded; getting it for the check…", flush=True)
    script = f"from huggingface_hub import snapshot_download; snapshot_download({repo!r})"
    subprocess.run([str(engine.python()), "-c", script], check=True)


def whisper_folder() -> Path | None:
    for copy in engine._snapshots(repo_folder(WHISPER)):
        if (copy / "config.json").is_file() and (copy / "tokenizer.json").is_file() and any(copy.glob("*.safetensors")):
            return copy
    return None


def pcm(parts: list[dict]) -> bytes:
    return b"".join(base64.b64decode(part["audio"]) for part in parts)


def hear(clips: list[bytes], scratch: Path) -> list[str]:
    """What Whisper heard in each clip of 24 kHz 16-bit sound."""
    listed = []
    for number, sound in enumerate(clips):
        path = scratch / f"{number}.pcm"
        path.write_bytes(sound)
        listed.append({"pcm": str(path), "rate": 24000})
    done = subprocess.run([str(voice.python()), str(HERE / "voice_hear.py"), "--whisper", str(whisper_folder())],
                          input=json.dumps(listed), capture_output=True, text=True, timeout=1800)
    if done.returncode:
        raise RuntimeError("Whisper could not listen: " + (done.stderr.strip().splitlines() or ["?"])[-1])
    return json.loads(done.stdout.strip().splitlines()[-1])


def main() -> int:
    got_kokoro = got_whisper = False
    try:
        if voice.folder() is None:
            download(voice.REPO, "Kokoro (0.4 GB)")
            got_kokoro = True
        if whisper_folder() is None:
            download(WHISPER, "Whisper large-v3 turbo, 4-bit (0.47 GB)")
            got_whisper = True
        report(voice.folder() is not None and whisper_folder() is not None, "Kokoro and Whisper are there")
        started = time.monotonic()
        report(voice.prepare(lambda text: print("      " + text, flush=True)), "the voice venv is set up and loads",
               f"{time.monotonic() - started:.0f} s")
        if failures:
            return 1
        voices = voice.voices()
        report(len(voices) >= 20, "English voices", f"{len(voices)}")

        with tempfile.TemporaryDirectory() as scratch:
            scratch = Path(scratch)
            started = time.monotonic()
            first = None
            parts = []
            for part in voice.speak([line for line, _ in LINES], voice.DEFAULT_VOICE, 1.0):
                first = first or time.monotonic() - started
                parts.append(part)
            report({part["line"] for part in parts} == set(range(len(LINES))), "every line came back as sound",
                   f"first after {first:.1f} s (loading included), all in {time.monotonic() - started:.1f} s")
            clips = [pcm([part for part in parts if part["line"] == number]) for number in range(len(LINES))]

            normal = pcm(list(voice.speak([LINES[1][0]], voice.DEFAULT_VOICE, 1.0)))
            quick = pcm(list(voice.speak([LINES[1][0]], voice.DEFAULT_VOICE, 1.6)))
            report(len(quick) < len(normal) * 0.85, "a faster speed is shorter",
                   f"{len(normal) / 48000:.1f} s at 1.0, {len(quick) / 48000:.1f} s at 1.6")

            reading = voice.speak([line for line, _ in LINES], voice.DEFAULT_VOICE, 1.0)
            next(reading)
            reading.close()
            after_stop = "Stopped, and on to the next one."
            after = list(voice.speak([after_stop], voice.DEFAULT_VOICE, 1.0))
            report(all(part["line"] == 0 for part in after), "a stopped reading leaves the next one alone")

            pid = voice._worker.process.pid
            each = {name: pcm(list(voice.speak([VOICE_LINE], name, 1.0))) for name in voices}
            report(voice._worker.process.pid == pid, "Kokoro stays loaded between readings")
            voice.unload()

            heard = hear(clips + [pcm(after), quick] + list(each.values()), scratch)
            for (line, keys), text in zip(LINES, heard):
                gone = missing(keys, text)
                report(heard_right(line, text) and not gone, f"heard: {line}",
                       f"{rate(line, text)}{', missing ' + ', '.join(gone) if gone else ''}; Whisper wrote {text!r}")
            rest = heard[len(LINES):]
            report(heard_right(after_stop, rest[0]), "the reading after a stop is heard whole", repr(rest[0]))
            report(heard_right(LINES[1][0], rest[1]), "a faster reading is still heard", repr(rest[1]))
            bad = [f"{name} ({text!r})" for name, text in zip(each, rest[2:]) if not heard_right(VOICE_LINE, text)]
            report(not bad, f"every English voice is heard saying {VOICE_LINE!r}", ", ".join(bad))
            report(not heard_right(LINES[5][0], heard[1]),
                   "the check notices wrong speech (one line's sound against another's words)", rate(LINES[5][0], heard[1]))
    except (ModelError, RuntimeError, subprocess.CalledProcessError, subprocess.TimeoutExpired, StopIteration) as error:
        report(False, "reading aloud", str(error) or type(error).__name__)
    finally:
        voice.unload()
        for got, repo in ((got_kokoro, voice.REPO), (got_whisper, WHISPER)):
            if got:
                shutil.rmtree(repo_folder(repo), ignore_errors=True)
                print(f"      deleted {repo}, which this check downloaded")
    print("All OK." if not failures else f"{failures} FAILED.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
