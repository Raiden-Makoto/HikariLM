"""Kokoro and Whisper, for the HikariLM helper. Runs with the Python in ~/.hikarilm/voice-venv,
which has mlx-audio and misaki; speaks JSON lines on stdin and stdout.

Reading aloud, started with --model <Kokoro's folder>:
    -> {"ready": true, "voices": [...], "rate": 24000}
    {"op": "speak", "lines": ["...", ...], "voice": "af_heart", "speed": 1.0}
        -> {"audio": "<base64 16-bit PCM, mono>", "rate": 24000, "line": n} as each part is made,
           then {"done": true, "stopped": false}

Writing down speech, started with --model <Whisper's folder> --listen:
    -> {"ready": true}
    {"op": "transcribe", "audio": "<base64 16 kHz mono 16-bit PCM>", "language": "en" or null}
        -> {"text": "..."} for each part of up to 30 seconds, cut where it is quiet, then
           {"done": true, "language": "en", "stopped": false}

    {"op": "stop"}   ends either; its "done" line still comes.

Nothing is downloaded here: the voices are the ones in Kokoro's folder, Whisper's tokenizer is
in its folder, and spaCy's English model was installed with the venv.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import queue
import sys
import threading
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"

OUT = sys.stdout
# mlx-audio prints as it works ("Creating new KokoroPipeline…"); only these lines go to the helper.
sys.stdout = sys.stderr
RATE = 16000
PART = 30 * RATE          # Whisper hears 30 seconds at a time
LOOK_BACK = 6 * RATE      # how far back from 30 seconds to look for a quiet place to cut
QUIET = 0.003             # a part this quiet is silence: Whisper would make words up for it


def send(payload: dict) -> None:
    OUT.write(json.dumps(payload) + "\n")
    OUT.flush()


def pcm(audio) -> str:
    import numpy as np

    samples = np.clip(np.asarray(audio, dtype=np.float32).reshape(-1), -1.0, 1.0)
    return base64.b64encode((samples * 32767).astype("<i2").tobytes()).decode("ascii")


def commands_from_stdin(stop: threading.Event) -> queue.Queue:
    commands: queue.Queue = queue.Queue()

    def read() -> None:
        for line in sys.stdin:
            try:
                command = json.loads(line)
            except ValueError:
                continue
            if command.get("op") == "stop":
                stop.set()
            else:
                commands.put(command)
        stop.set()
        commands.put(None)

    threading.Thread(target=read, daemon=True).start()
    return commands


def parts(samples):
    """Pieces of up to 30 seconds, each cut at the quietest tenth of a second near its end."""
    import numpy as np

    start = 0
    while start < len(samples):
        end = min(start + PART, len(samples))
        if end < len(samples):
            window = samples[end - LOOK_BACK:end]
            frames = len(window) // (RATE // 10)
            loudness = np.sqrt((window[:frames * (RATE // 10)].reshape(frames, -1) ** 2).mean(axis=1))
            end = end - LOOK_BACK + int(loudness.argmin()) * (RATE // 10) + RATE // 20
        yield samples[start:end]
        start = end


def listen(where: Path) -> None:
    try:
        import numpy as np
        from mlx_audio.stt.utils import load_model

        model = load_model(where)
        if getattr(model, "_processor", None) is None:
            raise RuntimeError(f"there is no tokenizer in {where}; get an -asr- Whisper")
    except Exception as error:  # noqa: BLE001 - said to the helper, which tells the page
        send({"error": f"Whisper could not start: {error}"})
        return
    send({"ready": True})
    stop = threading.Event()
    commands = commands_from_stdin(stop)
    while True:
        command = commands.get()
        if command is None:
            return
        if command.get("op") != "transcribe":
            send({"error": f"unknown op {command.get('op')!r}"})
            continue
        stop.clear()
        language = command.get("language") or None
        written = ""
        try:
            samples = np.frombuffer(base64.b64decode(command.get("audio") or ""), dtype="<i2").astype(np.float32) / 32768
            for piece in parts(samples):
                if stop.is_set():
                    break
                if len(piece) < RATE // 4 or float(np.sqrt((piece ** 2).mean())) < QUIET:
                    continue
                result = model.generate(piece, language=language, temperature=0.0, verbose=None,
                                        condition_on_previous_text=False,
                                        initial_prompt=written[-200:] or None)
                language = language or getattr(result, "language", None)
                text = (result.text or "").strip()
                if text:
                    written += " " + text
                    send({"text": text})
        except Exception as error:  # noqa: BLE001
            print(f"writing down failed: {error}", file=sys.stderr, flush=True)
            send({"error": f"Whisper could not write that down: {error}"})
            continue
        send({"done": True, "language": language, "stopped": stop.is_set()})


def speak(where: Path) -> None:
    try:
        from mlx_audio.tts.utils import load_model

        model = load_model(where, model_type="kokoro")
        voices = sorted(path.stem for path in (where / "voices").glob("*.safetensors") if path.stem[:1] in "ab")
        if not voices:
            raise RuntimeError(f"there are no English voices in {where / 'voices'}")
        rate = int(model.sample_rate)
        # Reading one word loads misaki and spaCy now, not on the first Listen.
        for _ in model.generate("Ready.", voice=voices[0], speed=1.0, lang_code=voices[0][0]):
            pass
    except Exception as error:  # noqa: BLE001
        send({"error": f"Kokoro could not start: {error}"})
        return
    send({"ready": True, "voices": voices, "rate": rate})
    stop = threading.Event()
    commands = commands_from_stdin(stop)
    while True:
        command = commands.get()
        if command is None:
            return
        if command.get("op") != "speak":
            send({"error": f"unknown op {command.get('op')!r}"})
            continue
        stop.clear()
        voice = command.get("voice") if command.get("voice") in voices else voices[0]
        try:
            for number, text in enumerate(command.get("lines") or []):
                if stop.is_set():
                    break
                for part in model.generate(str(text), voice=voice, speed=float(command.get("speed") or 1.0),
                                           lang_code=voice[0], split_pattern=None):
                    send({"audio": pcm(part.audio), "rate": rate, "line": number})
                    if stop.is_set():
                        break
        except Exception as error:  # noqa: BLE001
            print(f"reading failed: {error}", file=sys.stderr, flush=True)
            send({"error": f"Kokoro could not read that: {error}"})
            continue
        send({"done": True, "stopped": stop.is_set()})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--listen", action="store_true")
    args = parser.parse_args()
    (listen if args.listen else speak)(Path(args.model))


if __name__ == "__main__":
    main()
