"""Kokoro, for the HikariLM helper: reads lines aloud. Runs with the Python in
~/.hikarilm/voice-venv, which has mlx-audio and misaki; speaks JSON lines on stdin and stdout.

    started with --model <Kokoro's folder>
        -> {"ready": true, "voices": [...], "rate": 24000}
    {"op": "speak", "lines": ["...", ...], "voice": "af_heart", "speed": 1.0}
        -> {"audio": "<base64 16-bit PCM, mono>", "rate": 24000, "line": n} as each part is made,
           then {"done": true, "stopped": false}
    {"op": "stop"}   ends the reading; its "done" line still comes.

Nothing is downloaded here: the voices are the ones in Kokoro's folder, and spaCy's English
model was installed with the venv.
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


def send(payload: dict) -> None:
    OUT.write(json.dumps(payload) + "\n")
    OUT.flush()


def pcm(audio) -> str:
    import numpy as np

    samples = np.clip(np.asarray(audio, dtype=np.float32).reshape(-1), -1.0, 1.0)
    return base64.b64encode((samples * 32767).astype("<i2").tobytes()).decode("ascii")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    where = Path(args.model)
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
    except Exception as error:  # noqa: BLE001 - said to the helper, which tells the page
        send({"error": f"Kokoro could not start: {error}"})
        return
    send({"ready": True, "voices": voices, "rate": rate})

    commands: queue.Queue = queue.Queue()
    stop = threading.Event()

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


if __name__ == "__main__":
    main()
