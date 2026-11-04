"""Stands in for helper/voice_worker.py in tests: no MLX, no Kokoro.

It says ready with the English voices in the folder, then for each line sends silence, 100
samples a character, with the voice and speed in the line number's place for the tests to
see. A line "slow" takes half a second, "fail" answers with an error and carries on, and
"crash" makes it exit.
"""

import argparse
import base64
import json
import queue
import sys
import threading
import time
from pathlib import Path


def say(event):
    sys.stdout.write(json.dumps(event) + "\n")
    sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    voices = sorted(p.stem for p in (Path(args.model) / "voices").glob("*.safetensors") if p.stem[:1] in "ab")
    say({"ready": True, "voices": voices, "rate": 24000})
    commands = queue.Queue()
    stop = threading.Event()

    def read():
        for line in sys.stdin:
            command = json.loads(line)
            if command.get("op") == "stop":
                stop.set()
            else:
                commands.put(command)
        commands.put(None)

    threading.Thread(target=read, daemon=True).start()
    while True:
        command = commands.get()
        if command is None:
            return
        stop.clear()
        failed = False
        for number, text in enumerate(command["lines"]):
            if stop.is_set():
                break
            if text == "crash":
                sys.exit(3)
            if text == "fail":
                say({"error": "Kokoro could not read that: no"})
                failed = True
                break
            if text == "slow":
                time.sleep(0.5)
            audio = base64.b64encode(b"\x00\x00" * 100 * len(text)).decode()
            say({"audio": audio, "rate": 24000, "line": number, "voice": command["voice"], "speed": command["speed"]})
        if not failed:
            say({"done": True, "stopped": stop.is_set()})


if __name__ == "__main__":
    main()
