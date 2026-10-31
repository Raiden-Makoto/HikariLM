"""Stands in for helper/mlx_worker.py in tests: the same lines in and out, no MLX.

The last message picks what it does: "crash" ends the process mid-reply, "slow"
writes many pieces until it is stopped, "options" answers with what it was sent,
anything else is echoed back word by word.

With --embed it stands in for nomic: each text becomes a unit vector of its
words, hashed, so texts sharing words are close. A text "crash" ends it.
"""

import argparse
import hashlib
import json
import math
import os
import re
import queue
import sys
import threading
import time


def send(event):
    sys.stdout.write(json.dumps(event) + "\n")
    sys.stdout.flush()


DIM = 64


def vector(text):
    out = [0.0] * DIM
    for word in re.findall(r"[a-z0-9]+", text.split(": ", 1)[-1].lower()):
        out[int(hashlib.sha1(word.encode()).hexdigest(), 16) % DIM] += 1.0
    size = math.sqrt(sum(x * x for x in out)) or 1.0
    return [x / size for x in out]


def embed_loop():
    for line in sys.stdin:
        command = json.loads(line)
        if "crash" in command.get("texts", []):
            os._exit(1)
        send({"vectors": [vector(text) for text in command["texts"]], "pid": os.getpid()})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--vision", action="store_true")
    parser.add_argument("--embed", action="store_true")
    args = parser.parse_args()
    if "broken" in args.model:
        send({"error": "The model could not be loaded: broken on purpose"})
        return
    send({"ready": True, "seconds": 0.1})
    if args.embed:
        embed_loop()
        return
    commands, stop = queue.Queue(), threading.Event()

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
        last = command["messages"][-1]["content"]
        if last == "crash":
            send({"text": "half"})
            os._exit(1)
        if last == "slow":
            pieces = (f"word{n} " for n in range(10000))
        elif last == "options":
            sent = {key: command.get(key) for key in ("think", "num_ctx", "num_predict", "kv_bits")}
            sent.update(vision=args.vision, pid=os.getpid(), images=len(command["messages"][-1].get("images") or []))
            pieces = iter([json.dumps(sent)])
        else:
            pieces = iter(("echo: " + last).split(" "))
        count = 0
        for piece in pieces:
            send({"text": piece if count == 0 else " " + piece})
            count += 1
            if stop.is_set():
                break
            if last == "slow":
                time.sleep(0.01)
        send({"done": True, "stopped": stop.is_set(), "reason": "stop", "prompt_tokens": 11, "tokens": count})


if __name__ == "__main__":
    main()
