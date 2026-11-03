"""Stands in for helper/mlx_worker.py in tests: the same lines in and out, no MLX.

The last message picks what it does: "crash" ends the process mid-reply, "slow"
writes many pieces until it is stopped, "options" answers with what it was sent,
"system" answers with the first message it was sent (and how many there were),
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


def calc_pieces(command, asked):
    """"calc: EXPR" calls the calculator with EXPR (and writes a made-up result after the call,
    which must never be seen), then answers with the result it is given and what it was sent.
    "calc-think:" calls it from inside its thinking. "calc-many:" calls EXPR+0, EXPR+1, … while
    it may, and "calc-repeat:" EXPR over and over; "calc-stubborn:" calls even with no tool.
    With no tool it says so, and what came with the question."""
    kind, expression = asked.split(":", 1)
    messages = command["messages"]
    results = [m["content"] for m in messages if m["role"] == "tool"]
    if not command.get("tools") and kind != "calc-stubborn":
        note = asked.split("\n\n", 1)[1] if "\n\n" in asked else ""
        return iter(["no tools" + (" | " + note if note else "")])
    if results and kind in ("calc", "calc-think"):
        sent = {"results": results, "roles": [m["role"] for m in messages], "tools": bool(command.get("tools")),
                "calls": [c["function"]["arguments"] for m in messages for c in m.get("tool_calls") or []],
                "written": [m["content"] for m in messages if m.get("tool_calls")]}
        return iter(["The answer is ", results[-1], ". ", json.dumps(sent)])
    if kind == "calc-many":
        expression = f"{expression.strip()}+{len(results)}"
    call = '<tool_call>\n{"name": "calculate", "arguments": {"expression": ' + json.dumps(expression.strip()) + "}}\n</tool_call>"
    lead = "<think>I should use the calculator. " if kind == "calc-think" else "Let me work it out. "
    text = lead + call + " The result is 42, I made it up."
    return iter(text[i:i + 3] for i in range(0, len(text), 3))


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
        asked = next((m["content"] for m in reversed(command["messages"]) if m["role"] == "user"), "")
        if asked.startswith(("calc:", "calc-think:", "calc-many:", "calc-repeat:", "calc-stubborn:")):
            pieces = calc_pieces(command, asked)
            last = ""
        if last == "crash":
            send({"text": "half"})
            os._exit(1)
        if not last:
            pass
        elif last == "slow":
            pieces = (f"word{n} " for n in range(10000))
        elif last == "system":
            first = command["messages"][0]
            pieces = iter([json.dumps({"role": first["role"], "content": first["content"],
                                       "count": len(command["messages"])})])
        elif last == "options":
            sent = {key: command.get(key) for key in ("think", "num_ctx", "num_predict", "kv_bits")}
            sent.update(vision=args.vision, pid=os.getpid(), images=len(command["messages"][-1].get("images") or []))
            pieces = iter([json.dumps(sent)])
        else:
            pieces = iter(("echo: " + last).split(" "))
        count = 0
        for piece in pieces:
            send({"text": piece if count == 0 or not last else " " + piece})
            count += 1
            if stop.is_set():
                break
            if last == "slow":
                time.sleep(0.01)
        send({"done": True, "stopped": stop.is_set(), "reason": "stop", "prompt_tokens": 11, "tokens": count})


if __name__ == "__main__":
    main()
