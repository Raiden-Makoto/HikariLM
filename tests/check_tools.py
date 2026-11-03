"""Which chat models call the calculator, on a Mac. Not part of the test suite.

Asks each downloaded chat model a few sums through the real helper (the same /api/chat the page
uses, in this process) and reports, for each, whether it called the calculator, what it sent, and
whether its answer has the right number. Nothing is downloaded; models that are not on this Mac
are skipped. Picture and embedding models are left out.

    python3 tests/check_tools.py                  every downloaded chat model
    python3 tests/check_tools.py mlx-community/Qwen3-8B-4bit

Your settings are used as they are (the calculator must be on), including thinking and its limit.
Write what it finds in PLAN.md under v4.8.0.
"""

import json
import subprocess
import sys
import threading
import time
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from helper import context, engine, server, tools  # noqa: E402

SUMS = [
    ("What is the square root of 4761?", "69"),
    ("What is 2 to the power of 40?", str(2 ** 40)),
    ("What is 1234567 times 7654321?", str(1234567 * 7654321)),
    ("What is the sine of 30 degrees, times 12.5?", "6.25"),
]
SKIP = ("flux", "embed", "whisper", "kokoro")


def chat_models(wanted: list[str]) -> list[str]:
    engine.forget_scan()
    found = sorted(engine.found())
    if wanted:
        missing = [name for name in wanted if name not in found]
        for name in missing:
            print(f"{name} is not downloaded; skipped.")
        return [name for name in wanted if name in found]
    return [name for name in found if not any(word in name.lower() for word in SKIP) and not tools.CODERS.search(name)]


def ask(base: str, model: str, question: str) -> dict:
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": question}]}).encode()
    request = urllib.request.Request(base + "/api/chat", body, {"Content-Type": "application/json"})
    started = time.monotonic()
    calls, answer, thought, error = [], "", "", ""
    with urllib.request.urlopen(request, timeout=900) as response:
        for line in response:
            if not line.strip():
                continue
            event = json.loads(line)
            if event.get("error"):
                error = event["error"]
            if event.get("hikarilm") == "tool":
                calls.append(event)
            message = event.get("message") or {}
            answer += message.get("content", "")
            thought += message.get("thinking", "")
    return {"calls": calls, "answer": answer, "thought": thought, "error": error,
            "seconds": round(time.monotonic() - started, 1)}


def versions() -> str:
    """mlx-lm, mlx-vlm and transformers in the helper's venv, which decide how templates load."""
    script = ("import importlib.metadata as m\n"
              "for p in ('mlx-lm', 'mlx-vlm', 'transformers'):\n"
              "    try: print(p, m.version(p))\n"
              "    except m.PackageNotFoundError: print(p, 'missing')")
    try:
        done = subprocess.run([str(engine.python()), "-c", script], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as error:
        return f"(could not ask the venv: {error})"
    return ", ".join(done.stdout.split("\n")[:3]).strip(", ") or done.stderr.strip()[-300:]


def main() -> int:
    if not context.load_settings()["tools"]["calculator"]:
        print("Turn on Settings, Let models use the calculator, first.")
        return 1
    print("Venv:", versions())
    models = chat_models(sys.argv[1:])
    if not models:
        print("No chat models are downloaded.")
        return 1
    http = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
    threading.Thread(target=http.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{http.server_address[1]}"
    summary = []
    try:
        for model in models:
            print(f"\n== {model}")
            called = right = 0
            for question, expected in SUMS:
                try:
                    result = ask(base, model, question)
                except Exception as error:  # noqa: BLE001 - reported, the next one still runs
                    print(f"  {question}\n    failed: {error}")
                    continue
                said = result["answer"].replace(",", "").replace("\u202f", "").replace(" ", "")
                ok = expected in said
                called += bool(result["calls"])
                right += ok
                print(f"  {question}  ({result['seconds']} s)")
                for call in result["calls"]:
                    print(f"    called: {call['expression']} -> {call.get('result') or 'Error: ' + call.get('tool_error', '')}")
                if not result["calls"]:
                    mentioned = "calculat" in result["thought"].lower()
                    print("    no call" + (" (its thinking mentions the calculator)" if mentioned else ""))
                if result["error"]:
                    print(f"    error: {result['error']}")
                print(f"    answer {'right' if ok else 'WRONG'}: {result['answer'].strip()[:160]!r}")
            summary.append((model, called, right))
            engine.unload()
    finally:
        http.shutdown()
        engine.unload()
    print("\nModel: questions where it called the calculator, right answers, of", len(SUMS))
    for model, called, right in summary:
        print(f"  {model}: called {called}, right {right}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
