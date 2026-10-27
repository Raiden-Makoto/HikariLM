"""Start LightUsagi: check the basics, run the helper, open the page.

    python3 start.py              start and open the browser
    python3 start.py --no-browser start without opening a browser
    python3 start.py --port 9000  use another port

Python 3.9 or newer is all it needs. No packages to install.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import threading
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def say(text: str = "") -> None:
    print(text, flush=True)


def check_python() -> None:
    if sys.version_info < (3, 9):
        sys.exit(f"LightUsagi needs Python 3.9 or newer. This is Python {sys.version.split()[0]}.")


def check_files() -> None:
    missing = [name for name in ("helper/server.py", "web/index.html") if not (ROOT / name).is_file()]
    if missing:
        sys.exit("These files are missing, so this folder is not a full copy of LightUsagi: " + ", ".join(missing))


def running_state(host: str, port: int) -> dict | None:
    """What a LightUsagi on this port says about itself, or None if it is not one."""
    try:
        with urllib.request.urlopen(f"http://{host}:{port}/api/state", timeout=2) as response:
            state = json.loads(response.read().decode())
    except (OSError, ValueError):
        return None
    return state if isinstance(state, dict) and "large_model" in state else None


def port_is_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        return probe.connect_ex((host, port)) != 0


def report_ollama() -> None:
    """Say what Ollama has. Never stops the start: it can be launched later."""
    from helper.ollama import OLLAMA_HOST, OllamaError, tags
    from helper.router import chat_models

    try:
        names = chat_models(tags())
    except OllamaError:
        say(f"Ollama is not running at {OLLAMA_HOST}.")
        say("  Install it from https://ollama.com and start it, then reload the page.")
        say("  The page will list your models as soon as Ollama answers.")
        return
    if names:
        say("Ollama models: " + ", ".join(names))
    else:
        say("Ollama is running but has no chat models yet. Try:  ollama pull gemma3")


def main() -> None:
    parser = argparse.ArgumentParser(description="Start LightUsagi.")
    parser.add_argument("--port", type=int, default=int(os.environ.get("LIGHTUSAGI_PORT", "8787")))
    parser.add_argument("--host", default=os.environ.get("LIGHTUSAGI_HOST", "127.0.0.1"))
    parser.add_argument("--no-browser", action="store_true", help="do not open the page")
    args = parser.parse_args()

    check_python()
    check_files()

    url = f"http://{args.host}:{args.port}"
    if not port_is_free(args.host, args.port):
        state = running_state(args.host, args.port)
        if state is not None:
            from helper.version import code_stamp

            if state.get("code") != code_stamp():
                # A copy started before the files changed keeps running its old
                # code, so pages added since would show "not found".
                sys.exit(
                    f"An older copy of LightUsagi is still running at {url}, started before the files were updated.\n"
                    "Stop it (Ctrl+C in the window where it runs, or close that window) and run this again."
                )
            say(f"LightUsagi is already running at {url}")
            if not args.no_browser:
                webbrowser.open(url)
            return
        sys.exit(f"Port {args.port} is used by another program. Try:  python3 start.py --port {args.port + 1}")

    # The helper reads these when it is imported, so set them first.
    os.environ["LIGHTUSAGI_HOST"] = args.host
    os.environ["LIGHTUSAGI_PORT"] = str(args.port)

    from helper import store
    from helper.server import main as serve

    say("LightUsagi")
    say(f"  page:   {url}")
    say(f"  saved:  {store.home()}")
    report_ollama()
    say("Press Ctrl+C to stop.")
    say()

    if not args.no_browser:
        threading.Timer(0.8, webbrowser.open, args=(url,)).start()
    serve()


if __name__ == "__main__":
    main()
