"""Start HikariLM: check the basics, run the helper in the background, open the page.

    hikarilm start              start it and open the browser; the terminal can be closed after
    hikarilm stop               stop it
    hikarilm restart            stop it and start it again
    hikarilm status             say whether it runs, and what is loaded
    hikarilm log                the last lines of its log, ~/.hikarilm/hikarilm.log
    hikarilm update             git pull, and restart it if that brought anything
    hikarilm start --foreground run it in this terminal instead, until Ctrl+C
    hikarilm start --no-browser start without opening a browser
    hikarilm start --port 9000  use another port

The first ./start.sh installs the hikarilm command; ./start.sh takes the same words.
Python 3.9 or newer is all it needs. No packages to install.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from helper import __version__  # noqa: E402


def say(text: str = "") -> None:
    print(text, flush=True)


def check_python() -> None:
    if sys.version_info < (3, 9):
        sys.exit(f"HikariLM needs Python 3.9 or newer. This is Python {sys.version.split()[0]}.")
    from helper.imagegen import apple_silicon

    if not apple_silicon():
        sys.exit("HikariLM runs only on a Mac with Apple Silicon (M1 or newer).")


def check_files() -> None:
    missing = [name for name in ("helper/server.py", "web/index.html") if not (ROOT / name).is_file()]
    if missing:
        sys.exit("These files are missing, so this folder is not a full copy of HikariLM: " + ", ".join(missing))


def running_state(host: str, port: int) -> dict | None:
    """What a HikariLM on this port says about itself, or None if it is not one."""
    try:
        with urllib.request.urlopen(f"http://{host}:{port}/api/state", timeout=2) as response:
            state = json.loads(response.read().decode())
    except (OSError, ValueError):
        return None
    return state if isinstance(state, dict) and "large_model" in state else None


def port_is_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        return probe.connect_ex((host, port)) != 0


def report_pictures() -> None:
    """Say which picture models are ready. mflux is installed once there is one to run."""
    from helper import engine, imagegen

    pictures, notes = imagegen.picture_models()
    if pictures:
        engine.prepare_pictures(say)
        say("Picture models: " + ", ".join(item["label"] for item in pictures))
    else:
        say("No picture models yet. To make pictures in Chat, get FLUX.2 klein:  " + imagegen.GET_ONE)
    for note in notes:
        say("  " + note)


def prepare(url: str) -> None:
    """The first-start steps: the venvs, the notes model, what models there are. Shown in the
    terminal, since the first time installs packages."""
    from helper import engine, names, python_tool, store

    say("HikariLM " + __version__)
    moved = store.move_old_home()
    if moved:
        say("  " + moved)
    say(f"  page:   {url}")
    say(f"  saved:  {store.home()}")
    if engine.prepare(say):
        engine.prepare_embed(say)
    python_tool.prepare(say)
    mlx = engine.names()
    say("MLX models: " + ", ".join(names.label(name) for name in mlx) if mlx else
        "No MLX models downloaded yet. Get any MLX model, for example:  ~/.hikarilm/venv/bin/hf download "
        "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit")
    report_pictures()


def serve_here(background: bool) -> None:
    """Be the helper. In the background, everything it prints goes to the capped log."""
    from helper import daemon

    if background:
        os.environ["HIKARILM_BACKGROUND"] = "1"
        daemon.capture_output(daemon.CappedLog(daemon.log_path()))
    from helper.server import main as serve

    daemon.write_pid()
    try:
        serve()
    finally:
        daemon.forget_pid()


def stop_running(host: str, port: int, quiet: bool = False) -> bool:
    """Stop the HikariLM running in the background. True if one was stopped."""
    from helper import daemon

    state = running_state(host, port)
    pid = state.get("pid") if state else daemon.saved_pid()
    if not pid:
        if state is not None:
            sys.exit("A HikariLM from before 4.3.0 is running. Stop it with Ctrl+C in the window where it runs.")
        if not quiet:
            say("HikariLM is not running.")
        return False
    say("Stopping HikariLM…")
    if not daemon.stop(pid):
        sys.exit(f"HikariLM (process {pid}) did not stop. Stop it with:  kill {pid}")
    say("Stopped.")
    return True


def wait_until_up(host: str, port: int, child, seconds: float = 60) -> dict | None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        state = running_state(host, port)
        if state is not None:
            return state
        if child is not None and child.poll() is not None:
            return None
        time.sleep(0.3)
    return None


def show_log(lines: int = 40) -> None:
    from helper import daemon

    try:
        text = daemon.log_path().read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        say("There is no log yet.")
        return
    say("\n".join(text.splitlines()[-lines:]))
    say(f"\n(the last {lines} lines of {daemon.log_path()}; follow it with:  tail -f {daemon.log_path()})")


def install_command() -> None:
    from helper import command

    try:
        for line in command.install():
            say(line)
    except OSError as error:
        say(f"The hikarilm command could not be installed ({error}). ./start.sh does the same.")


def update_here(args) -> None:
    """hikarilm update: git pull, then restart with the new code if it brought anything."""
    from helper import update

    try:
        pulled = update.pull()
    except update.UpdateError as error:
        sys.exit(str(error))
    if not pulled["updated"]:
        say(f"HikariLM {__version__} is up to date.")
        return
    say(f"Updated {pulled['from']} to {pulled['to']}.")
    if running_state(args.host, args.port) is None:
        say("Start it with:  hikarilm start")
        return
    # This process still has the old code loaded; the new start.py does the restart.
    os.execv(sys.executable, [sys.executable, str(ROOT / "start.py"), "restart", "--host", args.host,
                              "--port", str(args.port), "--no-browser"])


def main() -> None:
    parser = argparse.ArgumentParser(prog="hikarilm", description="Run HikariLM.")
    parser.add_argument("action", nargs="?", default="start",
                        choices=("start", "stop", "restart", "status", "log", "update"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("HIKARILM_PORT", "8787")))
    parser.add_argument("--host", default=os.environ.get("HIKARILM_HOST", "127.0.0.1"))
    parser.add_argument("--no-browser", action="store_true", help="do not open the page")
    parser.add_argument("--foreground", action="store_true", help="run in this terminal until Ctrl+C")
    parser.add_argument("--serve", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--after", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--version", action="version", version="HikariLM " + __version__)
    args = parser.parse_args()

    check_python()
    check_files()
    # The helper reads these when it is imported, so set them first.
    os.environ["HIKARILM_HOST"] = args.host
    os.environ["HIKARILM_PORT"] = str(args.port)
    url = f"http://{args.host}:{args.port}"

    if args.serve:
        serve_here(background=True)
        return
    if args.action == "log":
        show_log()
        return
    if args.action == "status":
        state = running_state(args.host, args.port)
        if state is None:
            say("HikariLM is not running.")
        else:
            loaded = state.get("large_model")
            say(f"HikariLM {state.get('version', '')} is running at {url}" +
                (f" (process {state['pid']})" if state.get("pid") else "") +
                (", in the background." if state.get("background") else ", in a terminal.") +
                (f" Loaded: {loaded}." if loaded else ""))
        return
    if args.action == "stop":
        stop_running(args.host, args.port)
        return
    if args.action == "update":
        update_here(args)
        return
    if not args.after:
        install_command()
    if args.action == "restart":
        stop_running(args.host, args.port, quiet=True)

    if args.after:
        # Started by Update: the copy it replaces is stopping.
        from helper import daemon

        if not daemon.wait_gone(args.after, daemon.STOP_SECONDS + 10):
            sys.exit(f"The copy being updated (process {args.after}) did not stop, so the new one cannot start.")

    if not port_is_free(args.host, args.port):
        state = running_state(args.host, args.port)
        if state is None:
            sys.exit(f"Port {args.port} is used by another program. Try:  hikarilm start --port {args.port + 1}")
        from helper.version import code_stamp

        if state.get("code") == code_stamp():
            say(f"HikariLM is already running at {url}")
            if not args.no_browser:
                webbrowser.open(url)
            return
        # A copy started before the files changed keeps running its old code, so pages
        # added since would show "not found".
        say("A copy started before the files were updated is running. Restarting it.")
        stop_running(args.host, args.port)

    prepare(url)
    if args.foreground:
        say("Press Ctrl+C to stop.")
        say()
        if not args.no_browser:
            threading.Timer(0.8, webbrowser.open, args=(url,)).start()
        serve_here(background=False)
        return

    from helper import daemon

    child = daemon.spawn(["--serve", "--host", args.host, "--port", str(args.port)])
    if wait_until_up(args.host, args.port, child) is None:
        say("HikariLM did not start. The end of its log:")
        show_log(20)
        sys.exit(1)
    if args.after:
        say(f"Restarted after the update, at {url}.")
        return
    say()
    say(f"HikariLM is running in the background at {url}. You can close this window.")
    say("  hikarilm stop       stops it")
    say("  hikarilm status     says whether it runs, and what is loaded")
    say(f"  hikarilm log        shows its log ({daemon.log_path()}, at most {daemon.MAX_LINES} lines)")
    say("  hikarilm update     gets the newest HikariLM and restarts it")
    if not args.no_browser:
        webbrowser.open(url)


if __name__ == "__main__":
    main()
