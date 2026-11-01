"""The hikarilm command: a link to bin/hikarilm in ~/.local/bin, put on the PATH once."""

from __future__ import annotations

import os
from pathlib import Path

from helper.daemon import ROOT

SCRIPT = ROOT / "bin" / "hikarilm"
MARK = "# Added by HikariLM, for the hikarilm command"


def bin_dir() -> Path:
    return Path.home() / ".local" / "bin"


def shell_file() -> Path:
    """The file the user's login shell reads: zsh on a Mac unless they changed it."""
    shell = os.path.basename(os.environ.get("SHELL", "zsh"))
    return Path.home() / (".bash_profile" if shell == "bash" else ".zshrc")


def on_path(folder: Path) -> bool:
    return any(Path(part).expanduser() == folder for part in os.environ.get("PATH", "").split(os.pathsep) if part)


def install() -> list[str]:
    """Make sure the hikarilm command is there. Returns what was done, for the start window;
    nothing when it already was."""
    said = []
    SCRIPT.chmod(SCRIPT.stat().st_mode | 0o111)
    link = bin_dir() / "hikarilm"
    if link.is_symlink() and link.resolve() == SCRIPT.resolve():
        pass
    elif link.exists() and not link.is_symlink():
        return [f"{link} is another program, so the hikarilm command was not installed. Use ./start.sh instead."]
    else:
        moved = link.is_symlink()
        bin_dir().mkdir(parents=True, exist_ok=True)
        if moved:
            link.unlink()
        link.symlink_to(SCRIPT)
        said.append(f"The hikarilm command now runs this copy ({ROOT})." if moved
                    else "Installed the hikarilm command: hikarilm start, stop, restart, status, log, update.")
    if not on_path(bin_dir()):
        rc = shell_file()
        try:
            text = rc.read_text(encoding="utf-8")
        except FileNotFoundError:
            text = ""
        if MARK not in text:
            with open(rc, "a", encoding="utf-8") as handle:
                handle.write(("" if not text or text.endswith("\n") else "\n") +
                             f'\n{MARK}\nexport PATH="$HOME/.local/bin:$PATH"\n')
            said.append(f"Added ~/.local/bin to the PATH in {rc}. Open a new Terminal window to use hikarilm.")
    return said
