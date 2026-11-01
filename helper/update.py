"""Updating HikariLM from GitHub with git, for the Update button in Settings."""

from __future__ import annotations

import shutil
import subprocess

from helper.daemon import ROOT


class UpdateError(Exception):
    """Why an update could not be done, worded for the user."""


def _git(*arguments: str, seconds: int = 60) -> subprocess.CompletedProcess:
    if shutil.which("git") is None:
        raise UpdateError("git is not installed, so HikariLM cannot update itself. Install it with:  xcode-select --install")
    try:
        return subprocess.run(["git", *arguments], cwd=ROOT, capture_output=True, text=True, timeout=seconds)
    except subprocess.TimeoutExpired:
        raise UpdateError("git took too long. Check the internet connection and try again.") from None


def _last_lines(text: str, count: int = 4) -> str:
    return " ".join(line.strip() for line in text.strip().splitlines()[-count:])


def is_git_copy() -> bool:
    return (ROOT / ".git").exists()


def head() -> str:
    return _git("rev-parse", "--short", "HEAD").stdout.strip()


def status() -> dict:
    """Whether this copy can update itself, and how many changes GitHub has that it does not."""
    if not is_git_copy():
        return {"git": False}
    out = {"git": True, "head": head(), "behind": None}
    fetched = _git("fetch", "--quiet")
    if fetched.returncode != 0:
        out["error"] = "Could not reach GitHub: " + _last_lines(fetched.stderr)
        return out
    behind = _git("rev-list", "--count", "HEAD..@{u}")
    if behind.returncode == 0 and behind.stdout.strip().isdigit():
        out["behind"] = int(behind.stdout.strip())
    out["changed"] = bool(_git("status", "--porcelain", "--untracked-files=no").stdout.strip())
    return out


def pull() -> dict:
    """git pull, only when it can fast-forward. Says whether anything changed."""
    if not is_git_copy():
        raise UpdateError("This copy of HikariLM was not made with git clone, so it cannot update itself. "
                          "Download the new release from GitHub instead.")
    before = head()
    pulled = _git("pull", "--ff-only", "--quiet", seconds=120)
    if pulled.returncode != 0:
        raise UpdateError("git could not update: " + _last_lines(pulled.stderr or pulled.stdout) +
                          " Files changed by hand in the HikariLM folder can stop it; see git status there.")
    after = head()
    return {"updated": before != after, "from": before, "to": after}
