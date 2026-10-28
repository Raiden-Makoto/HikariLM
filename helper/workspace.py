"""Read and change the files in one folder the user has opened.

The folder is named on every call and nothing is remembered here. Only files
that are already inside it, and of an everyday kind, can be read or written.
"""

from __future__ import annotations

import codecs
import hashlib
import os
import tempfile
from pathlib import Path

MAX_FILE = 500_000
MAX_FILES = 200
MAX_DEPTH = 4

EXTENSIONS = {
    ".txt", ".md", ".rst", ".log", ".csv", ".tsv", ".json", ".yaml", ".yml", ".toml",
    ".ini", ".xml", ".html", ".css",
    ".py", ".r", ".c", ".h", ".cpp", ".hpp", ".java", ".kt", ".go", ".rs", ".rb",
    ".php", ".swift", ".cs", ".scala", ".lua", ".pl", ".js", ".ts", ".jsx", ".tsx",
    ".sh", ".sql", ".tex",
}
NAMES = {"makefile", "dockerfile", "readme", "license"}
SKIP_DIRS = {"node_modules", "__pycache__", "venv"}


class WorkspaceError(ValueError):
    """A problem the user can fix, worded for them."""


class Conflict(WorkspaceError):
    """The file changed on disk after the page opened it."""


def open_root(raw: object) -> Path:
    if not isinstance(raw, str) or not raw.strip():
        raise WorkspaceError("Type the path of a folder to open.")
    root = Path(os.path.expanduser(raw.strip()))
    try:
        root = root.resolve(strict=True)
    except (FileNotFoundError, NotADirectoryError):
        raise WorkspaceError(f"There is no folder at {raw.strip()}.") from None
    except OSError as error:
        raise WorkspaceError(f"Cannot open {raw.strip()}: {error.strerror or error}.") from None
    if not root.is_dir():
        raise WorkspaceError(f"{raw.strip()} is a file, not a folder.")
    if not os.access(root, os.R_OK | os.X_OK):
        raise WorkspaceError(f"No permission to read {raw.strip()}.")
    return root


def browse(raw: object) -> dict:
    """List the folders inside one folder, so the page can be clicked through.

    Nothing is read from the files themselves, only their names.
    """
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        raw = "~"
    here = open_root(raw)
    folders: list[str] = []
    files = 0
    try:
        with os.scandir(here) as entries:
            for entry in entries:
                try:
                    if entry.is_dir():
                        if not entry.name.startswith("."):
                            folders.append(entry.name)
                    elif entry.is_file() and allowed(entry.name):
                        files += 1
                except OSError:
                    continue
    except OSError as error:
        raise WorkspaceError(f"Cannot list {here}: {error.strerror or error}.") from None
    folders.sort(key=str.lower)
    parent = None if here.parent == here else str(here.parent)
    return {
        "path": str(here),
        "parent": parent,
        "home": str(Path.home()),
        "folders": folders[:1000],
        "cut": len(folders) > 1000,
        "files": files,
    }


def allowed(name: str) -> bool:
    lower = name.lower()
    return Path(lower).suffix in EXTENSIONS or lower in NAMES


def listing(raw_root: object) -> dict:
    root = open_root(raw_root)
    found: list[str] = []
    cut = False

    def walk(folder: Path, depth: int) -> None:
        nonlocal cut
        try:
            entries = sorted(os.scandir(folder), key=lambda e: e.name.lower())
        except OSError:
            return
        for entry in entries:
            if cut:
                return
            if entry.is_symlink():
                continue
            if entry.is_dir():
                if entry.name.startswith(".") or entry.name in SKIP_DIRS or depth >= MAX_DEPTH:
                    continue
                walk(Path(entry.path), depth + 1)
            elif entry.is_file() and allowed(entry.name):
                if len(found) >= MAX_FILES:
                    cut = True
                    return
                found.append(Path(entry.path).relative_to(root).as_posix())

    walk(root, 0)
    return {"root": str(root), "name": root.name or str(root), "files": found, "cut": cut}


def _inside(raw_root: object, rel: object) -> tuple[Path, Path]:
    root = open_root(raw_root)
    if not isinstance(rel, str) or not rel or "\x00" in rel:
        raise WorkspaceError("Choose a file from the list.")
    if rel.startswith(("/", "\\")) or ".." in Path(rel.replace("\\", "/")).parts:
        raise WorkspaceError("That path leaves the opened folder.")
    target = root / rel
    try:
        resolved = target.resolve(strict=True)
    except (FileNotFoundError, NotADirectoryError):
        raise WorkspaceError(f"{rel} is not in the folder any more.") from None
    except OSError as error:
        raise WorkspaceError(f"Cannot open {rel}: {error.strerror or error}.") from None
    if root != resolved and root not in resolved.parents:
        raise WorkspaceError("That path leaves the opened folder.")
    if not resolved.is_file():
        raise WorkspaceError(f"{rel} is not a file.")
    if not allowed(resolved.name):
        raise WorkspaceError(f"{resolved.name} is not a kind of file this page opens.")
    return root, resolved


def _decode(raw: bytes, name: str) -> tuple[str, str, bytes]:
    """Return text, the codec, and the BOM it started with."""
    if raw.startswith(codecs.BOM_UTF8):
        return raw[3:].decode("utf-8"), "utf-8", codecs.BOM_UTF8
    if raw.startswith(codecs.BOM_UTF16_LE):
        return raw[2:].decode("utf-16-le"), "utf-16-le", codecs.BOM_UTF16_LE
    if raw.startswith(codecs.BOM_UTF16_BE):
        return raw[2:].decode("utf-16-be"), "utf-16-be", codecs.BOM_UTF16_BE
    if b"\x00" in raw:
        raise WorkspaceError(f"{name} has binary bytes, so it is not opened.")
    try:
        return raw.decode("utf-8"), "utf-8", b""
    except UnicodeDecodeError:
        pass
    try:
        return raw.decode("cp1252"), "cp1252", b""
    except UnicodeDecodeError:
        raise WorkspaceError(f"{name} is not text this page can read.") from None


def _version(raw: bytes) -> str:
    return hashlib.sha1(raw).hexdigest()


def _read_bytes(path: Path) -> bytes:
    try:
        size = path.stat().st_size
        if size > MAX_FILE:
            raise WorkspaceError(f"{path.name} is {size // 1000} KB. Files over 500 KB are not opened.")
        return path.read_bytes()
    except PermissionError:
        raise WorkspaceError(f"No permission to read {path.name}.") from None
    except OSError as error:
        raise WorkspaceError(f"Cannot read {path.name}: {error.strerror or error}.") from None


def read(raw_root: object, rel: object) -> dict:
    root, path = _inside(raw_root, rel)
    raw = _read_bytes(path)
    text, _, _ = _decode(raw, path.name)
    return {
        "path": path.relative_to(root).as_posix(),
        "text": text.replace("\r\n", "\n"),
        "version": _version(raw),
    }


def write(raw_root: object, rel: object, text: object, version: object) -> dict:
    root, path = _inside(raw_root, rel)
    if not isinstance(text, str):
        raise WorkspaceError("There is no text to save.")
    if not isinstance(version, str) or not version:
        raise WorkspaceError("The file's version is missing. Reload the file.")
    raw = _read_bytes(path)
    if _version(raw) != version:
        raise Conflict(f"{path.name} was changed on disk after you opened it.")
    old_text, codec, bom = _decode(raw, path.name)
    body = text.replace("\r\n", "\n")
    if "\r\n" in old_text:
        body = body.replace("\n", "\r\n")
    try:
        data = bom + body.encode(codec)
    except UnicodeEncodeError:
        raise WorkspaceError(
            f"{path.name} is saved as {codec}, which cannot hold a character you typed."
        ) from None
    if len(data) > MAX_FILE:
        raise WorkspaceError("The file would be over 500 KB, so it was not saved.")
    if data == raw:
        return {"version": version, "changed": False}
    try:
        mode = path.stat().st_mode
        fd, temp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
            os.chmod(temp, mode & 0o7777)
            os.replace(temp, path)
        except BaseException:
            try:
                os.unlink(temp)
            except OSError:
                pass
            raise
    except PermissionError:
        raise WorkspaceError(f"No permission to write {path.name}.") from None
    except OSError as error:
        raise WorkspaceError(f"Could not save {path.name}: {error.strerror or error}.") from None
    return {"version": _version(data), "changed": True}


def _new_target(raw_root: object, rel: object) -> tuple[Path, Path]:
    root = open_root(raw_root)
    if not isinstance(rel, str) or not rel.strip() or "\x00" in rel:
        raise WorkspaceError("Type a name for the new file.")
    rel = rel.strip().replace("\\", "/")
    parts = Path(rel).parts
    if rel.startswith("/") or not parts or any(part in ("..", ".") for part in parts):
        raise WorkspaceError("That path leaves the opened folder.")
    if any(part.startswith(".") for part in parts):
        raise WorkspaceError("Names that start with a dot are not shown here, so they are not created.")
    if not allowed(parts[-1]):
        raise WorkspaceError(f"{parts[-1]} is not a kind of file this page opens, for example .py, .md or .txt.")
    target = root.joinpath(*parts)
    # Check the nearest folder that already exists, so a link cannot lead out.
    existing = target.parent
    while not existing.exists() and existing != root:
        existing = existing.parent
    resolved = existing.resolve()
    if root != resolved and root not in resolved.parents:
        raise WorkspaceError("That path leaves the opened folder.")
    return root, target


def create(raw_root: object, rel: object) -> dict:
    """Make a new, empty file. An existing file is never replaced."""
    root, target = _new_target(raw_root, rel)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "xb"):
            pass
    except FileExistsError:
        raise WorkspaceError(f"{target.name} already exists in that folder.") from None
    except PermissionError:
        raise WorkspaceError(f"No permission to create {target.name} there.") from None
    except OSError as error:
        raise WorkspaceError(f"Could not create {target.name}: {error.strerror or error}.") from None
    return {"path": target.relative_to(root).as_posix()}


def remove(raw_root: object, rel: object) -> dict:
    """Delete one file. Folders are never deleted."""
    _, path = _inside(raw_root, rel)
    try:
        path.unlink()
    except PermissionError:
        raise WorkspaceError(f"No permission to delete {path.name}.") from None
    except OSError as error:
        raise WorkspaceError(f"Could not delete {path.name}: {error.strerror or error}.") from None
    return {"deleted": True}
