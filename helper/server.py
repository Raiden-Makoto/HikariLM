"""Serve the page and stream one Ollama chat at a time."""

from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from helper import store, workspace
from helper.ollama import KEEP_ALIVE, OllamaError, open_chat, tags, unload
from helper.router import ModelRouter, chat_models
from helper.version import code_stamp

HOST = os.environ.get("LIGHTUSAGI_HOST", "127.0.0.1")
PORT = int(os.environ.get("LIGHTUSAGI_PORT", "8787"))
WEB_ROOT = Path(__file__).resolve().parent.parent / "web"
MAX_BODY = 48_000_000
MAX_TRANSCRIPT = 20_000_000
CHATS = "/api/chats"
DRAFTS = "/api/drafts"
WORKSPACE = "/api/workspace"
# One page, three addresses: the page shows the section the address names.
PAGES = ("/", "/index.html", "/chat", "/chat/", "/editor", "/editor/")
VENDOR = ("/vendor/pdf.min.mjs", "/vendor/pdf.worker.min.mjs", "/vendor/codemirror.js")

router = ModelRouter(unload)
STARTED_WITH = code_stamp()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:
        if not self._same_origin() or self._workspace("GET"):
            return
        if self.path == "/api/state":
            self._send_json({"large_model": router.large, "code": STARTED_WITH})
            return
        if self.path == "/api/tags":
            try:
                self._send_json({"models": chat_models(tags())})
            except OllamaError as error:
                self._send_json({"error": str(error)}, status=502)
            return
        if self.path == CHATS:
            self._send_json({"chats": store.listing()})
            return
        if self.path.startswith(DRAFTS + "/"):
            try:
                self._send_json(store.load_draft(self._chat_id()))
            except FileNotFoundError:
                self._send_json({"error": "no draft"}, status=404)
            except ValueError as error:
                self._send_json({"error": str(error)}, status=400)
            return
        if self.path.startswith(CHATS + "/"):
            try:
                self._send_json(store.load(self._chat_id()))
            except FileNotFoundError:
                self._send_json({"error": "chat not found"}, status=404)
            except ValueError as error:
                self._send_json({"error": str(error)}, status=400)
            return
        if urlparse(self.path).path in PAGES:
            self._send_file(WEB_ROOT / "index.html", "text/html; charset=utf-8")
            return
        if self.path in VENDOR:
            self._send_file(WEB_ROOT / "vendor" / self.path.rsplit("/", 1)[1], "text/javascript; charset=utf-8")
            return
        self._send_json({"error": "not found"}, status=404)

    def do_PUT(self) -> None:
        if not self._same_origin() or self._workspace("PUT"):
            return
        if self.path.startswith(DRAFTS + "/"):
            try:
                payload = self._body(MAX_BODY)
                kept = store.save_draft(self._chat_id(), payload.get("text"), payload.get("files"))
            except ValueError as error:
                self._send_json({"error": str(error)}, status=400)
                return
            except OSError as error:
                self._send_json({"error": f"could not save: {error.strerror or error}"}, status=500)
                return
            self._send_json({"kept": kept})
            return
        if not self.path.startswith(CHATS + "/"):
            self._send_json({"error": "not found"}, status=404)
            return
        try:
            payload = self._body(MAX_TRANSCRIPT)
            summary = store.save(self._chat_id(), payload.get("model"), payload.get("messages"))
        except ValueError as error:
            self._send_json({"error": str(error)}, status=400)
            return
        except OSError as error:
            self._send_json({"error": f"could not save: {error.strerror or error}"}, status=500)
            return
        self._send_json(summary)

    def do_PATCH(self) -> None:
        if not self._same_origin():
            return
        if self.path == "/api/folders":
            try:
                payload = self._body(MAX_BODY)
                self._send_json(store.rename_folder(payload.get("from"), payload.get("to")))
            except ValueError as error:
                self._send_json({"error": str(error)}, status=400)
            except OSError as error:
                self._send_json({"error": f"could not change folder: {error.strerror or error}"}, status=500)
            return
        if not self.path.startswith(CHATS + "/"):
            self._send_json({"error": "not found"}, status=404)
            return
        try:
            payload = self._body(MAX_BODY)
            self._send_json(store.patch(self._chat_id(), payload))
        except FileNotFoundError:
            self._send_json({"error": "chat not found"}, status=404)
        except ValueError as error:
            self._send_json({"error": str(error)}, status=400)
        except OSError as error:
            self._send_json({"error": f"could not rename: {error.strerror or error}"}, status=500)

    def do_DELETE(self) -> None:
        if not self._same_origin() or self._workspace("DELETE"):
            return
        if self.path.startswith(DRAFTS + "/"):
            store.delete_draft(self._chat_id())
            self._send_json({"deleted": True})
            return
        if not self.path.startswith(CHATS + "/"):
            self._send_json({"error": "not found"}, status=404)
            return
        try:
            removed = store.delete(self._chat_id())
        except ValueError as error:
            self._send_json({"error": str(error)}, status=400)
            return
        except OSError as error:
            self._send_json({"error": f"could not delete: {error.strerror or error}"}, status=500)
            return
        self._send_json({"deleted": removed}, status=200 if removed else 404)

    def do_POST(self) -> None:
        if not self._same_origin() or self._workspace("POST"):
            return
        if self.path != "/api/chat":
            self._send_json({"error": "not found"}, status=404)
            return
        try:
            model, messages, keep_alive = self._read_chat()
        except ValueError as error:
            self._send_json({"error": str(error)}, status=400)
            return
        try:
            router.begin(model)
            response = open_chat(model, messages, keep_alive)
        except OllamaError as error:
            self._send_json({"error": str(error)}, status=502)
            return
        router.commit(model)
        self._stream(response)

    def _same_origin(self) -> bool:
        # A page from another address must not be able to read or change files.
        origin = self.headers.get("Origin")
        if origin and urlparse(origin).netloc != self.headers.get("Host", ""):
            self._send_json({"error": "requests from other sites are refused"}, status=403)
            return False
        return True

    def _workspace(self, method: str) -> bool:
        parts = urlparse(self.path)
        if parts.path not in (WORKSPACE, WORKSPACE + "/file", WORKSPACE + "/browse"):
            return False
        query = parse_qs(parts.query)
        try:
            if method == "GET" and parts.path.endswith("/browse"):
                self._send_json(workspace.browse(query.get("path", [""])[0]))
            elif method == "GET" and parts.path == WORKSPACE:
                self._send_json(workspace.listing(query.get("root", [""])[0]))
            elif method == "GET":
                self._send_json(workspace.read(query.get("root", [""])[0], query.get("path", [""])[0]))
            elif method == "PUT" and parts.path.endswith("/file"):
                payload = self._body(MAX_BODY)
                self._send_json(workspace.write(
                    payload.get("root"), payload.get("path"), payload.get("text"), payload.get("version")))
            elif method == "POST" and parts.path.endswith("/file"):
                payload = self._body(MAX_BODY)
                self._send_json(workspace.create(payload.get("root"), payload.get("path")), status=201)
            elif method == "DELETE" and parts.path.endswith("/file"):
                self._send_json(workspace.remove(query.get("root", [""])[0], query.get("path", [""])[0]))
            else:
                self._send_json({"error": "not found"}, status=404)
        except workspace.Conflict as error:
            self._send_json({"error": str(error), "conflict": True}, status=409)
        except ValueError as error:
            self._send_json({"error": str(error)}, status=400)
        return True

    def _chat_id(self) -> str:
        return self.path.split("?", 1)[0].rsplit("/", 1)[1]

    def _body(self, limit: int) -> dict:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as error:
            raise ValueError("body length is not valid") from error
        if length <= 0 or length > limit:
            raise ValueError("body is missing or too large")
        try:
            payload = json.loads(self.rfile.read(length).decode())
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise ValueError("body is not JSON") from error
        if not isinstance(payload, dict):
            raise ValueError("body must be a JSON object")
        return payload

    def _read_chat(self) -> tuple[str, list[dict], str]:
        payload = self._body(MAX_BODY)
        model = payload.get("model")
        messages = payload.get("messages")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model is required")
        if not isinstance(messages, list) or not messages:
            raise ValueError("messages are required")
        cleaned = []
        for message in messages:
            if not isinstance(message, dict):
                raise ValueError("each message needs a role and content")
            role = message.get("role")
            content = message.get("content")
            if role not in ("system", "user", "assistant") or not isinstance(content, str):
                raise ValueError("each message needs a role and content")
            entry = {"role": role, "content": content}
            images = message.get("images")
            if images is not None:
                if not isinstance(images, list) or not all(isinstance(i, str) for i in images):
                    raise ValueError("images must be a list of base64 strings")
                if images:
                    entry["images"] = images
            cleaned.append(entry)
        keep_alive = payload.get("keep_alive", KEEP_ALIVE)
        if not isinstance(keep_alive, str) or not keep_alive.strip():
            raise ValueError("keep_alive must be a duration such as 5m")
        return model.strip(), cleaned, keep_alive.strip()

    def _stream(self, response) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        try:
            # Ollama sends one JSON object per line. read(n) would wait for n
            # bytes and hold tokens back, so pass each line on as it arrives.
            while True:
                line = response.readline()
                if not line:
                    break
                self.wfile.write(line)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            return
        finally:
            response.close()

    def _send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)
        self.close_connection = True

    def _send_file(self, path: Path, content_type: str) -> None:
        if not path.is_file():
            self._send_json({"error": "page is missing"}, status=500)
            return
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        # Always ask again, so an updated page is picked up on the next reload.
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)
        self.close_connection = True

    def log_message(self, fmt: str, *args) -> None:
        print(f"{self.address_string()} {fmt % args}", flush=True)


def main() -> None:
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"LightUsagi at http://{HOST}:{PORT}", flush=True)
    print(f"Ollama at {os.environ.get('OLLAMA_HOST', 'http://127.0.0.1:11434')}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping", flush=True)
    finally:
        server.server_close()
