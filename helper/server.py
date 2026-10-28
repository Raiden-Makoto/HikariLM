"""Serve the page and stream one Ollama chat at a time."""

from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from helper import budget, context, descriptions, notes, store, workspace
from helper.ollama import (KEEP_ALIVE, OllamaError, complete, embed, open_chat, running, show, sizes, tags,
                           unload)
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
indexer = notes.Indexer(embed)
STARTED_WITH = code_stamp()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:
        if not self._same_origin() or self._workspace("GET"):
            return
        if self.path == "/api/state":
            self._send_json({"large_model": router.large, "code": STARTED_WITH})
            return
        if self.path == "/api/settings":
            self._send_json(context.load_settings())
            return
        if self.path.startswith("/api/context?"):
            model = parse_qs(urlparse(self.path).query).get("model", [""])[0]
            out = {"memory": None, "model": model}
            mem = budget.memory()
            if mem:
                out["memory"] = {"total": mem[0], "available": mem[1]}
            try:
                settings = context.load_settings()
                room = self._room(model, settings)
                out.update(room)
                out["threshold"] = context.threshold_for(settings, room["num_ctx"])
            except (budget.NoRoom, OllamaError) as error:
                out["error"] = str(error)
            self._send_json(out)
            return
        if self.path == "/api/notes":
            self._send_json({**notes.count(), "error": indexer.last_error})
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
        if self.path == "/api/settings":
            try:
                self._send_json(context.save_settings(self._body(MAX_BODY)))
            except ValueError as error:
                self._send_json({"error": str(error)}, status=400)
            except OSError as error:
                self._send_json({"error": f"could not save: {error.strerror or error}"}, status=500)
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
        indexer.chat(summary["id"])
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
            indexer.chat(self._chat_id())
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
        indexer.forget_chat(self._chat_id())
        self._send_json({"deleted": removed}, status=200 if removed else 404)

    def do_POST(self) -> None:
        if not self._same_origin() or self._workspace("POST"):
            return
        if self.path == "/api/describe":
            self._describe()
            return
        if self.path == "/api/search":
            self._search()
            return
        if self.path != "/api/chat":
            self._send_json({"error": "not found"}, status=404)
            return
        try:
            model, messages, keep_alive = self._read_chat()
        except ValueError as error:
            self._send_json({"error": str(error)}, status=400)
            return
        settings = context.load_settings()
        try:
            room = self._room(model, settings)
        except budget.NoRoom as error:
            self._send_json({"error": str(error), "memory": True}, status=400)
            return
        except OllamaError as error:
            self._send_json({"error": str(error)}, status=502)
            return
        num_ctx = room["num_ctx"]
        decision = context.compaction_needed(messages, settings, num_ctx)
        system = [m for m in messages[:1] if m["role"] == "system"]
        active = system + messages[decision["start"]:]
        summary = decision["summary"]
        self._stream_start()
        started = True
        self._stream_line({"lightusagi": "context", **room, "tokens": decision["tokens"],
                           "threshold": decision["threshold"]})
        if decision["cut"] is not None:
            # Like Open WebUI: summarize the older part with the same model, keep the rest.
            self._stream_line({"lightusagi": "compaction", "done": False})
            cut = decision["cut"]
            compacted = messages[decision["start"]:cut]
            recent = messages[cut:]
            max_tokens = settings["compaction"]["max_tokens"]
            prompt = context.summary_prompt(settings["compaction"]["prompt"], compacted, recent, summary,
                                            num_ctx - max_tokens - 256)
            try:
                router.begin(model)
                written = complete(model, [{"role": "user", "content": prompt}],
                                   {"num_ctx": num_ctx, "num_predict": max_tokens}, keep_alive)
                router.commit(model)
                summary = written.strip() or context.fallback_summary(summary, compacted)
                active = system + recent
                self._stream_line({"lightusagi": "compaction", "done": True, "checkpoint": cut,
                                   "summary": summary, "dropped": len(compacted), "kept": len(recent)})
            except OllamaError as error:
                # Like Open WebUI: a failed compaction sends the whole active window instead.
                self._stream_line({"lightusagi": "compaction", "done": True, "error": str(error)})
        if getattr(self, "_gone", False):
            return
        try:
            router.begin(model)
            response = open_chat(model, context.with_summary(active, summary), keep_alive, {"num_ctx": num_ctx})
        except OllamaError as error:
            if started:
                self._stream_line({"error": str(error)})
            return
        router.commit(model)
        self._stream(response, started=True)

    def _room(self, model: str, settings: dict) -> dict:
        """How much context this model gets right now. Raises budget.NoRoom if it does not fit at all."""
        shown = show(model)
        size = sizes().get(model, 0)
        return budget.plan(model, context.requested_context(settings, model), shown, size, running(),
                           budget.memory())

    def _describe(self) -> None:
        """Have a vision model write a picture down as text, once.

        Streams what Ollama sends so the page can show progress and stop, then
        one last line, {"description": ...}. A picture already described is
        answered from descriptions/ without touching Ollama at all.
        """
        try:
            payload = self._body(MAX_BODY)
            model = payload.get("model")
            if not isinstance(model, str) or not model.strip():
                raise ValueError("model is required")
            model = model.strip()
            name = payload.get("name") if isinstance(payload.get("name"), str) else "picture"
            image = payload.get("image")
            key = descriptions.image_key(image)
        except ValueError as error:
            self._send_json({"error": str(error)}, status=400)
            return
        saved = descriptions.load(key)
        if saved:
            self._stream_start()
            self._stream_line({"description": saved["text"], "cached": True, "model": saved.get("model", "")})
            return
        try:
            room = self._room(model, context.load_settings())
            router.begin(model)
            response = open_chat(
                model, [{"role": "user", "content": descriptions.PROMPT, "images": [image]}], KEEP_ALIVE,
                {"num_ctx": room["num_ctx"]})
        except budget.NoRoom as error:
            self._send_json({"error": str(error), "memory": True}, status=400)
            return
        except OllamaError as error:
            self._send_json({"error": str(error)}, status=502)
            return
        router.commit(model)
        self._stream_start()
        text = ""
        try:
            while True:
                line = response.readline()
                if not line:
                    break
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                if event.get("error"):
                    self._stream_line({"error": str(event["error"])})
                    return
                text += (event.get("message") or {}).get("content", "")
                # Only progress goes to the page; the answer is sent once, at the end.
                self._stream_line({"progress": len(text), "text": (event.get("message") or {}).get("content", "")})
                if event.get("done"):
                    break
            text = text.strip()
            if not text:
                self._stream_line({"error": "The model sent no description of the picture."})
                return
            descriptions.save(key, text, model, name)
            indexer.picture(key)
            self._stream_line({"description": text[:descriptions.MAX_TEXT], "cached": False, "model": model})
        except (BrokenPipeError, ConnectionResetError):
            return
        except OSError as error:
            self._stream_line({"error": f"The description could not be saved: {error.strerror or error}"})
        finally:
            response.close()

    def _search(self) -> None:
        """A few earlier notes close to a question. Never an error: a send must not wait on this."""
        try:
            payload = self._body(MAX_BODY)
            query = payload.get("query")
            exclude = payload.get("exclude")
            if not isinstance(query, str):
                raise ValueError("query is required")
            source = notes.chat_source(exclude) if isinstance(exclude, str) and exclude else ""
        except ValueError as error:
            self._send_json({"error": str(error)}, status=400)
            return
        try:
            found = notes.search(query, embed, exclude=source)
        except (OllamaError, ValueError, OSError) as error:
            self._send_json({"notes": [], "error": str(error)})
            return
        self._send_json({"notes": found})

    def _stream_start(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True

    def _stream_line(self, payload: dict) -> bool:
        """Send one line. False once the page has gone (Stop, or the tab closed)."""
        if getattr(self, "_gone", False):
            return False
        try:
            self.wfile.write(json.dumps(payload).encode() + b"\n")
            self.wfile.flush()
            return True
        except (BrokenPipeError, ConnectionResetError):
            self._gone = True
            return False

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
            summary = store.clean_summary(message.get("contextSummary"))
            if summary:
                entry["contextSummary"] = summary
            usage = store.clean_usage(message.get("usage"))
            if usage:
                entry["usage"] = usage
            cleaned.append(entry)
        keep_alive = payload.get("keep_alive", KEEP_ALIVE)
        if not isinstance(keep_alive, str) or not keep_alive.strip():
            raise ValueError("keep_alive must be a duration such as 5m")
        return model.strip(), cleaned, keep_alive.strip()

    def _stream(self, response, started: bool = False) -> None:
        if not started:
            self._stream_start()
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
    indexer.everything()
    print(f"LightUsagi at http://{HOST}:{PORT}", flush=True)
    print(f"Ollama at {os.environ.get('OLLAMA_HOST', 'http://127.0.0.1:11434')}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping", flush=True)
    finally:
        server.server_close()
