"""Serve the page and stream one Ollama chat at a time."""

from __future__ import annotations

import json
import os
import signal
import threading
from contextlib import ExitStack
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from helper import budget, context, descriptions, imagegen, keepalive, notes, sdserver, store, thinking, workspace
from helper.ollama import (KEEP_ALIVE, KEEPER, OllamaError, complete, embed, open_chat, open_generate, running,
                           show, sizes, tags, unload)
from helper.router import ModelRouter, Work, chat_models, is_nomic
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
VENDOR = ("/vendor/pdf.min.mjs", "/vendor/pdf.worker.min.mjs", "/vendor/codemirror.js", "/vendor/marked.esm.js",
          "/vendor/katex.min.js")
IMAGES = "/api/images"

managed = sdserver.Managed(KEEP_ALIVE)


def unload_any(model: str) -> None:
    """Free a large model, wherever it runs: sd-server is stopped, Ollama is asked."""
    if model.startswith(imagegen.SD_PREFIX):
        managed.stop()
    elif not model.startswith(imagegen.REMOTE_PREFIX):
        unload(model)


router = ModelRouter(unload_any)
work = Work()
indexer = notes.Indexer(embed)
STARTED_WITH = code_stamp()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:
        if not self._same_origin() or self._workspace("GET"):
            return
        if self.path == "/api/state":
            self._send_json({**self._loaded(), "code": STARTED_WITH})
            return
        if self.path == "/api/pictures":
            self._send_json(self._picture_setup())
            return
        if self.path.startswith(IMAGES + "/") and self.path.endswith(".png"):
            try:
                self._send_bytes(imagegen.read(self.path[len(IMAGES) + 1:-4]), "image/png")
            except (FileNotFoundError, ValueError):
                self._send_json({"error": "picture not found"}, status=404)
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
            except budget.NoRoom as error:
                out["error"] = str(error)
                out["default"], out["default_from"] = budget.default_source(show(model))
            except OllamaError as error:
                out["error"] = str(error)
            self._send_json(out)
            return
        if self.path == "/api/notes":
            self._send_json({**notes.count(), "error": indexer.last_error})
            return
        if self.path == "/api/tags":
            # Picture models are listed apart: they answer with a picture, never with words.
            settings = context.load_settings()
            try:
                chat, ollama_images = imagegen.split(chat_models(tags()), show)
            except OllamaError as error:
                pictures, notes_ = imagegen.picture_models(settings, [])
                self._send_json({"error": str(error), "pictures": pictures, "picture_notes": notes_,
                                 "sizes": imagegen.SIZES}, status=502)
                return
            pictures, notes_ = imagegen.picture_models(settings, ollama_images)
            self._send_json({"models": chat, "pictures": pictures, "picture_notes": notes_,
                             "sizes": imagegen.SIZES})
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
        if urlparse(self.path).path.rstrip("/") == "/generate":
            # Pictures used to have their own section here; they are made in Chat now.
            self.send_response(301)
            self.send_header("Location", "/chat")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        vendor = urlparse(self.path).path
        if vendor in VENDOR:
            self._send_file(WEB_ROOT / "vendor" / vendor.rsplit("/", 1)[1], "text/javascript; charset=utf-8")
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
        if self.path.startswith(IMAGES + "/"):
            try:
                removed = imagegen.delete(self.path[len(IMAGES) + 1:])
            except ValueError as error:
                self._send_json({"error": str(error)}, status=400)
                return
            except OSError as error:
                self._send_json({"error": f"could not delete: {error.strerror or error}"}, status=500)
                return
            self._send_json({"deleted": removed}, status=200 if removed else 404)
            return
        if not self.path.startswith(CHATS + "/"):
            self._send_json({"error": "not found"}, status=404)
            return
        made = store.picture_ids(self._chat_id())
        try:
            removed = store.delete(self._chat_id())
        except ValueError as error:
            self._send_json({"error": str(error)}, status=400)
            return
        except OSError as error:
            self._send_json({"error": f"could not delete: {error.strerror or error}"}, status=500)
            return
        indexer.forget_chat(self._chat_id())
        if removed:
            # The pictures a chat made belong to it, so they go with it.
            for image_id in made:
                try:
                    imagegen.delete(image_id)
                except (OSError, ValueError):
                    pass
        self._send_json({"deleted": removed}, status=200 if removed else 404)

    def do_POST(self) -> None:
        if not self._same_origin() or self._workspace("POST"):
            return
        if self.path == "/api/generate":
            self._generate()
            return
        if self.path == "/api/search":
            self._search()
            return
        if self.path not in ("/api/chat", "/api/describe"):
            self._send_json({"error": "not found"}, status=404)
            return
        busy = work.start_chat()
        if busy:
            self._send_json({"error": busy, "busy": True}, status=409)
            return
        try:
            # The keepalive check leaves a model alone while a request here is using it.
            with ExitStack() as self._models:
                if self.path == "/api/describe":
                    self._describe()
                else:
                    self._chat()
        finally:
            work.end_chat()

    def _refuse_picture_model(self, model: str) -> bool:
        """Image models make pictures and never answer messages. True if refused."""
        if not model.startswith((imagegen.SD_PREFIX, imagegen.REMOTE_PREFIX)):
            try:
                shown = show(model)
            except OllamaError:
                shown = None
            if not imagegen.is_image_model(model, shown):
                return False
        self._send_json({"error": f"{model} makes pictures and cannot answer messages. "
                                  "Pick a chat model to talk, or send it a description of a picture."},
                        status=400)
        return True

    def _chat(self) -> None:
        try:
            model, messages, keep_alive = self._read_chat()
        except ValueError as error:
            self._send_json({"error": str(error)}, status=400)
            return
        if self._refuse_picture_model(model):
            return
        self._models.enter_context(KEEPER.using(model, keep_alive))
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
        think = thinking.think_value(model, show(model), settings["thinking"])
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
                # Thinking would spend the summary's tokens before any summary is written.
                written = complete(model, [{"role": "user", "content": prompt}],
                                   {"num_ctx": num_ctx, "num_predict": max_tokens}, keep_alive,
                                   think=None if think is None else thinking.off(model))
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
        sent = context.with_summary(active, summary)
        options = {"num_ctx": num_ctx}
        try:
            router.begin(model)
            response = open_chat(model, sent, keep_alive, options, think)
        except OllamaError as error:
            if started:
                self._stream_line({"error": str(error)})
            return
        router.commit(model)
        budget_tokens = settings["thinking"]["budget"]
        thought = self._stream_reply(response, budget_tokens)
        if thought is None or getattr(self, "_gone", False):
            return
        # Thinking reached the limit before any answer: ask again with thinking off and
        # the reasoning so far, so the model answers now.
        self._stream_line({"lightusagi": "thinking_cut", "budget": budget_tokens})
        try:
            response = open_chat(model, thinking.answer_now(sent, thought), keep_alive, options,
                                 None if think is None else thinking.off(model))
        except OllamaError as error:
            self._stream_line({"error": str(error)})
            return
        if self._stream_reply(response, budget_tokens) is not None:
            # It kept thinking even when told not to; end the reply here.
            self._stream_line({"lightusagi": "thinking_cut", "budget": budget_tokens, "final": True})
            self._stream_line({"message": {"role": "assistant", "content": ""}, "done": True})

    def _stream_reply(self, response, budget_tokens: int | None) -> str | None:
        """Pass a chat stream to the page, with inline <think> text moved to `thinking`.
        Returns the reasoning if thinking reached budget_tokens before any answer (the
        stream is stopped there), otherwise None. Each streamed piece is about one token."""
        split = thinking.InlineThink()
        thought, pieces, answered = "", 0, False
        try:
            while True:
                line = response.readline()
                if not line:
                    return None
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                message = event.get("message")
                if isinstance(message, dict):
                    inline, content = split.feed(str(message.get("content") or ""))
                    if event.get("done"):
                        more_thought, more_content = split.flush()
                        inline, content = inline + more_thought, content + more_content
                    reasoning = str(message.get("thinking") or "") + inline
                    event["message"] = {**message, "content": content, "thinking": reasoning}
                    if reasoning:
                        pieces += 1
                        thought += reasoning
                    if content.strip():
                        answered = True
                if not self._stream_line(event):
                    return None
                if budget_tokens and not answered and pieces >= budget_tokens and not event.get("done"):
                    return thought
        finally:
            response.close()

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
        if self._refuse_picture_model(model):
            return
        self._models.enter_context(KEEPER.using(model))
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

    def _generate(self) -> None:
        """Make one picture. Streams {"lightusagi":"loading"}, then {"progress", "total"} lines,
        then {"image": notes} once it is saved. Stop (the page going away) saves nothing.

        The model id says where the picture is made: "sd:<name>" is an sd-server the
        helper starts, "remote:<name>" a picture server the user runs, anything else Ollama.
        """
        try:
            request = imagegen.clean_request(self._body(MAX_BODY))
        except ValueError as error:
            self._send_json({"error": str(error)}, status=400)
            return
        model = request["model"]
        settings = context.load_settings()
        pictures = settings["pictures"]
        spec = program = None
        try:
            try:
                loaded = running()
            except OllamaError:
                loaded = []
            if model.startswith(imagegen.SD_PREFIX):
                spec = imagegen.find_sd_model(settings, model)
                if spec is None:
                    raise ValueError(f"{model[len(imagegen.SD_PREFIX):]} is not in the picture models in Settings.")
                program = sdserver.find_program(pictures["program"])
                if not program:
                    raise ValueError("sd-server was not found. Set where it is in Settings, under Pictures.")
                if not sdserver.model_files(spec["args"]):
                    raise ValueError(f"None of the model files for {spec['name']} were found. Check its line in Settings.")
                budget.check_picture(spec["name"], sdserver.model_size(spec["args"]), loaded, budget.memory())
            elif model.startswith(imagegen.REMOTE_PREFIX):
                if not pictures["address"]:
                    raise ValueError("No picture server address is set in Settings.")
            else:
                allowed = imagegen.support()
                if not allowed["supported"]:
                    self._send_json({"error": allowed["reason"], "unsupported": True}, status=403)
                    return
                if model not in chat_models(tags()):
                    raise ValueError(f"{model} is not installed in Ollama.")
                if not imagegen.is_image_model(model, show(model)):
                    raise ValueError(f"{model} answers messages and cannot make pictures. Choose a picture model.")
                budget.check_picture(model, sizes().get(model, 0), loaded, budget.memory())
        except budget.NoRoom as error:
            self._send_json({"error": str(error), "memory": True}, status=400)
            return
        except ValueError as error:
            self._send_json({"error": str(error)}, status=400)
            return
        except OllamaError as error:
            self._send_json({"error": str(error)}, status=502)
            return
        busy = work.start_picture()
        if busy:
            self._send_json({"error": busy, "busy": True}, status=409)
            return
        try:
            self._stream_start()
            self._stream_line({"lightusagi": "loading", "model": model, "unloading": router.large
                               if router.large and router.large != model else None})
            if spec is not None or model.startswith(imagegen.REMOTE_PREFIX):
                self._picture_from_server(request, spec, program, pictures["address"])
            else:
                with KEEPER.using(model):
                    self._picture_from_ollama(request)
        finally:
            work.end_picture()

    def _picture_from_server(self, request: dict, spec: dict | None, program: str | None, address: str) -> None:
        model = request["model"]
        try:
            # The chat model leaves memory before sd-server loads its own.
            router.begin(model)
            base = managed.ensure(program, spec) if spec is not None else address
            router.commit(model)
            if not self._stream_line({"progress": 0, "total": 0}):
                return
            image = sdserver.txt2img(
                base, request, lambda step, total: self._stream_line({"progress": step, "total": total}),
                lambda: getattr(self, "_gone", False))
            if not image or not self._stream_line({"progress": -1, "total": 0}):
                # Stopped: sd-server is stopped outright, which also frees its memory.
                if spec is not None:
                    managed.stop()
                else:
                    sdserver.interrupt(base)
                return
            record = imagegen.save(imagegen.decode(image), request)
            self._stream_line({"image": record})
        except sdserver.PictureError as error:
            self._stream_line({"error": str(error)})
        except ValueError as error:
            self._stream_line({"error": str(error)})
        except OSError as error:
            self._stream_line({"error": f"The picture could not be saved: {error.strerror or error}"})
        finally:
            if spec is not None:
                managed.touch()

    def _picture_from_ollama(self, request: dict) -> None:
        model = request["model"]
        try:
            router.begin(model)
            response = open_generate(imagegen.ollama_payload(request, KEEP_ALIVE))
        except OllamaError as error:
            self._stream_line({"error": str(error)})
            return
        router.commit(model)
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
                if event.get("image"):
                    if getattr(self, "_gone", False) or not self._stream_line({"progress": -1, "total": 0}):
                        return
                    record = imagegen.save(imagegen.decode(event["image"]), request)
                    self._stream_line({"image": record})
                    return
                if event.get("done"):
                    self._stream_line({"error": "Ollama finished without sending a picture. "
                                                "This Ollama may not make pictures with " + model + "."})
                    return
                if not self._stream_line({"progress": int(event.get("completed") or 0),
                                          "total": int(event.get("total") or 0)}):
                    return
            self._stream_line({"error": "Ollama stopped before the picture was finished."})
        except ValueError as error:
            self._stream_line({"error": str(error)})
        except OSError as error:
            self._stream_line({"error": f"The picture could not be saved: {error.strerror or error}"})
        finally:
            response.close()

    def _loaded(self) -> dict:
        """What is really in memory, asked of Ollama and sd-server rather than remembered,
        since models leave on their own when the keepalive runs out."""
        large = router.large or ""
        if large.startswith(imagegen.SD_PREFIX):
            running_sd = managed.running()
            return {"large_model": large if running_sd else None,
                    "unloads_at": managed.stops_at if running_sd else None}
        try:
            loaded = [item for item in running() if item["name"] and not is_nomic(item["name"])]
        except OllamaError:
            return {"large_model": None, "unloads_at": None}
        if not loaded:
            return {"large_model": None, "unloads_at": None}
        ours = keepalive.key(large) if large else None
        pick = next((item for item in loaded if keepalive.key(item["name"]) == ours), loaded[0])
        return {"large_model": pick["name"], "unloads_at": pick["expires"], "also_loaded": len(loaded) - 1}

    def _picture_setup(self) -> dict:
        """What Settings shows under Pictures: what was found and what is wrong."""
        pictures = context.load_settings()["pictures"]
        models, problems = sdserver.parse_models(pictures["models"])
        program = sdserver.find_program(pictures["program"])
        missing = [f"{model['name']}: no model file found" for model in models if not sdserver.model_files(model["args"])]
        server = None
        if pictures["address"]:
            try:
                server = {"model": sdserver.current_model(pictures["address"])}
            except sdserver.PictureError as error:
                server = {"error": str(error)}
        return {"program": program, "models": [model["name"] for model in models], "problems": problems + missing,
                "server": server, "ollama": imagegen.support()}

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

    def _send_bytes(self, body: bytes, content_type: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "private, max-age=31536000, immutable")
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


def _stop_on_signal(*_args) -> None:
    raise KeyboardInterrupt


def main() -> None:
    # A helper that is told to stop must not leave sd-server holding its model in memory.
    signal.signal(signal.SIGTERM, _stop_on_signal)
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    stopping = threading.Event()
    KEEPER.watch(running, unload, stopping, on_unload=router.forget)
    indexer.everything()
    print(f"LightUsagi at http://{HOST}:{PORT}", flush=True)
    print(f"Ollama at {os.environ.get('OLLAMA_HOST', 'http://127.0.0.1:11434')}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping", flush=True)
    finally:
        stopping.set()
        managed.stop()
        server.server_close()
