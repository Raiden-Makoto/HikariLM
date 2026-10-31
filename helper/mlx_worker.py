"""Run one MLX model and answer chats with it, one at a time.

The helper starts this with the Python in ~/.hikarilm/venv, where mlx-lm and
mlx-vlm are installed; the helper itself never imports MLX. It loads the model
given on the command line once, then reads one JSON command per line on stdin
and writes JSON lines on stdout:

    {"op": "chat", "messages": [...], "num_ctx": 8192, "num_predict": 512, "think": true}
        -> {"text": "..."} as it is written, then {"done": true, ...counts}
    {"op": "stop"}   ends the reply being written; its "done" line still comes.

Started with --embed, it runs nomic-embed-text (nomic_mlx.py) instead:

    {"op": "embed", "texts": ["search_document: ...", ...]}  -> {"vectors": [[...], ...]}

A failure is {"error": "..."}. To free the model the helper ends this process,
which hands all of its memory back at once.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import queue
import sys
import tempfile
import threading
import time

THINK_OPEN = "<think>"
EMPTY_THINK = "<think>\n\n</think>\n\n"
# Where a turn ends, in the text of the models we run. A model whose config names a different
# end token (Qwen2.5 says <|endoftext|>) would otherwise write these and keep going.
TURN_ENDS = ("<|im_end|>", "<|endoftext|>", "<|im_start|>", "<end_of_turn>", "<start_of_turn>",
             "<｜end▁of▁sentence｜>", "<｜User｜>", "<|eot_id|>")


class TurnEnd:
    """Cuts a reply at the first turn marker in its text. A marker can arrive across
    pieces, so text that could be the start of one is held back until the next piece."""

    def __init__(self) -> None:
        self.buffer = ""
        self.ended = False

    def feed(self, text: str) -> str:
        self.buffer += text
        found = [at for at in (self.buffer.find(marker) for marker in TURN_ENDS) if at >= 0]
        if found:
            out, self.buffer, self.ended = self.buffer[:min(found)], "", True
            return out
        keep = max((n for marker in TURN_ENDS for n in range(1, min(len(marker), len(self.buffer)) + 1)
                    if self.buffer.endswith(marker[:n])), default=0)
        out, self.buffer = self.buffer[:len(self.buffer) - keep], self.buffer[len(self.buffer) - keep:]
        return out

    def flush(self) -> str:
        out, self.buffer = self.buffer, ""
        return out


def _stop_at_turn_ends(tokenizer) -> None:
    """Add the tokenizer's own end token, and the turn markers it has, to what ends a reply."""
    add = getattr(tokenizer, "add_eos_token", None)
    vocab = getattr(tokenizer, "get_vocab", lambda: {})() or {}
    wanted = [getattr(tokenizer, "eos_token", None)] + [marker for marker in TURN_ENDS if marker in vocab]
    for token in wanted:
        if not token:
            continue
        try:
            if add:
                add(token)
            elif isinstance(getattr(tokenizer, "_eos_token_ids", None), set) and token in vocab:
                tokenizer._eos_token_ids.add(vocab[token])
        except (ValueError, KeyError, AttributeError):
            pass

_out_lock = threading.Lock()


def send(event: dict) -> None:
    with _out_lock:
        sys.stdout.write(json.dumps(event) + "\n")
        sys.stdout.flush()


def _clear_cache(mx) -> None:
    clear = getattr(mx, "clear_cache", None) or getattr(getattr(mx, "metal", None), "clear_cache", None)
    if clear:
        clear()


def _sampling(path: str) -> dict:
    """The makers' own sampling settings from generation_config.json, else a gentle default."""
    out = {"temperature": 0.7, "top_p": 0.9, "top_k": 0}
    try:
        with open(os.path.join(path, "generation_config.json"), encoding="utf-8") as handle:
            config = json.load(handle)
    except (OSError, ValueError):
        return out
    for key in out:
        if isinstance(config.get(key), (int, float)):
            out[key] = config[key]
    if config.get("do_sample") is False:
        out["temperature"] = 0.0
    return out


def _is_bitnet(path: str) -> bool:
    try:
        with open(os.path.join(path, "config.json"), encoding="utf-8") as handle:
            quantization = json.load(handle).get("quantization_config")
    except (OSError, ValueError, AttributeError):
        return False
    return isinstance(quantization, dict) and quantization.get("quant_method") == "bitnet" \
        and quantization.get("linear_class", "bitlinear") != "autobitlinear"


def round_bitnet_activations() -> None:
    """Make mlx-lm's BitLinear compute what transformers' BitLinear does, the code Falcon3
    1.58-bit was trained and published with: each token's inputs are rounded to 8-bit levels
    (127 / its largest value), the packed ternary weights multiply those whole numbers, and
    the result is divided by both scales. mlx-lm's kernel alone multiplies the inputs unrounded."""
    import mlx.core as mx
    from mlx_lm.models.bitlinear_layers import BitLinear

    if getattr(BitLinear, "_rounds_inputs", False):
        return

    def __call__(self, x):
        dtype = self.weight_scale.dtype
        wide = x.astype(mx.float32)
        scale = 127.0 / mx.maximum(mx.abs(wide).max(axis=-1, keepdims=True), 1e-5)
        whole = mx.clip(mx.round(wide * scale), -128, 127).astype(dtype)
        y = (self.execute_matmul_kernel(whole, self.weight).astype(mx.float32) / scale).astype(dtype)
        return y if self.bias is None else y + self.bias

    BitLinear.__call__ = __call__
    BitLinear._rounds_inputs = True


def _text(content) -> str:
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return str(content or "")


class TextModel:
    """mlx-lm. The last chat's prompt cache is kept, so a follow-up only reads what is new."""

    def __init__(self, path: str) -> None:
        import mlx.core as mx
        from mlx_lm import load, stream_generate
        from mlx_lm.models import cache as cache_module
        from mlx_lm.sample_utils import make_sampler

        self.mx, self.stream_generate, self.caches = mx, stream_generate, cache_module
        if _is_bitnet(path):
            round_bitnet_activations()
        self.model, self.tokenizer = load(path)
        _stop_at_turn_ends(self.tokenizer)
        sampling = _sampling(path)
        self.sampler = make_sampler(temp=sampling["temperature"], top_p=sampling["top_p"], top_k=sampling["top_k"])
        template = str(getattr(self.tokenizer, "chat_template", "") or "")
        self.template_switch = "enable_thinking" in template
        self.cache = None
        self.cached: list[int] = []
        self.kv_bits = 0

    def _prompt(self, messages: list[dict], think) -> tuple[list[int], str]:
        """Token ids to send, and text the model is taken to have written already."""
        chat = [{"role": m["role"], "content": _text(m.get("content"))} for m in messages]
        extra = {"enable_thinking": bool(think)} if self.template_switch and think is not None else {}
        text = self.tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False, **extra)
        lead = ""
        if think is False and not self.template_switch:
            # An R1 distill always reasons; an empty reasoning block makes it answer at once.
            text = text[:-len("<think>\n")] if text.endswith("<think>\n") else text
            text += EMPTY_THINK
        elif text.rstrip().endswith(THINK_OPEN):
            # The template opens the reasoning itself; hand the tag on so it can be split off.
            lead = THINK_OPEN + "\n"
        elif think and not self.template_switch:
            # R1 distills often start reasoning without writing <think>, so the helper could not
            # tell it from the answer (or cut it at the limit). DeepSeek's advice: open it for them.
            text += THINK_OPEN + "\n"
            lead = THINK_OPEN + "\n"
        return self.tokenizer.encode(text, add_special_tokens=False), lead

    def _reuse(self, ids: list[int]) -> list[int]:
        """Trim the kept cache to what this prompt shares with it; returns the ids still to read."""
        caches = self.caches
        same = 0
        for old, new in zip(self.cached, ids):
            if old != new:
                break
            same += 1
        same = min(same, len(ids) - 1)
        if self.cache is not None and same > 0 and caches.can_trim_prompt_cache(self.cache):
            caches.trim_prompt_cache(self.cache, len(self.cached) - same)
            self.cached = ids[:same]
            return ids[same:]
        self.cache = caches.make_prompt_cache(self.model)
        self.cached = []
        return ids

    def _quantizable(self) -> bool:
        # Sliding-window caches (RotatingKVCache) cannot be quantized by mlx-lm.
        return all(type(entry).__name__ in ("KVCache", "QuantizedKVCache") for entry in self.cache)

    def chat(self, request: dict, stop: threading.Event):
        ids, lead = self._prompt(request["messages"], request.get("think"))
        if lead:
            yield {"text": lead}
        bits = request.get("kv_bits") if request.get("kv_bits") in (4, 8) else 0
        if bits != self.kv_bits:
            # A cache kept at one setting cannot be read at another, so it is started again.
            self.cache, self.cached, self.kv_bits = None, [], bits
        fresh = self._reuse(ids)
        made: list[int] = []
        last = None
        quantize = {"kv_bits": bits, "kv_group_size": 64, "quantized_kv_start": 0} if bits and self._quantizable() else {}
        pieces = self.stream_generate(self.model, self.tokenizer, fresh, max_tokens=_room(request, len(ids)),
                                      sampler=self.sampler, prompt_cache=self.cache, **quantize)
        turn = TurnEnd()
        try:
            for last in pieces:
                made.append(last.token)
                text = turn.feed(last.text)
                if text:
                    yield {"text": text}
                if stop.is_set() or turn.ended:
                    break
        finally:
            pieces.close()
        rest = turn.flush()
        if rest:
            yield {"text": rest}
        offset = getattr(self.cache[0], "offset", None) if self.cache else None
        if isinstance(offset, int):
            self.cached = (self.cached + fresh + made)[:offset]
        else:
            self.cache, self.cached = None, []
        _clear_cache(self.mx)
        yield _counts(last, len(ids), stop.is_set())


class VisionModel:
    """mlx-vlm, for models that read pictures. Pictures come as base64, as the page sends them."""

    def __init__(self, path: str) -> None:
        import mlx.core as mx
        from mlx_vlm import load, stream_generate
        from mlx_vlm.prompt_utils import apply_chat_template

        self.mx, self.stream_generate, self.template = mx, stream_generate, apply_chat_template
        self.model, self.processor = load(path)
        _stop_at_turn_ends(getattr(self.processor, "tokenizer", self.processor))
        self.sampling = _sampling(path)

    def chat(self, request: dict, stop: threading.Event):
        messages = request["messages"]
        pictures = [image for m in messages for image in m.get("images") or []]
        chat = [{"role": m["role"], "content": _text(m.get("content"))} for m in messages]
        prompt = self.template(self.processor, self.model.config, chat, num_images=len(pictures))
        with tempfile.TemporaryDirectory() as folder:
            paths = []
            for number, image in enumerate(pictures):
                path = os.path.join(folder, f"{number}.img")
                with open(path, "wb") as handle:
                    handle.write(base64.b64decode(image.split(",", 1)[-1]))
                paths.append(path)
            last = None
            guess = sum(len(m["content"]) for m in chat) // 3 + 300 * len(paths)
            pieces = self.stream_generate(self.model, self.processor, prompt, image=paths or None,
                                          max_tokens=_room(request, guess), temperature=self.sampling["temperature"])
            turn = TurnEnd()
            try:
                for last in pieces:
                    text = turn.feed(last.text)
                    if text:
                        yield {"text": text}
                    if stop.is_set() or turn.ended:
                        break
            finally:
                pieces.close()
            rest = turn.flush()
            if rest:
                yield {"text": rest}
        _clear_cache(self.mx)
        yield _counts(last, 0, stop.is_set())


def _room(request: dict, prompt_tokens: int) -> int:
    """How long the reply may be: what is asked for, and never past the context length."""
    left = max(64, int(request.get("num_ctx") or 8192) - prompt_tokens)
    wanted = request.get("num_predict")
    return min(left, int(wanted)) if isinstance(wanted, int) and wanted > 0 else left


def _counts(last, prompt_tokens: int, stopped: bool) -> dict:
    return {"done": True, "stopped": stopped,
            "reason": getattr(last, "finish_reason", None) or ("stop" if stopped else "length"),
            # The whole prompt: with a kept cache, mlx-lm counts only the part it had to read.
            "prompt_tokens": int(prompt_tokens or getattr(last, "prompt_tokens", 0) or 0),
            "tokens": int(getattr(last, "generation_tokens", 0) or 0),
            "prompt_tps": float(getattr(last, "prompt_tps", 0) or 0),
            "tps": float(getattr(last, "generation_tps", 0) or 0),
            "peak": float(getattr(last, "peak_memory", 0) or 0)}


def _embed_loop(embedder) -> None:
    for line in sys.stdin:
        try:
            command = json.loads(line)
        except ValueError:
            continue
        texts = command.get("texts")
        if command.get("op") != "embed" or not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
            send({"error": "embed needs a list of texts"})
            continue
        try:
            send({"vectors": embedder.embed(texts)})
        except Exception as error:  # noqa: BLE001 - the helper shows it; the next batch still works
            send({"error": str(error) or type(error).__name__})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--vision", action="store_true")
    parser.add_argument("--embed", action="store_true")
    args = parser.parse_args()
    started = time.monotonic()
    try:
        if args.embed:
            from nomic_mlx import Embedder

            runner = Embedder(args.model)
        else:
            runner = (VisionModel if args.vision else TextModel)(args.model)
    except Exception as error:  # noqa: BLE001 - any load failure is reported, then the process ends
        send({"error": f"The model could not be loaded: {error}"})
        return
    send({"ready": True, "seconds": round(time.monotonic() - started, 1)})
    if args.embed:
        _embed_loop(runner)
        return

    commands: queue.Queue = queue.Queue()
    stop = threading.Event()

    def read() -> None:
        for line in sys.stdin:
            try:
                command = json.loads(line)
            except ValueError:
                continue
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
        try:
            for event in runner.chat(command, stop):
                send(event)
        except Exception as error:  # noqa: BLE001 - the helper shows it; the next chat still works
            send({"error": str(error) or type(error).__name__})


if __name__ == "__main__":
    main()
