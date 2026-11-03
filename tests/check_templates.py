"""Render HikariLM's chat prompts through each suggested model's own chat template. Not part of the test suite.

Fetches only each repo's template (tokenizer_config.json, chat_template.jinja or .json), not the
model, and renders it with jinja2 the way transformers does, for three prompts: a question with
the Python tool offered (and your rules in the system message), the chat after a call and its
result, and the last answer once the tool is withdrawn. It says, for each model, whether the
template takes tools, whether HikariLM gives them to it as they are or as text, whether any
prompt fails to render, and whether the result reaches the model. Runs anywhere with internet
and jinja2; no MLX.

    python3 tests/check_templates.py            the suggested models
    python3 tests/check_templates.py -v         and the prompt after a call
    python3 tests/check_templates.py mlx-community/Qwen3-8B-4bit
"""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jinja2.sandbox import ImmutableSandboxedEnvironment  # noqa: E402

from helper import mlx_worker, tools  # noqa: E402

MODELS = [
    "mlx-community/Qwen3-8B-4bit",
    "mlx-community/DeepSeek-R1-0528-Qwen3-8B-4bit",
    "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit",
    "mlx-community/gemma-3-12b-it-4bit",
    "tiiuae/Falcon3-10B-Instruct-1.58bit",
    "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
    "mlx-community/gpt-oss-20b-MXFP4-Q8",
]
# Models that read pictures run in mlx-vlm, whose templates are never given tools.
VISION = ("gemma-3", "qwen2.5-vl", "llava", "pixtral")


def fetch(repo: str, name: str) -> str | None:
    out = subprocess.run(["curl", "-sfL", f"https://huggingface.co/{repo}/resolve/main/{name}"],
                         capture_output=True, text=True)
    return out.stdout if out.returncode == 0 else None


def template_of(repo: str) -> tuple[str | None, dict]:
    config = json.loads(fetch(repo, "tokenizer_config.json") or "{}")
    found = config.get("chat_template")
    if isinstance(found, list):
        found = next((t["template"] for t in found if t.get("name") == "default"), found[0]["template"])
    for name in ("chat_template.jinja", "chat_template.json"):
        if found:
            break
        text = fetch(repo, name)
        if text:
            found = json.loads(text).get("chat_template") if name.endswith(".json") else text

    def token(key):
        value = config.get(key)
        return value.get("content") if isinstance(value, dict) else value

    return found, {"bos_token": token("bos_token") or "", "eos_token": token("eos_token") or ""}


def render(template: str, specials: dict, chat: list[dict], offered) -> str:
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    env.filters["tojson"] = lambda x, indent=None, ensure_ascii=False, separators=None, sort_keys=False: json.dumps(
        x, indent=indent, ensure_ascii=ensure_ascii, separators=separators, sort_keys=sort_keys)

    def raise_exception(message):
        raise ValueError(message)

    env.globals.update(raise_exception=raise_exception, strftime_now=lambda f: datetime.now().strftime(f))
    extra = {"tools": offered} if offered else {}
    return env.from_string(template).render(messages=chat, add_generation_prompt=True, **specials, **extra)


def main() -> int:
    verbose = "-v" in sys.argv
    repos = [arg for arg in sys.argv[1:] if arg != "-v"] or MODELS
    answer = {"name": "python", "code": "import math\nprint(math.isqrt(4489))", "output": "67\n", "pictures": []}
    first = [{"role": "system", "content": "The user's rules. Follow them in every answer:\n\nBe brief."},
             {"role": "user", "content": "What is the square root of 4489?"}]
    prompts = {"asked, Python offered": (first, True),
               "after the call and its result": (first + tools.as_messages("", {}, answer, 1), True),
               "last answer, tool withdrawn": (tools.answer_from(first, [answer]), False)}
    failed = 0
    for repo in repos:
        template, specials = template_of(repo)
        print(f"\n{repo}")
        if not template:
            print("  no chat template found")
            failed += 1
            continue
        native = mlx_worker.native_tools(template) and not any(word in repo.lower() for word in VISION)
        print(f"  template takes tools: {'tools' in template}; the tool goes in "
              f"{'as its own tools' if native else 'as text in the system message'}")
        for label, (messages, offer) in prompts.items():
            chat = mlx_worker.tool_chat(messages, tools.TOOLS if offer else None, tools.AS_TEXT, native)
            try:
                text = mlx_worker.with_system(
                    lambda turns: render(template, specials, turns, tools.TOOLS if offer and native else None), chat)
            except Exception as error:  # noqa: BLE001 - reported; the next prompt still runs
                print(f"  {label}: FAILS: {error}")
                failed += 1
                continue
            reached = "" if label.startswith("asked") else (", the result reaches it" if "67" in text
                                                            else ", the RESULT IS MISSING")
            failed += "MISSING" in reached
            print(f"  {label}: renders{reached}")
            if verbose and label.startswith("after"):
                print("    ..." + text[-700:].replace("\n", "\n    "))
    print("\nAll render." if not failed else f"\n{failed} problem(s).")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
