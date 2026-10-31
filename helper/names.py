"""Short model names for people, the same as the page's modelLabel in web/js/models.js.

"mlx-community/DeepSeek-R1-0528-Qwen3-8B-4bit" is "DeepSeek R1". Earlier
entries win, so a distill is named for its teacher. tests/test_web.py checks
that this table and the page's are the same.
"""

from __future__ import annotations

import re

FAMILIES = [
    (r"deepseek[-_ ]?r1", "DeepSeek R1"), (r"deepseek[-_ ]?v3", "DeepSeek V3"),
    (r"qwen2\.5[-_ ]?coder", "Qwen2.5 Coder"), (r"qwen3[-_ ]?coder", "Qwen3 Coder"),
    (r"qwen2\.5[-_ ]?vl", "Qwen2.5 VL"), (r"qwen3", "Qwen3"), (r"qwen2\.5", "Qwen2.5"),
    (r"gemma[-_ ]?3n", "Gemma 3n"), (r"gemma[-_ ]?3", "Gemma 3"), (r"gemma[-_ ]?2", "Gemma 2"),
    (r"llama[-_ ]?3\.3", "Llama 3.3"), (r"llama[-_ ]?3\.2", "Llama 3.2"), (r"llama[-_ ]?3\.1", "Llama 3.1"),
    (r"gpt-oss", "gpt-oss"), (r"devstral", "Devstral"), (r"mistral", "Mistral"), (r"phi[-_ ]?4", "Phi-4"),
    (r"falcon[-_ ]?3", "Falcon3"), (r"nomic-embed", "nomic-embed-text"),
]
NOISE = r"^(\d+(\.\d+)?[bm]|\d+bit|\d{4}|instruct|it|chat|mlx|gguf|latest|q\d\w*|fp16|bf16)$"


def label(name: str) -> str:
    base = str(name or "").split("/")[-1]
    for pattern, short in FAMILIES:
        if re.search(pattern, base, re.I):
            return short
    words = [word for word in re.split(r"[-_:]", base) if word and not re.match(NOISE, word, re.I)]
    return " ".join(words) or base
