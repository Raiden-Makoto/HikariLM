"""Turning a request for a picture into words a picture model can draw.

FLUX.2 klein draws what a prompt describes, not what it names: "Usagi from Chiikawa"
gets a generic rabbit. A chat model that knows the name writes down how it looks
instead. With a reference picture, a vision model's description of it goes along.
"""

from __future__ import annotations

import re

from helper import thinking

MAX_WORDS = 120
MAX_TOKENS = 400

INSTRUCTION = (
    "You write prompts for a picture model. It only understands what things look like: it does not "
    "know the names of characters, people, shows, games, brands or places. Rewrite the request below "
    "as a description of the picture. Replace every name with how it looks: body shape and proportions, "
    "colors, face, hair, clothing, markings, and the art style it is usually drawn in. Keep everything "
    "the request asks for: the subject, what it is doing, the setting, mood, style and framing. Do not "
    "add people, objects or text it did not ask for. If you do not know what a name looks like, keep the "
    "name and describe the rest; never make up a look. "
    f"Write one paragraph of plain description, at most {MAX_WORDS} words. No title, no quotes, no list, "
    "no explanation, nothing before or after the prompt."
)


def messages(request: str, reference: str = "") -> list[dict]:
    text = INSTRUCTION + "\n\nRequest: " + request.strip()
    if reference.strip():
        text += ("\n\nA reference picture is attached. Use it for how the subject looks, over what you "
                 "remember. This is what it shows:\n" + reference.strip())
    return [{"role": "user", "content": text}]


LEAD = re.compile(r"^\s*(?:\*\*)?(?:here(?:'s| is) (?:the |a |your )?(?:rewritten )?prompt|rewritten prompt|prompt)"
                  r"\s*(?:\*\*)?\s*:\s*(?:\*\*)?\s*", re.IGNORECASE)
QUOTES = "\"'“”‘’`"


def clean(text: str) -> str:
    """The prompt alone, without thinking, a "Prompt:" lead, quotes or a code fence around it."""
    text = thinking.without_thinking(text).strip()
    text = re.sub(r"^```\w*\s*|\s*```$", "", text).strip()
    text = LEAD.sub("", text, count=1).strip()
    if len(text) > 1 and text[0] in QUOTES and text[-1] in QUOTES:
        text = text[1:-1].strip()
    return re.sub(r"\s+", " ", text)
