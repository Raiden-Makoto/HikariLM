"""Turning a request for a picture into words a picture model can draw.

FLUX.2 klein draws what a prompt describes, not what it names: "Usagi from Chiikawa"
gets a generic rabbit. A chat model that knows the name writes down how it looks
instead. With a reference picture, a vision model's description of it goes along.

A LoRA trained on booru tags (1girl, blue_hair, school_uniform) draws best from tags. For
those, a model that turns sentences into tags, such as BooruNL, is asked in the form it was
trained on ("Convert this to danbooru tags: ..."), and its comma-separated list is kept.
"""

from __future__ import annotations

import re

from helper import thinking

MAX_WORDS = 120
MAX_TOKENS = 400

START = (
    "You write prompts for a picture model. It only understands what things look like: it does not "
    "know the names of characters, people, shows, games, brands or places. Rewrite the request below "
    "as a description of the picture. "
)
NAMES = (
    "Replace every name with how it looks: body shape and proportions, colors, face, hair, clothing, "
    "markings, and the art style it is usually drawn in. If you do not know what a name looks like, keep "
    "the name and describe the rest; never make up a look. "
)
# The picture model is given the reference picture too and copies the subject from it,
# but it also copies the rest of the picture, so anything new is easily left out.
SHOWN = (
    "The picture model is also given a reference picture and copies the subject's look from it. It "
    "tends to copy the rest of that picture too, so anything new is easily left out. Begin with one "
    "sentence that says the subject in a few words (for example \"the white rabbit-like character from "
    "the reference picture\"), what it is doing and every object the request names. Then the setting, "
    "mood, style and framing. Do not describe the subject's look, and do not carry over the reference "
    "picture's pose, background or objects unless the request asks for them. "
)
END = (
    "Keep everything the request asks for: the subject, what it is doing, the setting, mood, style and "
    "framing. Every object, food, prop and action the request names must be in the prompt, named plainly "
    "and shown being used (for \"eating a burger\": a cheeseburger with a sesame bun held in both hands, "
    "mid-bite). Do not add people, objects or text it did not ask for. "
    f"Write one paragraph of plain description, at most {MAX_WORDS} words. No title, no quotes, no list, "
    "no explanation, nothing before or after the prompt."
)


def messages(request: str, reference: str = "", shown: bool = False) -> list[dict]:
    """`shown`: the picture model is given the reference picture too, so it copies the
    subject from the picture and the prompt only needs to say what to do with it."""
    text = START + (SHOWN if shown else NAMES) + END + "\n\nRequest: " + request.strip()
    if shown:
        if reference.strip():
            text += "\n\nThis is what the reference shows, for you to know the subject:\n" + reference.strip()
    elif reference.strip():
        text += ("\n\nA reference picture is attached. Use it for how the subject looks, over what you "
                 "remember. This is what it shows:\n" + reference.strip())
    return [{"role": "user", "content": text}]


TAGS = "Convert this to danbooru tags: "


def tag_messages(request: str, reference: str = "") -> list[dict]:
    """BooruNL takes a description and gives tags; a reference picture's description adds to it."""
    text = request.strip()
    if reference.strip():
        text += ("" if text.endswith((".", "!", "?")) else ".") + " " + reference.strip()
    return [{"role": "user", "content": TAGS + text}]


def clean_tags(text: str) -> str:
    """The tags alone, comma-separated, each once, in the order written: no thinking, lead,
    code fence or list marks around them."""
    text = thinking.without_thinking(text).strip()
    text = re.sub(r"^```\w*\s*|\s*```$", "", text).strip()
    text = re.sub(r"^\s*(?:here are (?:the )?|danbooru )?tags\s*:\s*", "", text, flags=re.IGNORECASE)
    seen, out = set(), []
    for tag in re.split(r",|\n", text):
        tag = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s+", "", tag).strip().strip("\"'`")
        if tag and tag.lower() not in seen:
            seen.add(tag.lower())
            out.append(tag)
    return ", ".join(out)


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
