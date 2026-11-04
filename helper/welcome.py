"""The welcome page's list (/welcome): the models to start with, and which are on this Mac.

The same list as Recommended models in PLAN.md.
"""

from __future__ import annotations

from helper import engine, imagegen, voice

GROUPS = ("Chat", "Pictures", "Voice", "Optional")

# No BitNet model is recommended over the others, so its entry names none.
BITNET = "<bitnet model>"

RECOMMENDED = [
    {"group": "Chat", "name": "Qwen3 8B", "repo": "mlx-community/Qwen3-8B-4bit", "download": "4.6 GB",
     "for": "Everyday chat, thinking or not",
     "note": "The one to start with. Follows Let models think in Settings: off, it answers straight away; "
             "on, it works things out first. Runs Python for math and plots."},
    {"group": "Chat", "name": "DeepSeek R1 8B", "repo": "mlx-community/DeepSeek-R1-0528-Qwen3-8B-4bit",
     "download": "4.6 GB", "for": "Dense reasoning, math and step-by-step analysis",
     "note": "Always thinks, at length, which is what it is for: hard math, proofs, logic and working a "
             "problem through step by step. Slow on simple questions; use Qwen3 for those."},
    {"group": "Chat", "name": "Qwen2.5 Coder 7B", "repo": "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit",
     "download": "4.3 GB", "for": "Code",
     "note": "For the code editor and code questions. Answers straight away, without thinking."},
    {"group": "Chat", "name": "Gemma 3 12B", "repo": "mlx-community/gemma-3-12b-it-4bit", "download": "8.1 GB",
     "for": "Reading pictures",
     "note": "Reads attached pictures, and chats too. Runs Python for math and plots."},
    {"group": "Pictures", "name": "FLUX.2 klein 4B", "repo": imagegen.SUGGESTED, "download": "4.6 GB",
     "for": "Drawing pictures",
     "note": "For Macs with less than 18 GB of RAM, and the quicker one."},
    {"group": "Pictures", "name": "FLUX.2 klein 9B", "repo": imagegen.LARGER, "download": "9.5 GB",
     "for": "Drawing, more detail", "note": "For Macs with 18 GB of RAM or more."},
    {"group": "Pictures", "name": "FLUX.2 klein base 4B", "repo": "mflux-community/flux-2-klein-base-4b-mflux-q4",
     "download": "4.6 GB", "for": "Drawing with LoRAs",
     "note": "Slower (50 steps); the one LoRAs draw on."},
    {"group": "Voice", "name": "Whisper large-v3 turbo", "repo": voice.WHISPER, "download": "1.6 GB",
     "for": "Speech to text",
     "note": "Press the microphone by Attach and speak; your words are written into the message box to read "
             "and fix before you send. Many languages. After downloading it, run  hikarilm restart  once."},
    {"group": "Voice", "name": "Whisper small", "repo": voice.SMALL_WHISPER, "download": "0.5 GB",
     "for": "Speech to text, smaller", "note": "Less memory and quicker, and a little less accurate."},
    {"group": "Voice", "name": "Kokoro 82M", "repo": voice.REPO, "download": "0.4 GB",
     "for": "Reading answers aloud",
     "note": "Press Listen under an answer to hear it, in one of Kokoro's English voices. Stays loaded beside "
             "the chat model. After downloading it, run  hikarilm restart  once to set it up."},
    {"group": "Optional", "name": "Any BitNet model", "repo": BITNET, "download": "varies", "for": "BitNet",
     "note": "1.58-bit weights: small for their size class. Any one HikariLM runs will do (the Falcon3 and "
             "Falcon-E 1.58-bit family, Microsoft's bitnet-b1.58-2B-4T); put its repo in the command. "
             "Chat only, no tools."},
]


def listing() -> list[dict]:
    """Each recommended model with its download command, and whether it is on this Mac."""
    engine.forget_scan()
    chat = set(engine.found())
    have = chat | {item["id"] for item in imagegen.picture_models()[0]}
    if any(engine.bitnet_model(name) for name in chat):
        have.add(BITNET)
    if voice.folder() is not None:
        have.add(voice.REPO)
    have |= {item["id"] for item in voice.whisper_models()}
    return [{**model, "command": imagegen.DOWNLOAD + model["repo"], "have": model["repo"] in have}
            for model in RECOMMENDED]
