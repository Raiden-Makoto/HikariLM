"""The welcome page's list (/welcome): the models to start with, and which are on this Mac.

The same list as Recommended models in PLAN.md. Voice models are left out until HikariLM uses them.
"""

from __future__ import annotations

from helper import engine, imagegen

GROUPS = ("Chat", "Pictures", "Optional")

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
    {"group": "Optional", "name": "Falcon3 10B 1.58-bit", "repo": "tiiuae/Falcon3-10B-Instruct-1.58bit",
     "download": "4.0 GB", "for": "BitNet",
     "note": "1.58-bit weights: small for its size class. Chat only, no tools."},
]


def listing() -> list[dict]:
    """Each recommended model with its download command, and whether it is on this Mac."""
    engine.forget_scan()
    have = set(engine.found()) | {item["id"] for item in imagegen.picture_models()[0]}
    return [{**model, "command": imagegen.DOWNLOAD + model["repo"], "have": model["repo"] in have}
            for model in RECOMMENDED]
