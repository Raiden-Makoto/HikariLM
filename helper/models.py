"""One way to reach a model. The rest of the helper calls these and never asks
which server it is talking to.
"""

from __future__ import annotations

from helper.ollama import complete, open_chat, running, show, sizes, tags, unload

__all__ = ["complete", "open_chat", "running", "show", "sizes", "tags", "unload"]
