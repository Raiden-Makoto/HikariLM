# LightUsagi

A local chat page for Ollama. One model stays loaded at a time, and chats, files, and image notes live on the SSD, so that memory stays available for the model and its KV cache.

## Run it

You need [Ollama](https://ollama.com) running and Python 3.9 or newer. There is nothing to install.

| System | Start it |
|---|---|
| Windows | double-click `start.bat` |
| macOS, Linux | `./start.sh` |
| Anywhere | `python3 start.py` |

It checks Ollama, lists the models it finds, and opens the page at http://127.0.0.1:8787. Press Ctrl+C to stop. Starting it a second time just opens the page that is already running.

Options: `--port 9000` to use another port, `--no-browser` to skip opening the page.

To make pictures in Chat, download `sd-server` from [stable-diffusion.cpp's releases](https://github.com/leejet/stable-diffusion.cpp/releases) for your system and a model for it, then add them in Settings under *Making pictures*. LightUsagi runs it only while a picture is being made.

Your chats and drafts are saved in `~/.lightusagi` (set `LIGHTUSAGI_HOME` to move it).

How the pieces fit, and the order we will build them, is in [PLAN.md](PLAN.md).
