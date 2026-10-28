# LightUsagi

A local chat page for Ollama. One model stays loaded at a time, and chats, files, and image notes live on the SSD, so that memory stays available for the model and its KV cache.

## Run it

LightUsagi runs on macOS and Linux. Windows is not supported. You need [Ollama](https://ollama.com) running and Python 3.9 or newer. There is nothing to install.

Start it with `./start.sh`, or `python3 start.py`.

It checks Ollama, lists the models it finds, and opens the page at http://127.0.0.1:8787. Press Ctrl+C to stop. Starting it a second time just opens the page that is already running.

Options: `--port 9000` to use another port, `--no-browser` to skip opening the page.

To make pictures in Chat, pick *Set up picture making…* in the model list (or the button in Settings) and press *Download and set up*. That downloads once, about 5 GB, into `~/.lightusagi`: `sd-server` from [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp/releases) and the FLUX.2 klein picture model. Then *flux2-klein* shows up in the model list. LightUsagi runs sd-server only while a picture is being made.

If LM Studio's server (llmster) is already running, its models show up in the picker under *LM Studio*. LightUsagi does not start it. If it is not at `http://127.0.0.1:1234`, set `LMSTUDIO_HOST`. If it needs a key, set `LM_API_TOKEN`.

Keys: Cmd+B (Ctrl+B on Linux) hides the side list. In the code editor, Cmd+K adds the open file to your message and Cmd+L adds the highlighted lines. Ctrl+Option+N (Ctrl+Alt+N) starts a new chat; you can change that one in Settings.

Your chats and drafts are saved in `~/.lightusagi` (set `LIGHTUSAGI_HOME` to move it).

The tests are in `tests/`. Run them from this folder with `python3 -m unittest`.

How the pieces fit, and the order we will build them, is in [PLAN.md](PLAN.md).
