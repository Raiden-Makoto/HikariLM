<img src="web/icon.svg" alt="" width="96">

# HikariLM

A light, local chat page with a code editor beside it. Models run inside HikariLM with Apple's [MLX](https://github.com/ml-explore/mlx). It keeps one large model in memory at a time and puts everything else (chats, files, pictures) on disk, so your RAM goes to the model.

No Docker, no Ollama, no account. Your chats never leave your computer.

## Run it

HikariLM runs only on a Mac with Apple Silicon (M1 or newer). You need Python 3.9 or newer.

```bash
./start.sh            # or: python3 start.py
```

The first start makes `~/.hikarilm/venv` and installs `mlx-lm` and `mlx-vlm` into it (about 1 GB, once). Then it lists your models and opens http://127.0.0.1:8787. Press Ctrl+C to stop. Running it again while it is running just opens the page.

Options: `--port 9000` for another port, `--no-browser` to skip opening the page, `--version`.

## Models

Download them from Hugging Face with the `hf` tool the venv has. These three are what each part is tuned for:

| For | Model | Size |
|---|---|---|
| Questions about code | `mlx-community/Qwen2.5-Coder-7B-Instruct-4bit` | 4.3 GB |
| General chat, with thinking | `mlx-community/DeepSeek-R1-0528-Qwen3-8B-4bit` | 4.6 GB |
| Pictures you attach | `mlx-community/gemma-3-12b-it-4bit` | 8 GB |

```bash
~/.hikarilm/venv/bin/hf download mlx-community/Qwen2.5-Coder-7B-Instruct-4bit
~/.hikarilm/venv/bin/hf download mlx-community/DeepSeek-R1-0528-Qwen3-8B-4bit
~/.hikarilm/venv/bin/hf download mlx-community/gemma-3-12b-it-4bit
```

Any other MLX model in the Hugging Face cache, or in a folder of its own in `~/.hikarilm/models`, is listed too. Reload the page after a download.

## What it does

- **Chat** with any of your models. Replies are drawn as Markdown, with tables, code, and math. A model's thinking is folded into one line, and Settings can cap how long it thinks.
- **Attach files**: text and code, PDFs, Word and Excel files, and pictures. A picture goes to a model that can read it, such as Gemma 3. Long documents are read in parts so they fit the model's context.
- **Make pictures** with your own picture models. Pick *Set up picture making…* in the model list for a one-click start with FLUX.2 klein: it downloads `sd-server` and the model once (about 5 GB) and runs them only while a picture is being made. In Settings you can add any other model `sd-server` runs, or point HikariLM at a picture server you already run (AUTOMATIC1111, Forge, Draw Things).
- **Code editor**: open a folder, edit a file, and ask about it in the chat beside it. It has syntax colours, suggestions as you type (Tab takes one), and a Markdown preview. Apply works like Cursor's. The model labels code meant for a file with that file's path, and Apply puts it into that file, or creates the file if it is new. A block can leave out unchanged code with `... existing code ...` comments. If it can't be placed exactly, the chosen model writes the file with the change in it. Each change then shows in red and green to accept or reject. Commands, output and examples get no Apply button. Nothing is written until you press Save.
- **Long chats** are summarized when they get close to the context length, the way Open WebUI does it. The whole chat stays saved. Settings sets the context length, and how the model keeps what it has read (the KV cache: 8-bit by default, which holds about twice as much).
- **Organize chats**: pin them, put them in folders, rename them. New chats are named from their first message.

## Keys

| Keys | Does |
|---|---|
| ⌘B | Hide or show the chat list, or the file list in the code editor |
| ⌘K | Code editor: add the open file to your message |
| ⇧⌘K | Code editor: add the highlighted lines, or the cursor's line (also right-click, Add code to chat) |
| ⌃⌥N | New chat (change it in Settings) |
| ⌘S | Code editor: save |
| ⌘⏎ | Code editor: accept all changes from Apply |
| ⌘⌫ | Code editor: reject all changes from Apply |

## Where things are saved

Everything is in `~/.hikarilm`: chats, drafts, settings, pictures you made, and the MLX venv. The models are in the Hugging Face cache (`~/.cache/huggingface/hub`). Your project folders stay where they are.

## Settings from the environment

| Variable | Default | Does |
|---|---|---|
| `HIKARILM_HOME` | `~/.hikarilm` | Where HikariLM saves things |
| `HIKARILM_PORT`, `HIKARILM_HOST` | `8787`, `127.0.0.1` | Where the page is served |
| `HIKARILM_KEEP_ALIVE` | `5m` | How long a model stays loaded after its last reply |
| `HF_HUB_CACHE`, `HF_HOME` | Hugging Face's | Where the downloaded models are |

## Development

The helper is Python with the standard library only; the models run in a separate worker process with the venv's Python. The page is plain HTML, CSS, and JavaScript modules, so there is no build step. Run the tests from this folder with `python3 -m unittest`. The code editor is a CodeMirror bundle in `web/vendor/`; `tools/editor/build.sh` rebuilds it (only needed to change it).

## License

Apache 2.0. The libraries in `web/vendor/` keep their own licenses, next to them.
