# HikariLM

A light, local chat page for [Ollama](https://ollama.com), with a code editor beside it. It keeps one large model in memory at a time and puts everything else (chats, files, notes) on disk, so your RAM goes to the model.

No Docker, no packages to install, no account. Your chats never leave your computer.

## Run it

HikariLM runs on macOS and Linux (Windows is not supported). You need Ollama running and Python 3.9 or newer.

```bash
./start.sh            # or: python3 start.py
```

It checks Ollama, lists your models, and opens http://127.0.0.1:8787. Press Ctrl+C to stop. Running it again while it is running just opens the page.

Options: `--port 9000` for another port, `--no-browser` to skip opening the page, `--version`.

## What it does

- **Chat** with any Ollama model. Replies are drawn as Markdown, with tables, code, and math. A model's thinking is folded into one line, and Settings can cap how long it thinks.
- **Attach files**: text and code, PDFs, Word and Excel files, and pictures. A picture goes to a model that can read it: any model Ollama lists with vision, such as `gemma3`, `llava`, or `qwen2.5vl`. Long documents are read in parts so they fit the model's context.
- **Make pictures** with your own picture models. Pick *Set up picture making…* in the model list for a one-click start with FLUX.2 klein: it downloads `sd-server` and the model once (about 5 GB) and runs them only while a picture is being made. In Settings you can add any other model `sd-server` runs, or point HikariLM at a picture server you already run (AUTOMATIC1111, Forge, Draw Things).
- **Code editor**: open a folder, edit a file, and ask about it in the chat beside it. It has syntax colours, suggestions as you type (Tab takes one), and a Markdown preview. Press Apply on a code block in a reply to put it into the file. Nothing is written until you press Save.
- **Remembers earlier chats**: with `nomic-embed-text` installed, a few matching notes from past chats are sent along with a new question.
- **Long chats** are summarized when they get close to the context length, the way Open WebUI does it. The whole chat stays saved.
- **Organize chats**: pin them, put them in folders, rename them. New chats are named from their first message.
- **LM Studio**: if its server (llmster) is running, its models join the list under *LM Studio*.

## Models

Any chat model works. These are what each part is tuned for:

| For | Model |
|---|---|
| General chat | `deepseek-r1:8b`, `qwen3`, or any chat model |
| Questions about code | a coder model, such as `qwen2.5-coder:7b` |
| Pictures you attach | any vision model, such as `gemma3`, `llava`, or `qwen2.5vl` |
| Remembering earlier chats | `nomic-embed-text` |

```bash
ollama pull qwen2.5-coder:7b && ollama pull gemma3 && ollama pull nomic-embed-text
```

## Keys

| Keys (Mac / Linux) | Does |
|---|---|
| ⌘B / Ctrl+B | Hide or show the chat list, or the file list in the code editor |
| ⌘K / Ctrl+K | Code editor: add the open file to your message |
| ⌘L / Ctrl+L | Code editor: add the highlighted lines, or the cursor's line |
| ⌃⌥N / Ctrl+Alt+N | New chat (change it in Settings) |
| ⌘S / Ctrl+S | Code editor: save |

## Where things are saved

Everything is in `~/.hikarilm`: chats, drafts, settings, notes, and pictures you made. Your project folders stay where they are.

## Settings from the environment

| Variable | Default | Does |
|---|---|---|
| `OLLAMA_HOST` | `http://127.0.0.1:11434` | Where Ollama is |
| `HIKARILM_HOME` | `~/.hikarilm` | Where HikariLM saves things |
| `HIKARILM_PORT`, `HIKARILM_HOST` | `8787`, `127.0.0.1` | Where the page is served |
| `HIKARILM_KEEP_ALIVE` | `5m` | How long a model stays loaded after its last reply |
| `HIKARILM_MEMORY_GB` | this computer's | Memory to plan for, when Ollama runs on another machine |
| `HIKARILM_IMAGES` | automatic | `on` or `off` for Ollama's own picture models (Apple Silicon only) |
| `LMSTUDIO_HOST`, `LM_API_TOKEN` | `http://127.0.0.1:1234` | Where LM Studio's server is, and its key |

## Development

The helper is Python with the standard library only, and the page is plain HTML, CSS, and JavaScript modules, so there is no build step. Run the tests from this folder with `python3 -m unittest`. The code editor is a CodeMirror bundle in `web/vendor/`; `tools/editor/build.sh` rebuilds it (only needed to change it).

## License

Apache 2.0. The libraries in `web/vendor/` keep their own licenses, next to them.
