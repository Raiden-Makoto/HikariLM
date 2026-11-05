<img src="web/icon.svg" alt="" width="96">

# HikariLM

A light, local chat page with a code editor beside it. Models run inside HikariLM with Apple's [MLX](https://github.com/ml-explore/mlx). It keeps one large model in memory at a time and puts everything else (chats, files, pictures) on disk, so your RAM goes to the model.

No Docker, no Ollama, no account. Your chats never leave your computer.

## Run it

HikariLM runs only on a Mac with Apple Silicon (M1 or newer). You need Python 3.9 or newer.

The first time, from the HikariLM folder:

```bash
./start.sh
```

That installs the `hikarilm` command (a link in `~/.local/bin`, which it adds to your PATH in `~/.zshrc` if it is not there yet). From then on, in any folder:

```bash
hikarilm start        # start it; the terminal can be closed after
hikarilm stop         # stop it
hikarilm restart      # stop it and start it again
hikarilm status       # whether it runs, and what is loaded
hikarilm log          # the end of its log
hikarilm update       # git pull, and restart it if that brought anything
```

The first start makes `~/.hikarilm/venv` and installs `mlx-lm` and `mlx-vlm` into it (about 1 GB, once), and downloads Nomic's `nomic-embed-text-v1.5` (about 0.55 GB, once) for finding earlier chats. You see that in the terminal. Then HikariLM keeps running in the background and opens http://127.0.0.1:8787, its welcome page (what HikariLM is, and the models to get), and the terminal can be closed. *Open HikariLM* there goes to the chat or the code editor, whichever you used last; http://127.0.0.1:8787/chat and /editor go straight to them. `hikarilm start` while it runs just opens the page; if the files were updated since it started, it restarts it first.

What it prints goes to `~/.hikarilm/hikarilm.log`, which keeps only its last 2,000 lines. Only failed requests are logged, not every one.

**Updating**: *Update* in Settings, or `hikarilm update`, runs `git pull` in the HikariLM folder and, when that brought anything, restarts HikariLM and reloads the page. It says first how many changes GitHub has. It needs a copy made with `git clone`, and refuses while a reply or a picture is being made. Chats, settings and models are kept.

Options: `--foreground` to run in the terminal until Ctrl+C, as before 4.3.0; `--port 9000` for another port; `--no-browser` to skip opening the page; `--version`.

## Models

Any MLX model works: whatever you download from Hugging Face is listed in the page, quantized or full precision. mlx-lm runs text models and mlx-vlm the ones that read pictures. Pick what fits your Mac's memory.

If you are not sure where to start, these fit a Mac with less than 18 GB of RAM (one is loaded at a time). The welcome page, http://127.0.0.1:8787, lists them with their download commands and says which you have.

| For | Suggested model | Size |
|---|---|---|
| Everyday chat, thinking or not | `mlx-community/Qwen3-8B-4bit` | 4.6 GB |
| Dense reasoning, math, step-by-step analysis | `mlx-community/DeepSeek-R1-0528-Qwen3-8B-4bit` | 4.6 GB |
| Questions about code | `mlx-community/Qwen2.5-Coder-7B-Instruct-4bit` | 4.3 GB |
| Pictures you attach | `mlx-community/gemma-3-12b-it-4bit` | 8.1 GB |
| Making pictures | `mlx-community/FLUX.2-Klein-4B-4bit` | 4.6 GB |
| Reading answers aloud | `mlx-community/Kokoro-82M-bf16` | 0.4 GB |
| Speaking instead of typing | `mlx-community/whisper-large-v3-turbo-asr-fp16` | 1.6 GB |
| Optional: BitNet | any BitNet model, `<bitnet model>` | varies |

DeepSeek R1 always thinks, at length: that is what makes it the one for hard math, proofs and problems worked through step by step, and also what makes it slow on simple questions, where Qwen3 with thinking off answers straight away.

BitNet models have every weight −1, 0 or 1. HikariLM runs their makers' files as they are and computes each layer the way the makers' code does. None is recommended over the others; these work: the Falcon3 and Falcon-E 1.58-bit family, Microsoft's `microsoft/bitnet-b1.58-2B-4T` (and its `-bf16` master weights, packed when loaded), `HF1BitLLM/Llama3-8B-1.58-100B-tokens`, and mlx-community's copies of them. GGUF files (for bitnet.cpp) and BitNet experiments with their own code, such as `1bitLLM/bitnet_b1_58-large`, are not MLX models and are not listed.

Pictures are drawn by Black Forest Labs' FLUX.2 klein through [mflux](https://github.com/filipstrand/mflux). The 4B in the table is the one to take with less than 18 GB of RAM; with 18 GB or more, the 9B (`mlx-community/flux2-klein-9b-4bit`, 9.5 GB) draws more detail. Pick *Get a picture model…* in the model list, or open http://127.0.0.1:8787/setup/pictures, for a setup page that walks through it and says how far along you are. Every FLUX.2 klein works: 4B and 9B, 9B-KV, and the base 4B and 9B (slower, 50 steps, with guidance), from `black-forest-labs` or mlx-community's and mflux-community's MLX copies at any bit width. Once one is downloaded, `hikarilm restart` puts mflux in a venv of its own, `~/.hikarilm/pictures-venv`, once, so it never changes the packages the chat models run with. Only the models themselves are listed: fine-tunes, LoRA adapters and other people's re-uploads are left out, and the start window says which were skipped.

LoRAs, small add-ons that teach FLUX.2 klein a style, a character or an object, draw on FLUX.2 klein base 4B. Add one in Settings by its Hugging Face repo (`user/name`): HikariLM looks it up, says what it was made for and its trigger word, refuses what cannot work on the base 4B (made for the 9B or FLUX.1, or a DoRA), and gives the `hf download` command. Once it is downloaded, pick it and its strength under Options; the trigger word, if it has one, is put in front of the prompt (many LoRAs need none: leave it empty), and the picture's caption names the LoRA. A LoRA trained on booru tags (`1girl, blue_hair`) rather than sentences can have *Prompt with tags* turned on: if you have a model that turns sentences into tags, pick it under Settings, *Make tags with*, and *Make tags*, by Options, turns your sentence into tags to read and edit before drawing. A picture with a LoRA is drawn by mflux's own command (`mflux-generate-flux2 --config-from-conf`), which loads the model with the LoRA for each picture. To make one of your own with mflux's trainer, see http://127.0.0.1:8787/setup/lora.

With more memory, take bigger ones or less quantized ones (8-bit instead of 4-bit, say) from [mlx-community](https://huggingface.co/mlx-community).

```bash
~/.hikarilm/venv/bin/hf download mlx-community/Qwen2.5-Coder-7B-Instruct-4bit   # or any other MLX model
```

Models are found in the Hugging Face cache, and in folders of their own in `~/.hikarilm/models` (for ones you converted yourself). Reload the page after a download.

## What it does

- **Chat** with any of your models. Replies are drawn as Markdown, with tables, code, and math. A model's thinking is folded into one line, and Settings can cap how long it thinks.
- **Attach files**: text and code, PDFs, Word and Excel files, and pictures. A picture goes to a model that can read it (any vision model mlx-vlm runs, such as Gemma 3 or Qwen2.5-VL). Long documents are read in parts so they fit the model's context, or only the passages closest to the question are sent.
- **Remembers earlier chats**: a few notes from earlier chats and pictures that match your question are sent along with it. They are found by meaning with nomic-embed-text, which HikariLM runs in MLX beside the chat model (0.3 GB). Settings can turn this off.
- **Python the models can run**: instead of working out "the square root of 4761" or a standard deviation by hand, a chat model can write Python and run it, the way ChatGPT does, then answer from what it printed. numpy, sympy, pandas, scipy and matplotlib are there, and plots come back as pictures. Each run shows above the answer as *Ran Python*; open it for the code, its output and its plots, which are saved with the chat. Scripts run in macOS's own sandbox (`sandbox-exec`): no internet, no other programs, nothing of yours to read, and writing only in a scratch folder that is deleted after; a script stops after 30 seconds or 1.25 GB of memory, and output past 25,000 characters is cut. Its packages are in a venv of their own, `~/.hikarilm/python-venv`, which `hikarilm restart` makes once. It needs Python 3.11 or newer, since older ones only get a scipy that macOS 27 refuses to load; if HikariLM runs on an older Python, it uses a newer one from Homebrew or python.org, or says to install one (`brew install python@3.13`). Not every model calls it; coding models (Qwen2.5 Coder and the like), DeepSeek R1 and BitNet models never get it, since they did not use the calculator before it well. Settings can turn it off.
- **Answers read aloud** with Kokoro: press *Listen* under an answer, or turn on *Read answers aloud when they finish* in Settings, which also picks one of Kokoro's 28 English voices and the speed. Reading starts with the first sentence while the rest are made. Code blocks are named rather than read ("There is a Python code block here, 3 lines long"), small tables are read row by row, math is said in words ("x squared plus 1 equals 0") or named when it is too much to say, links by their text, plots from Python as "There is a plot above", and emoji, markup and the model's thinking never. Kokoro (0.4 GB) stays loaded beside the chat model. Once it is downloaded, `hikarilm restart` makes it a venv of its own, `~/.hikarilm/voice-venv`, with mlx-audio and misaki (Kokoro's English reader), with Python 3.11 or 3.12, which misaki needs; it finds one from Homebrew or python.org, or says to install one (`brew install python@3.12`).
- **Speak instead of typing** with Whisper: press the microphone by Attach, speak, and press it again; what you said is written into the message box where the cursor was, to read and fix before you send (Esc cancels). The browser asks for the microphone the first time, and if it is blocked the page says how to allow it, in the browser and in macOS. The sound never leaves the Mac: the page makes it 16 kHz and Whisper writes it down here, 30 seconds at a time, up to 10 minutes a recording. Settings, *Speech to text*, picks the Whisper and the language (or detects it). Only mlx-community's `-asr-` Whisper repos work, since mlx-audio needs the tokenizer in the model's folder; `whisper-small-asr-fp16` (0.5 GB) is the smaller one. It uses the same `~/.hikarilm/voice-venv` as Kokoro, made by `hikarilm restart`, and stays loaded beside the chat model until 5 minutes after it was last used.
- **Your rules**: standing instructions such as "You are a senior engineer. Answer with the code first." go in Settings and are sent first, before every chat, the way Claude Code reads `CLAUDE.md`. In the code editor a folder's own `HIKARILM.md` (or `AGENTS.md`, `CLAUDE.md` or `.cursorrules`) is added after them, and the line under the folder says which. A chat can turn them off in its menu. They are never sent for rewriting, Apply, summaries or notes. Small models follow them more loosely than large ones.
- **Make pictures** with FLUX.2 klein: pick it under *Make pictures* in the model list and describe the picture. It takes the chat model's place in memory while it draws and leaves after the keepalive, like any other model. FLUX.2 klein draws what a prompt describes, not what it names, so Rewrite (next to Options) has a chat model rewrite the request into how things look (a general model to start; Settings, Rewrite prompts with, picks which one, including a coder if you want, or turns it off; models that always think are never used for it). The rewrite appears in the message box to edit or undo before it draws; Send alone draws your words as they are. Attach up to four reference pictures and FLUX.2 klein draws from them as well as the prompt, and a model that can see describes them for the rewrite (Settings, Read pictures with, picks which one; the first installed by default). After a picture, a message like "add a sunset to the background" changes that picture instead of drawing a new one; the note above the message box says which it will do and turns it the other way.
- **Code editor**: open a folder, edit a file, and ask about it in the chat beside it. It has syntax colors, suggestions as you type (Tab takes one), and a Markdown preview. Apply works like Cursor's. The model labels code meant for a file with that file's path, and Apply puts it into that file, or creates the file if it is new. A block can leave out unchanged code with `... existing code ...` comments. If it can't be placed exactly, the chosen model writes the file with the change in it. Each change then shows in red and green to accept or reject. Commands, output and examples get no Apply button. Nothing is written until you press Save.
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

Everything is in `~/.hikarilm`: chats, drafts, settings, your rules (`rules.md`, plain Markdown you can edit anywhere), pictures you made, the notes of earlier chats (`index/`), and the venvs (MLX, mflux for pictures, Python for the models, and Kokoro for reading aloud). The models are in the Hugging Face cache (`~/.cache/huggingface/hub`). Your project folders stay where they are.

## Settings from the environment

| Variable | Default | Does |
|---|---|---|
| `HIKARILM_HOME` | `~/.hikarilm` | Where HikariLM saves things |
| `HIKARILM_PORT`, `HIKARILM_HOST` | `8787`, `127.0.0.1` | Where the page is served |
| `HIKARILM_KEEP_ALIVE` | `5m` | How long a model stays loaded after its last reply |
| `HF_HUB_CACHE`, `HF_HOME` | Hugging Face's | Where the downloaded models are |

## Development

The helper is Python with the standard library only; the models run in a separate worker process with the venv's Python. The page is plain HTML, CSS, and JavaScript modules, so there is no build step. The code editor is a CodeMirror bundle in `web/vendor/`; `tools/editor/build.sh` rebuilds it (only needed to change it).

## License

Apache 2.0. The libraries in `web/vendor/` keep their own licenses, next to them.
