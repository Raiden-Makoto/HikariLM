# Plan

LightUsagi is a local chat page for Ollama. Ollama runs on the machine, with one large model loaded at a time. A small helper reads and writes files on the SSD. The page is one browser tab: a chat, and a view of a small folder you already have.

No Docker. The memory Docker would reserve stays available for the loaded model and its KV cache.

## Pieces

Three pieces, and only two of them are ours.

- **Page.** Chat, a list of about six files, and the one file you have open.
- **Helper.** Picks the model, reads the files and images that belong to this message, streams the reply, and saves the chat to disk. Follow-ups reuse the model Ollama still has loaded. The helper forces a large model out only when the next request needs a different large model. `nomic-embed-text` is small enough to load beside it.
- **Ollama.** Already installed. It holds the weights and the KV cache.

The page talks to the helper. The helper talks to Ollama. The workspace folder stays where it already is. The helper reads from it and does not keep a second copy.

## Models

| This send | Model |
|---|---|
| General chat | `deepseekr1:8b` |
| A question about the open file | `qwencoder2.5-7b` |
| That same kind of question, if you switch it | `deepseekr1:8b` |
| A message that includes an image | `gemma3` |
| Indexing saved text for later search | `nomic-embed-text` |

The screen shows which model will answer before you send.

`nomic-embed-text` is not a chat model, and it is small enough to stay loaded next to whichever large model is already in memory. Indexing and search run without forcing Gemma, R1, or Qwen out. A six-file workspace does not go through the index. That file is small enough to send whole.

R1's reasoning is stored in the chat file. Later turns send R1 its earlier answers, not the reasoning trace.

## What you can ask about a file

The file view is for questions about code you already have: what this file does, where the bug is. You open one file, or highlight a section. The helper reads that text from disk for this message only.

The reply stays in the chat. The file changes only if you paste a small fix in, or apply it onto the selection. Applying a small fix writes that block back to the same path.

## Image, then code

Qwen cannot see images. Gemma writes the image down as text, and Qwen reads that text.

1. Load `gemma3`, forcing out whatever large model is loaded now. Gemma reads the image from its existing path and writes a structure under `descriptions/` in the app folder. For a diagram such as an AVL tree, that text is the tree: each node's value and its left and right children.
2. The code request needs Qwen, so the helper forces Gemma to unload and loads `qwencoder2.5-7b`. Qwen reads the text file and writes the code in the chat.
3. Follow-up questions to Qwen leave it loaded. Ollama drops it when the keepalive expires. nomic indexes the description and the answer while Qwen stays loaded.

If that description is already in the app folder and the image has not changed, Gemma is skipped. Qwen reads the saved text. nomic indexes the words Gemma wrote, not the pixels. A later question can find that description by search while Gemma stays unloaded.

## What is in memory

One large model at a time: Gemma, R1, or Qwen. Each request sets a keepalive, so a follow-up to the same model skips the load and Ollama resets the timer. Ollama unloads that model on its own when the timer runs out. The helper forces it out only to switch to a different large model. `nomic-embed-text` loads beside the large model and does not take its place.

The open chat and the open file are the only project text held in memory. Other chats, images, and descriptions stay on the SSD until a message needs them.

## What is on the SSD

LightUsagi keeps its own files in one folder in the home directory, `~/.lightusagi`, the same idea as `~/.ollama`. Your project folders and images stay at the paths they already have. The helper reads them from there and writes its own notes into the app folder.

```
~/.lightusagi/
  transcripts/     one file per conversation
  descriptions/    the text Gemma wrote for an image
  index/           the search index nomic wrote
```

A description remembers which image it came from, and whether that image has changed, so Gemma can be skipped the next time. Nothing in this folder is a copy of the workspace.

## Frameworks

The helper is a Python program and uses the standard library only. It serves the page, calls Ollama over HTTP, and reads and writes `~/.lightusagi`. It runs on the Python people already have installed. No web framework, and no embedding or tensor packages: nomic stays inside Ollama.

The page is HTML, CSS, and a short script. The helper serves those files. The chat stream is the browser's own `fetch`. The open file is a text area, so a question can point at real code and a small fix can be pasted or applied back. There is no frontend framework and no desktop shell. You open the page in the browser you already have.

## Repo layout

```
PLAN.md          this file
README.md        what the project is
helper/          the Python program, added when we build it
web/             the HTML, CSS, and script, added when we build it
```

Models are routing rules inside the helper, not folders. The page is one tab, not an app per feature.

## Build order

Each step is usable before the next one starts.

1. **Helper talks to Ollama.** Stream a reply and leave the model loaded for follow-ups. Each request sets a keepalive so Ollama resets that timer. Force a large model out only when the next request uses a different large model.
2. **Chat page.** Send a message, see the stream, save and reopen the chat from `transcripts/`.
3. **Folder view.** Open a small workspace, show one file, ask Qwen what it does or where the bug is. Paste or apply a small fix back to that file.
4. **Image handoff.** Gemma writes the structure into `descriptions/`. Switching to Qwen forces Gemma out. Follow-ups stay on Qwen until the keepalive expires or the model changes. Skip Gemma when the description is already saved and the image has not changed.
5. **Search index.** nomic indexes saved text while the chat model stays loaded. A later question can include a few matching notes.
