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

## The chat page

- **Sidebar.** Past chats, newest first, from `transcripts/`. Click to open, pencil or double-click to rename, trash (asks twice) to delete. It is a drawer on narrow screens.
- **Names.** A new chat is named from its first message: skip code blocks, drop greetings and filler ("hey, can you please…"), keep the first sentence, capitalise, cut at a word boundary. A name you typed is never overwritten by later saves.
- **Stop.** The send arrow becomes a stop square while streaming. Stopping keeps whatever was already written. Stopping before the first word removes the turn and puts your text back.
- **Transcript file.** `~/.lightusagi/transcripts/<id>.json` holds `id`, `title`, `created`, `updated`, `model`, and `messages`. Each message has `role`, `content`, and optionally `thinking`, `model` and `files`. It is written to a temp file and renamed, so a crash never leaves half a chat.
- **Drafts and reload.** Typing and attachments are saved to `drafts/` a moment after each change, and when the tab is hidden or closed. After a reload the page reopens the chat that was open and puts the draft back in the message box. Switching chats keeps each chat's own draft. If a send is stopped before the first word, the text and files go back into the box and into the draft.
- **Model picker.** Lives in the message box and lists what Ollama has, without `nomic-embed-text`.

## Attaching files

The paperclip button and drag-and-drop on the page both attach files to the next message.

- **Text, markdown, code, CSV and TSV files** (500 KB each, 2 MB of text per message). Only everyday kinds are accepted: `.txt .md .rst .log .csv .tsv .json .yaml .toml .ini .xml .html .css`, code in `.py .r .c .h .cpp .hpp .java .kt .go .rs .rb .php .swift .cs .scala .lua .pl .js .ts .jsx .tsx .sh .sql .tex`, and `Makefile`, `Dockerfile`, `README`, `LICENSE`. Anything else, such as `.asm` or model files like `.ckpt`, is refused with a note, even when it happens to be readable text. The text goes to the model inside the message. The chat shows only the file name. The text is saved in the transcript, so reopening the chat keeps the context. A longer file is cut at a line break and the user is told how many lines were kept. The page reads UTF-8, UTF-16 and Windows-1252, so a CSV saved by Excel opens correctly. A file with binary bytes is refused.
- **Images** (png, jpeg, webp, gif, up to 10 MB each, 30 MB per message). Attaching one switches the model to `gemma3` and tells the user. Only the message that carries the image sends it. The transcript keeps the file name and not the picture, so later turns rely on Gemma's own answer. If no Gemma model is installed the image is refused and the user is told.
- **PDFs** (up to 50 MB, first 25 pages). None of the chat models can read a PDF, so the page pulls the text out first and sends that text like any other text file. The extractor is `pdf.js`, kept as two files in `web/vendor/` and loaded only when a PDF is attached. A scanned PDF has no text inside, so its first 4 pages go to `gemma3` as pictures, with the same model switch and notice as an image.
- **Word and Excel** (`.docx` and `.xlsx`, up to 50 MB). Both are zip files of XML, and the browser can unzip them itself, so this needs no library. A `.docx` becomes its paragraphs, with table cells separated by tabs. An `.xlsx` becomes one tab-separated block per sheet, using the values Excel last calculated, with dates shown as Excel's serial numbers. Old `.doc` and `.xls` are refused with a note to save them in the new format.
- Other files are skipped, with the reason shown.

The browser cannot see a file's folder path, so this attaches a copy of the contents for one message. Reading from a path in the workspace (step 3) is separate.

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
  drafts/          what is typed and attached but not sent yet, one file per chat
  descriptions/    the text Gemma wrote for an image
  index/           the search index nomic wrote
```

A description remembers which image it came from, and whether that image has changed, so Gemma can be skipped the next time.

Files attached through the page (see "Attaching files") add nothing new to this layout, and no file is written anywhere else on the SSD:

- The text taken from an attached text file, PDF, `.docx`, or `.xlsx` is saved inside the transcript of the chat it was attached to, in `transcripts/`. That is the only copy, and it goes when the chat is deleted.
- The original file and any attached picture are never saved in a transcript. The transcript keeps the file name.
- The one exception is a draft. While a message is typed but not sent, the page keeps the text, the extracted file text, and any attached pictures in `drafts/<chat id>.json` (`new.json` before a chat has started), so a reload does not lose them. The file is deleted when the message is sent, when the draft is emptied, and when its chat is deleted. Nothing in `drafts/` outlives the message it was for.
- Nothing else in this folder is a copy of the workspace.

## Frameworks

The helper is a Python program and uses the standard library only. It serves the page, calls Ollama over HTTP, and reads and writes `~/.lightusagi`. It runs on the Python people already have installed. No web framework, and no embedding or tensor packages: nomic stays inside Ollama.

The page is HTML, CSS, and a short script. The helper serves those files. The chat stream is the browser's own `fetch`. The open file is a text area, so a question can point at real code and a small fix can be pasted or applied back. The one library the page carries is `pdf.js` in `web/vendor/`, for reading PDFs. There is no frontend framework and no desktop shell. You open the page in the browser you already have.

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
2. **Chat page.** Send a message, see the stream, save and reopen the chat from `transcripts/`. Sidebar, rename, attachments and drag-and-drop are part of this step. Done, except that it has only been tried against a fake Ollama.
3. **Folder view.** Open a small workspace, show one file, ask Qwen what it does or where the bug is. Paste or apply a small fix back to that file.
4. **Image handoff.** Gemma writes the structure into `descriptions/`. Switching to Qwen forces Gemma out. Follow-ups stay on Qwen until the keepalive expires or the model changes. Skip Gemma when the description is already saved and the image has not changed.
5. **Search index.** nomic indexes saved text while the chat model stays loaded. A later question can include a few matching notes.
