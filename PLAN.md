# Plan

LightUsagi is a local chat page for Ollama. Ollama runs on the machine, with one large model loaded at a time. A small helper reads and writes files on the SSD. The page is one browser tab with separate sections: a chat, a view of a small folder you already have, and a place to generate images.

No Docker. The memory Docker would reserve stays available for the loaded model and its KV cache.

## Pieces

Three pieces, and only two of them are ours.

- **Page.** Chat, a list of about six files and the one file you have open, and a separate Generate images section.
- **Helper.** Picks the model, reads the files and images that belong to this message, streams the reply, and saves the chat to disk. For an image request it asks the image model and saves the picture it makes. Follow-ups reuse the model Ollama still has loaded. The helper forces a large model out only when the next request needs a different large model. `nomic-embed-text` is small enough to load beside it.
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
| Generating an image, in its own section, on Apple Silicon only | `flux.2` (the name as `ollama list` shows it) |

The screen shows which model will answer before you send.

**Image generation models are not chat models.** `flux.2` and any model like it make pictures from a prompt and cannot answer a question. They never appear in the chat's model picker, in the same way `nomic-embed-text` does not. They appear only in the Generate images section, whose own picker lists only them. The helper enforces this as well as the page: a chat request that names an image model, or a generate request that names a text model, is refused with a plain message.

The helper tells the kinds apart by what Ollama reports a model can do, when it reports that, and otherwise by a short list of image model names kept in one place in the helper. A model it cannot place is treated as a chat model.

`nomic-embed-text` is not a chat model, and it is small enough to stay loaded next to whichever large model is already in memory. Indexing and search run without forcing Gemma, R1, or Qwen out. A six-file workspace does not go through the index. That file is small enough to send whole.

R1's reasoning is stored in the chat file. Later turns send R1 its earlier answers, not the reasoning trace.

## The chat page

- **Sidebar.** Past chats from `transcripts/`. Click one to open it. Double-click renames it. The `⋯` button on a chat opens a menu with Pin, Rename, Move to folder, and Delete (asks twice). It is a drawer on narrow screens.
- **Order.** Pinned chats come first, then folders, then the chats that are in neither. A chat shows in exactly one place, and pinned wins: a pinned chat that is also in a folder sits in Pinned with the folder's name beside it. Inside every group, and among the folders (by their newest chat), the most recently used is on top. A folder opens and closes with a click, and the page remembers which are closed.
- **Folders.** A folder is just a name on its chats, kept in each chat's file (`folder`), so there is no separate folder list to lose. Choosing "New folder name" in the Move menu creates one. A folder disappears when its last chat leaves. Pinning is the `pinned` flag in the same file. Saving a reply never changes either.
- **Deleting a folder never deletes a chat.** The folder's `⋯` menu has Rename folder and Delete folder. Delete asks twice, and the second prompt says how many chats stay. Every chat in it, pinned or not, loses only its `folder` and moves to the loose chats, keeping its messages, pin, and place in time. Deleting a folder that is already gone changes nothing and is not an error. If one chat file cannot be written, the others still move and the page says how many could not.
- **Names.** Folder names are trimmed, collapsed to single spaces, and cut at 40 characters. They match without regard to case, so typing `homework` puts a chat in the existing `Homework`. Renaming a folder to another folder's name merges the two. Changing only the case of a name is allowed. A blank name is refused.
- **Names.** A new chat is named from its first message: skip code blocks, drop greetings and filler ("hey, can you please…"), keep the first sentence, capitalise, cut at a word boundary. A name you typed is never overwritten by later saves.
- **Stop.** The send arrow becomes a stop square while streaming. Stopping keeps whatever was already written. Stopping before the first word removes the turn and puts your text back.
- **Transcript file.** `~/.lightusagi/transcripts/<id>.json` holds `id`, `title`, `created`, `updated`, `model`, and `messages`, and optionally `pinned` and `folder`. Each message has `role`, `content`, and optionally `thinking`, `model` and `files`. It is written to a temp file and renamed, so a crash never leaves half a chat.
- **Drafts and reload.** Typing and attachments are saved to `drafts/` a moment after each change, and when the tab is hidden or closed. After a reload the page reopens the chat that was open and puts the draft back in the message box. Switching chats keeps each chat's own draft. If a send is stopped before the first word, the text and files go back into the box and into the draft.
- **Welcome.** An empty chat greets on one line by the clock on the user's computer: "Good morning, what are we working on?" from 5:00 to 11:59, "Good afternoon" from 12:00 to 16:59, "Good evening" from 17:00 to 23:59, and "Late one" from 0:00 to 4:59. A page left open re-checks every minute and when the user returns to it, so it never keeps an old greeting. Under it, a line says what can be dropped in and a reminder that Enter sends. Lines are balanced so a wrap never leaves a single word alone.
- **Notifications.** When a reply finishes (or stops with an error) while the user is in another tab or window, the tab title changes to "● Reply ready" until they come back, and the browser shows a notification with the chat's name and the start of the answer if it was allowed. Clicking it opens that chat. A reply the user stopped does not notify. The browser asks for permission the first time a message is sent, and if it is refused the title change still works. Nothing is sent anywhere: it is the browser's own notification.
- **Settings.** A gear in the header opens a small popup with two compact switches and the background picker: **Dark mode** (off by default, so the page starts light) and **Notify when a reply finishes** (on by default). Both are remembered in the browser and applied before the page paints, so a reload never flashes the wrong theme. The background image is chosen and removed from the same popup.
- **Model picker.** Lives in the message box and lists what Ollama has, without `nomic-embed-text` and without image generation models.
- **Sections.** The header switches between Chat, the file view, and Generate images. The chat sidebar belongs to Chat only. Each section keeps its own draft.

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

### The Files section

- **Opening a folder.** A path box at the top of the section ("Folder to open", for example `~/projects/hw3`) and an Open button. The browser cannot see folder paths, so the path is typed, and the helper reads it from the machine it runs on. The page remembers the last folder in the browser, and the helper keeps nothing, so nothing is added to `~/.lightusagi`. A path that is missing, is not a folder, or cannot be read gets a one-line reason.
- **The list.** The files inside that folder, in subfolders too (up to four levels deep and 200 files), by name. Only the everyday kinds from "Attaching files" are listed, so `.asm`, model files, and the like never show. Folders that start with a dot, `node_modules`, `__pycache__`, `venv`, and `.git` are skipped. The list is meant for a small folder and says so when it was cut short. Click a file to open it.
- **The open file.** A plain editable text area in a monospace font, with the file's path above it. Files larger than 500 KB, files with binary bytes, and files that are not in the list are refused with the reason. UTF-8 and UTF-16 text open as they are, and Windows-1252 text is read and written back in the same encoding.
- **Asking about it.** Two buttons: "Ask about this file" and, when text is highlighted, "Ask about selection". Either one switches to Chat, picks `qwencoder2.5-7b` if it is installed (and says so if it is not), attaches the file text or the highlighted lines to the message box, and puts the cursor in the box. The user types the question and presses Enter, so the picker has shown which model will answer before anything is sent. The text is read again from disk when the button is pressed, not taken from the screen, so it is what is really in the file. If the text area has edits that are not saved, the buttons ask to save first. The attachment is named with the path and, for a selection, the line numbers, such as `hw3/avl.py lines 12–30`, and is kept in the transcript like any attached file. Switching the model to `deepseekr1:8b` before sending works as in any chat.
- **Changing the file.** The user copies a fix out of the chat reply and pastes it into the text area, which replaces whatever is highlighted, then presses Save (or Ctrl/Cmd+S). Save writes the whole file back to the same path, through a temp file and a rename, so a crash never leaves half a file. The helper keeps no copy of the old file. Saving with no changes does nothing. A file that uses Windows line endings keeps them.
- **Safe writes.** The helper only writes a file that already exists inside the opened folder and passes the same list as above. It never creates files, never follows a path or a link out of the folder, and refuses `..`. When the page opens a file it also gets that file's version. If the file was changed on disk by something else since then, Save is refused and the page says so, with a choice to reload the file (losing the edits) or keep editing. Nothing is overwritten silently.
- **Unsaved edits.** The title shows a dot while there are unsaved changes. Leaving the tab, switching files, or opening another folder with unsaved edits asks first. The text area is not kept as a draft. The file on disk is the only place it lives.
- **Only this machine can ask.** The helper refuses a request that a web page from another address sent to it, so a site open in another tab cannot read or change the folder.
- **Not in this step.** Creating, renaming, or deleting files and folders, searching across files, syntax colours, and running code.

## Image, then code

Qwen cannot see images. Gemma writes the image down as text, and Qwen reads that text.

1. Load `gemma3`, forcing out whatever large model is loaded now. Gemma reads the image from its existing path and writes a structure under `descriptions/` in the app folder. For a diagram such as an AVL tree, that text is the tree: each node's value and its left and right children.
2. The code request needs Qwen, so the helper forces Gemma to unload and loads `qwencoder2.5-7b`. Qwen reads the text file and writes the code in the chat.
3. Follow-up questions to Qwen leave it loaded. Ollama drops it when the keepalive expires. nomic indexes the description and the answer while Qwen stays loaded.

If that description is already in the app folder and the image has not changed, Gemma is skipped. Qwen reads the saved text. nomic indexes the words Gemma wrote, not the pixels. A later question can find that description by search while Gemma stays unloaded.

## Generating images

A section of its own, like the file view, because making a picture is a different job from answering a message. It is not in the chat sidebar, and a generated image is not a chat turn.

- **Where it works.** `flux.2` runs in Ollama only on Apple Silicon Macs, so the section is turned off everywhere else. See "Turned off by architecture" below.
- **What is in it.** A prompt box, a picker that lists only image generation models, a small Options area (a few size presets and a seed, with sensible defaults hidden until opened), and a Generate button that becomes a Stop square while it works, the same way Send does. A status line says what is happening: "Loading `flux.2`…", then "Generating…". The finished picture appears under the box.
- **Trying again.** The picture has Regenerate (same prompt, same seed, so the same result) and Vary (same prompt, new seed). Prompt and seed are kept so a result can be repeated.
- **Gallery.** Earlier pictures sit under the current one, newest first, each with its prompt. Each can be downloaded, reopened with its prompt and seed back in the boxes, or deleted (asks twice). The gallery is read from `images/`, so it survives a reload and a restart.
- **Memory.** `flux.2` is a large model, so it follows the one-large-model rule: loading it forces the chat model out, and the next chat message forces it out again. Several generations in a row keep it loaded for the keepalive, so tweaking a prompt does not reload it each time. `nomic-embed-text` is not affected. Under the Generate button a quiet line says which chat model will be unloaded, so a reload of the chat model afterwards is not a surprise.
- **One at a time.** A second generation cannot start while one runs. Stop cancels it, and nothing half-made is saved.
- **Where pictures go.** Each result is written to `images/<id>.png` with `images/<id>.json` beside it holding the prompt, model, seed, size, and time, using a temp file and a rename so a crash never leaves half a picture. Deleting a picture removes both files. A picture whose notes file is missing or damaged still shows, without its prompt.
- **Prompt draft.** The prompt box is kept in `drafts/generate.json` the same way a chat draft is, and removed when it is emptied or a generation starts.
- **Notifications.** A finished or failed generation notifies when the user is in another tab, by the same rules and the same switch as a chat reply.
- **Not in this step.** Sending a generated picture into a chat for Gemma to look at, editing an existing picture, and any model that is not `flux.2`-style prompt-to-image.

### Turned off by architecture

`flux.2` is supported by Ollama only on Apple Silicon (an M-series Mac). On any other machine the Generate section is disabled, decided by the helper from the machine and not by the user.

- **The test.** The machine counts as Apple Silicon when the helper is running on macOS (`platform.system()` is `Darwin`) on an ARM processor. A Python that is running under Rosetta reports an Intel processor on an M-series Mac, so the helper also asks macOS directly (`sysctl hw.optional.arm64`) and trusts that answer. Intel Macs, Windows (including Windows on ARM), Linux (including Linux on ARM), and WSL are all "not supported". The test is one small function in the helper, so widening support later is a one-place change.
- **What the user sees.** The Generate entry in the header stays visible but is greyed out and not clickable. Its tooltip and, if opened by address, the section itself say why in one line: "Image generation needs a Mac with Apple Silicon. This computer is Linux x86_64." The prompt box, picker, and gallery are not shown. The chat is unaffected.
- **The helper enforces it.** `/api/state` reports whether image generation is supported and, if not, the reason, and the page reads that rather than guessing from the browser. A generate request on an unsupported machine is refused with the same reason, even if someone calls the helper directly.
- **Image models stay out of chat everywhere.** On an unsupported machine an image model that happens to be installed is still hidden from the chat picker and refused in `/api/chat`, because it could never answer a message there either.
- **Supported, but not ready.** On Apple Silicon the section is enabled. If no image model is installed it shows "No image model installed" and the command that adds one. If Ollama refuses the request (for example an Ollama too old to run `flux.2`), Ollama's own message is shown and nothing is saved.
- **Override.** `LIGHTUSAGI_IMAGES=on` turns the section on and `LIGHTUSAGI_IMAGES=off` turns it off, whatever the machine is, for when Ollama's support widens or for testing. Without it the architecture test decides.
- **Start script.** `start.py` prints one line saying whether image generation is available on this computer, next to the Ollama model list.

The helper asks Ollama for the picture over HTTP like everything else, so no new component is added. If Ollama's request for image models turns out to differ from its chat request, that difference stays inside the helper.

## What is in memory

One large model at a time: Gemma, R1, Qwen, or the image model. Each request sets a keepalive, so a follow-up to the same model skips the load and Ollama resets the timer. Ollama unloads that model on its own when the timer runs out. The helper forces it out only to switch to a different large model. `nomic-embed-text` loads beside the large model and does not take its place.

The open chat and the open file are the only project text held in memory. Other chats, images, and descriptions stay on the SSD until a message needs them.

## What is on the SSD

LightUsagi keeps its own files in one folder in the home directory, `~/.lightusagi`, the same idea as `~/.ollama`. Your project folders and images stay at the paths they already have. The helper reads them from there and writes its own notes into the app folder.

```
~/.lightusagi/
  transcripts/     one file per conversation
  drafts/          what is typed and attached but not sent yet, one file per chat
  descriptions/    the text Gemma wrote for an image
  index/           the search index nomic wrote
  images/          the pictures the image model made, each with a small notes file
```

A description remembers which image it came from, and whether that image has changed, so Gemma can be skipped the next time.

Files attached through the page (see "Attaching files") add nothing new to this layout, and no file is written anywhere else on the SSD:

- The text taken from an attached text file, PDF, `.docx`, or `.xlsx` is saved inside the transcript of the chat it was attached to, in `transcripts/`. That is the only copy, and it goes when the chat is deleted.
- The original file and any attached picture are never saved in a transcript. The transcript keeps the file name.
- The one exception is a draft. While a message is typed but not sent, the page keeps the text, the extracted file text, and any attached pictures in `drafts/<chat id>.json` (`new.json` before a chat has started), so a reload does not lose them. The file is deleted when the message is sent, when the draft is emptied, and when its chat is deleted. Nothing in `drafts/` outlives the message it was for.
- `images/` holds pictures LightUsagi made itself, with the prompt, seed, and model beside each. They are its own output, not copies of anything of yours. A picture leaves `images/` only when the user deletes it, or downloads a copy to wherever they choose. The prompt draft for the Generate section is `drafts/generate.json`.
- Nothing else in this folder is a copy of the workspace.

## Frameworks

The helper is a Python program and uses the standard library only. It serves the page, calls Ollama over HTTP, and reads and writes `~/.lightusagi`. It runs on the Python people already have installed. No web framework, and no embedding or tensor packages: nomic stays inside Ollama.

The page is HTML, CSS, and a short script. The helper serves those files. The chat stream is the browser's own `fetch`. The open file is a text area, so a question can point at real code and a small fix can be pasted or applied back. The one library the page carries is `pdf.js` in `web/vendor/`, for reading PDFs. There is no frontend framework and no desktop shell. You open the page in the browser you already have.

## Repo layout

```
PLAN.md          this file
README.md        what the project is and how to run it
start.py         checks Ollama, starts the helper, opens the page
start.sh         runs start.py on macOS and Linux
start.bat        runs start.py on Windows, by double-click
helper/          the Python program
web/             the HTML, CSS, and script; vendor/ holds pdf.js
```

The start scripts are the one way to run it, so nobody has to find `helper/` or know how it is launched. They add no dependency beyond Python.

Models are routing rules inside the helper, not folders. The page is one tab with sections, not an app per feature.

## Build order

Each step is usable before the next one starts.

1. **Helper talks to Ollama.** Stream a reply and leave the model loaded for follow-ups. Each request sets a keepalive so Ollama resets that timer. Force a large model out only when the next request uses a different large model.
2. **Chat page.** Send a message, see the stream, save and reopen the chat from `transcripts/`. Sidebar, rename, attachments and drag-and-drop are part of this step. Done, except that it has only been tried against a fake Ollama.
3. **Folder view.** The Files section: open a small workspace, show one file, ask Qwen what it does or where the bug is. Paste or apply a small fix back to that file. The header gets its Chat and Files switch here.
4. **Image handoff.** Gemma writes the structure into `descriptions/`. Switching to Qwen forces Gemma out. Follow-ups stay on Qwen until the keepalive expires or the model changes. Skip Gemma when the description is already saved and the image has not changed.
5. **Search index.** nomic indexes saved text while the chat model stays loaded. A later question can include a few matching notes.
6. **Generate images.** Add the architecture test and report it in `/api/state`. Keep image models out of the chat picker and refuse them in `/api/chat`. Add the Generate section, disabled with its reason on anything but Apple Silicon. Where it is enabled, add its own picker, run one generation at a time under the one-large-model rule, and save each picture and its notes to `images/`. Usable on its own, so it does not wait for steps 3 to 5.

   Only the disabled path, the model filtering, and the `images/` handling can be tried on a machine that is not an Apple Silicon Mac, with a fake Ollama standing in. The generation itself has to be tried on a Mac with Apple Silicon, so it is not counted as done until it has run there.
