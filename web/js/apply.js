// Apply, the way Cursor does it. A code block from a reply goes into the file
// it names (made first if it is new), merged with what is there: exactly when
// the block's unchanged lines show where it belongs, otherwise by the chosen
// model, which writes the whole file back. The result is shown as changes to
// accept or reject one by one, and nothing reaches the disk until Save.
import { state } from "./state.js";
import { isMac, modKeys, modelSelect, plainError, status } from "./core.js";
import { discardOk, editor, editorText, fileNote, loadEditor, previewShown, showPreview } from "./files.js";
import { openFileAt, refreshList } from "./file-actions.js";
import { closeFile } from "./folder-view.js";
import { refreshState } from "./models.js";
import { setBusy } from "./turns.js";
import { applyPrompt, fileFromReply, fitsOpenFile, isMarker, mergeEdit, resolvePath } from "./edit-blocks.js";

const reviewBox = document.querySelector("#review");
const reviewText = document.querySelector("#review-text");
const acceptButton = document.querySelector("#review-accept");
const rejectButton = document.querySelector("#review-reject");
// A file Apply made, still empty on disk. Rejecting its code removes it again.
let madeForApply = "";

// A block with no file name shows Apply only while a file in its language is open.
export function syncApplyButtons() {
  for (const button of document.querySelectorAll(".code-apply[data-lang]")) {
    const fits = fitsOpenFile(button.dataset.lang, state.openPath);
    button.hidden = !fits;
    if (fits) button.title = "Put this into " + state.openPath + ", to accept or reject change by change";
  }
}

export async function applyCode(code, named) {
  if (state.activeRequest) {
    fileNote.textContent = "Wait for the reply to finish, then apply.";
    return;
  }
  if (!state.source) {
    fileNote.textContent = "Open a folder first, then apply the code.";
    return;
  }
  let path = state.openPath;
  if (named) {
    const found = resolvePath(named, state.allFiles, state.folderName,
      state.source.kind === "helper" ? state.source.root : "");
    if (!found) {
      fileNote.textContent = named + " is not inside " + state.folderName + ", so the code was not applied.";
      return;
    }
    path = found.path;
    if (path !== state.openPath) {
      if (!discardOk()) return;
      // The file list can be cut short, so a file missing from it may still be there.
      const exists = found.exists || (await state.source.read(path).then(() => true, () => false));
      if (!exists) {
        try {
          await state.source.create(path);
        } catch (error) {
          fileNote.textContent = path + " could not be made: " + plainError(error.message);
          return;
        }
        madeForApply = path;
        await refreshList().catch(() => {});
      }
      await openFileAt(path, true);
      if (state.openPath !== path) return;
    }
  }
  if (!path) {
    fileNote.textContent = "Open a file first, then apply the code.";
    return;
  }
  await loadEditor();
  if (previewShown()) showPreview(false);
  const current = editorText();
  const merged = mergeEdit(current, code) ?? (await mergeWithModel(path, current, code));
  if (merged === null) return;
  if (state.openPath !== path || editorText() !== current) {
    fileNote.textContent = "The file changed while the code was being fitted in, so nothing was applied. Press Apply again.";
    return;
  }
  if (!editor.review(merged)) fileNote.textContent = "That code is already in " + path + ".";
}

// Like Cursor's apply model: the chosen model gets the file and the block and
// writes the whole file back. Its answer is checked before it is shown.
async function mergeWithModel(path, current, code) {
  const model = modelSelect.value;
  if (!model) {
    fileNote.textContent = "This code has to be fitted into " + path + " by a model. Pick one in the model list, then press Apply again.";
    return null;
  }
  const say = (text) => {
    fileNote.textContent = text;
    status.textContent = text;
  };
  const total = current.split("\n").length;
  state.activeRequest = new AbortController();
  setBusy(true);
  say("Fitting the code into " + path + " with " + model + "…");
  let reply = "";
  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model, messages: applyPrompt(path, current, code) }),
      signal: state.activeRequest.signal,
    });
    if (!response.ok) throw new Error((await response.json().catch(() => ({}))).error || "The model could not be reached.");
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop();
      for (const line of lines) {
        if (!line.trim()) continue;
        const event = JSON.parse(line);
        if (event.error) throw new Error(event.error);
        if (event.hikarilm === "answer_was_thinking") reply = "";
        if (event.hikarilm) continue;
        const piece = event.message?.content || "";
        if (!piece) continue;
        reply += piece;
        say("Fitting the code into " + path + " with " + model + "… " +
          Math.min(reply.split("\n").length, total) + " of about " + total + " lines written");
      }
    }
  } catch (error) {
    fileNote.textContent = error.name === "AbortError" ? "Stopped. Nothing was applied." : "Not applied: " + plainError(error.message);
    return null;
  } finally {
    state.activeRequest = null;
    setBusy(false);
    refreshState().catch(() => {});
  }
  let merged = fileFromReply(reply);
  if (merged === null || !merged.trim()) {
    fileNote.textContent = model + " did not write the whole file back, so nothing was applied.";
    return null;
  }
  const leftOut = merged.split("\n").some((line) => isMarker(line) && !current.includes(line.trim()));
  if (leftOut) {
    fileNote.textContent = model + " left parts of the file out, so nothing was applied. Try again, or a bigger model.";
    return null;
  }
  merged = merged.replace(/\r\n/g, "\n");
  if (current.endsWith("\n") && !merged.endsWith("\n")) merged += "\n";
  return current.includes("\r\n") ? merged.replace(/\n/g, "\r\n") : merged;
}

// Told by the editor how many changes are left to decide, and how a review ended.
export function reviewChanged({ left, ended }) {
  reviewBox.hidden = !left;
  if (left) {
    reviewText.textContent = left + (left === 1 ? " change" : " changes") +
      " from the reply. Accept or reject each one, or all of them. Nothing is saved until Save.";
    return;
  }
  const made = madeForApply;
  madeForApply = "";
  if (ended === "accepted") fileNote.textContent = "Accepted. Not saved yet: press Save to write it to disk.";
  if (ended !== "rejected") return;
  fileNote.textContent = "Rejected. The file is as it was.";
  if (made && made === state.openPath && !editorText() && !state.savedText) removeMade(made);
}

async function removeMade(path) {
  try {
    await state.source.remove(path);
  } catch (error) {
    return;
  }
  if (state.openPath === path) closeFile();
  await refreshList().catch(() => {});
  fileNote.textContent = "Rejected. " + path + " was new, so it was removed again.";
}

// Listeners and first steps, run by main.js once every part is loaded.
export function start() {
  acceptButton.textContent = "Accept all " + modKeys(isMac ? "⏎" : "Enter");
  rejectButton.textContent = "Reject all " + modKeys(isMac ? "⌫" : "Backspace");
  acceptButton.addEventListener("click", () => editor?.acceptAll());
  rejectButton.addEventListener("click", () => editor?.rejectAll());
}
