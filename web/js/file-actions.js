// What can be done to the open file: open, save, apply a code block, create,
// delete, and ask the chat about it.
import { state } from "./state.js";
import { messageBox, modelSelect, plainError } from "./core.js";
import { MAX_FILES_TOTAL, renderChips } from "./attachments.js";
import { notify } from "./text-files.js";
import { codeRank, fillModels, modelsForView } from "./models.js";
import { draftKey, saveDraft } from "./chats.js";
import {
  conflictBox, deleteFile, discardOk, editor, editorText, fileList,
  fileNote, hasEdits, loadEditor, newFile, previewButton, previewShown, refreshFileButtons, saveFile, showPreview,
} from "./files.js";
import { FILE_KEY } from "./sections.js";
import { closeFile, renderFiles } from "./folder-view.js";

export async function openFileAt(path, force = false) {
  if (!force) {
    if (path === state.openPath && !hasEdits()) return;
    if (!discardOk()) return;
  }
  const token = ++state.openToken;
  fileNote.textContent = "Opening…";
  conflictBox.hidden = true;
  try {
    const [body, ed] = await Promise.all([state.source.read(path), loadEditor()]);
    if (token !== state.openToken) return;
    state.openPath = body.path;
    state.openVersion = body.version;
    state.savedText = body.text;
    ed.open(body.text, state.openPath);
    showPreview(false);
    fileNote.textContent = "";
    localStorage.setItem(FILE_KEY, JSON.stringify({
      kind: state.source.kind,
      folder: state.source.kind === "helper" ? state.source.root : state.source.root.name,
      path: state.openPath,
    }));
    fileList.querySelectorAll("button").forEach((button) => {
      if (button.title !== state.openPath) button.removeAttribute("aria-current");
      else button.setAttribute("aria-current", "true");
    });
  } catch (error) {
    if (token !== state.openToken) return;
    fileNote.textContent = plainError(error.message);
  }
  refreshFileButtons();
}

export async function saveOpenFile() {
  if (!hasEdits() || state.saving) return;
  state.saving = true;
  refreshFileButtons();
  const sent = editorText();
  const path = state.openPath;
  try {
    const body = await state.source.write(path, sent, state.openVersion);
    if (path === state.openPath) {
      state.openVersion = body.version;
      state.savedText = sent;
      fileNote.textContent = body.changed ? "Saved." : "Nothing to save.";
    }
  } catch (error) {
    if (error.conflict) {
      document.querySelector("#conflict-text").textContent = error.message;
      conflictBox.hidden = false;
      fileNote.textContent = "Not saved.";
    } else {
      fileNote.textContent = "Not saved: " + plainError(error.message);
    }
  }
  state.saving = false;
  refreshFileButtons();
}

// Put a code block from a reply into the editor: over the highlighted text
// if there is some, otherwise over the whole file after asking. It is an
// ordinary unsaved edit, so nothing reaches the disk until Save, and Ctrl+Z
// undoes it. If the file changed on disk meanwhile, Save still refuses.
// The editor leaves applied code highlighted. That highlight is not one the
// person made, so the next Apply must still ask rather than replace it.
let lastApplied = null;

export function applyBlock(code) {
  if (!state.openPath || !editor) {
    fileNote.textContent = "Open a file first, then apply the code.";
    return;
  }
  if (previewShown()) showPreview(false);
  const { from, to } = editor.getSelection();
  const current = editorText();
  const leftover = lastApplied && lastApplied.path === state.openPath && lastApplied.from === from &&
    lastApplied.to === to && current.slice(from, to) === lastApplied.text;
  const whole = from === to || leftover;
  const start = whole ? 0 : from;
  const end = whole ? current.length : to;
  if (whole && current && !confirm("Replace the whole of " + state.openPath + " with this code?\n\nNothing is saved until you press Save, and Ctrl+Z undoes it.\nTo replace only part of the file, highlight that part first.")) return;
  // Keep the trailing line break of whatever is being replaced; a whole file always gets one.
  let text = code;
  if ((whole || current.slice(start, end).endsWith("\n")) && !text.endsWith("\n")) text += "\n";
  editor.replace(start, end, text);
  lastApplied = { path: state.openPath, from: start, to: start + text.length, text };
  editor.focus();
  fileNote.textContent = (whole ? "Replaced the whole file" : "Replaced the highlighted text") +
    " with the code. Not saved yet: read it, then press Save. Ctrl+Z undoes it.";
}

async function refreshList() {
  const body = await state.source.list();
  state.allFiles = body.files;
  state.filesCut = body.cut;
  renderFiles();
}

// Both ask first. A new file is empty and never replaces one that exists.
async function createNewFile() {
  if (!state.source || state.saving) return;
  const typed = prompt("Name of the new file in " + state.folderName + " (for example notes.md or src/main.py):", "");
  if (typed === null) return;
  const path = typed.trim().replace(/\\/g, "/");
  if (!path) return;
  if (!discardOk()) return;
  try {
    await state.source.create(path);
    await refreshList();
    await openFileAt(path, true);
    fileNote.textContent = "Created " + path + ". It is empty.";
  } catch (error) {
    fileNote.textContent = "Not created: " + plainError(error.message);
  }
}

async function deleteOpenFile() {
  if (!state.source || !state.openPath || state.saving) return;
  const path = state.openPath;
  const warning = hasEdits() ? "\n\nYour unsaved changes will be lost too." : "";
  if (!confirm("Delete " + path + " from disk?\n\nThis removes the real file and cannot be undone." + warning)) return;
  try {
    await state.source.remove(path);
  } catch (error) {
    fileNote.textContent = "Not deleted: " + plainError(error.message);
    return;
  }
  closeFile();
  localStorage.removeItem(FILE_KEY);
  try {
    await refreshList();
  } catch (error) {
    fileNote.textContent = plainError(error.message);
    return;
  }
  fileNote.textContent = "Deleted " + path + ".";
}

// Put the open file, or its highlighted lines (the cursor's line if nothing
// is highlighted), into the next message. Cmd/Ctrl+K and Cmd/Ctrl+L.
export async function askAbout(selectionOnly) {
  if (!state.openPath) return;
  if (state.activeRequest) {
    fileNote.textContent = "Wait for the reply to finish, then add it.";
    return;
  }
  if (hasEdits()) {
    fileNote.textContent = "Save the file first, so the question is about what is really in it.";
    return;
  }
  let body;
  try {
    body = await state.source.read(state.openPath);
  } catch (error) {
    fileNote.textContent = plainError(error.message);
    return;
  }
  let text = body.text;
  let name = state.folderName + "/" + body.path;
  if (selectionOnly) {
    // Not dirty, so the highlight's offsets are offsets into the disk text.
    let { from: start, to: end } = editor ? editor.getSelection() : { from: 0, to: 0 };
    if (body.text !== editorText()) {
      fileNote.textContent = "The file changed on disk. Open it again first.";
      return;
    }
    if (start === end) {
      start = start === 0 ? 0 : text.lastIndexOf("\n", start - 1) + 1;
      const stop = text.indexOf("\n", end);
      end = stop < 0 ? text.length : stop;
    }
    text = text.slice(start, end);
    const first = body.text.slice(0, start).split("\n").length;
    const last = first + text.replace(/\n+$/, "").split("\n").length - 1;
    name += first === last ? " line " + first : " lines " + first + "–" + last;
  }
  const used = state.attached.reduce((sum, file) => sum + (file.text || "").length, 0);
  if (used + text.length > MAX_FILES_TOTAL) {
    fileNote.textContent = "That would make the attached files too big for one message.";
    return;
  }
  if (!text.trim()) {
    fileNote.textContent = "There is no text there to ask about.";
    return;
  }
  const coder = [...modelSelect.options].map((option) => option.value)
    .find((value) => value && codeRank(value) < 2);
  let note = "";
  if (coder) {
    modelSelect.value = coder;
    note = " Model: " + coder + ".";
  } else if (modelSelect.value) {
    note = " No coder model is installed, so " + modelSelect.value + " will answer.";
  }
  const same = state.attached.findIndex((file) => file.name === name);
  if (same >= 0) state.attached.splice(same, 1);
  state.attached.push({ name, text });
  renderChips();
  saveDraft(draftKey(), true);
  notify("Attached " + name + "." + note + " Type your question and press Enter.");
  messageBox.focus();
}

// Listeners and first steps, run by main.js once every part is loaded.
export function start() {
  newFile.addEventListener("click", createNewFile);
  deleteFile.addEventListener("click", deleteOpenFile);
  saveFile.addEventListener("click", saveOpenFile);
  previewButton.addEventListener("click", () => {
    showPreview(!previewShown());
    if (!previewShown() && editor) editor.focus();
  });
  document.querySelector("#conflict-keep").addEventListener("click", () => { conflictBox.hidden = true; });
  document.querySelector("#conflict-reload").addEventListener("click", () => {
    conflictBox.hidden = true;
    openFileAt(state.openPath, true);
  });
  window.addEventListener("beforeunload", (event) => {
    if (hasEdits()) event.preventDefault();
  });
  document.addEventListener("busychange", () => {
    refreshFileButtons();
    // A tab switched mid-reply left the picker on the other view's list.
    const shown = [...modelSelect.options].map((option) => option.value).filter(Boolean);
    if (!state.activeRequest && shown.join("\n") !== modelsForView().join("\n")) fillModels();
  });
}
