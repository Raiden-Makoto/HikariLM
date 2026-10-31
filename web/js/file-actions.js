// What can be done to the open file: open, save, apply a code block, create,
// delete, and ask the chat about it.
import { state } from "./state.js";
import { linesKeys, messageBox, modKeys, modelSelect, plainError } from "./core.js";
import { MAX_FILES_TOTAL, renderChips } from "./attachments.js";
import { notify } from "./text-files.js";
import { codeRank, fillModels, modelLabel, modelsForView } from "./models.js";
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
  // Saving keeps every change from a reply that is still waiting to be decided.
  if (editor?.changesLeft()) editor.acceptAll();
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

export async function refreshList() {
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
// is highlighted), into the next message. Cmd/Ctrl+K and Cmd/Ctrl+Shift+K.
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
    note = " Model: " + modelLabel(coder) + ".";
  } else if (modelSelect.value) {
    note = " No coder model is installed, so " + modelLabel(modelSelect.value) + " will answer.";
  }
  const same = state.attached.findIndex((file) => file.name === name);
  if (same >= 0) state.attached.splice(same, 1);
  state.attached.push({ name, text });
  renderChips();
  saveDraft(draftKey(), true);
  notify("Attached " + name + "." + note + " Type your question and press Enter.");
  messageBox.focus();
}

// Right-click in the editor opens the page's menu instead of the browser's
// (Shift+right-click still gets the browser's). Focus stays in the editor so
// Cut and Copy act on its highlight.
const menu = document.querySelector("#editor-menu");
const fileHost = document.querySelector("#file-text");
let keptHighlight = false;

function closeMenu() {
  menu.hidden = true;
}

function openMenu(event) {
  if (event.shiftKey || !editor || !state.openPath) return;
  event.preventDefault();
  // Without a highlight the cursor goes where the click was, so adding code
  // means the clicked line (not the old cursor line, nor the word Mac
  // browsers highlight under a right-click).
  if (!keptHighlight) editor.cursorAt(event.clientX, event.clientY);
  const { from, to } = editor.getSelection();
  const highlighted = keptHighlight && from !== to;
  menu.querySelector('[data-do="cut"]').disabled = !highlighted;
  menu.querySelector('[data-do="copy"]').disabled = !highlighted;
  menu.querySelector('[data-do="lines"]').firstChild.textContent = highlighted ? "Add code to chat " : "Add this line to chat ";
  menu.hidden = false;
  const x = Math.min(event.clientX, window.innerWidth - menu.offsetWidth - 4);
  const y = Math.min(event.clientY, window.innerHeight - menu.offsetHeight - 4);
  menu.style.left = Math.max(4, x) + "px";
  menu.style.top = Math.max(4, y) + "px";
}

function startMenu() {
  document.querySelector("#menu-lines-keys").textContent = linesKeys();
  document.querySelector("#menu-file-keys").textContent = modKeys("K");
  fileHost.addEventListener("mousedown", (event) => {
    if (event.button !== 2 && !(event.button === 0 && event.ctrlKey)) return;
    const selection = editor?.getSelection();
    keptHighlight = Boolean(selection && selection.from !== selection.to);
    // A right-click keeps the highlight it lands on instead of moving the cursor.
    if (keptHighlight && !event.shiftKey) event.preventDefault();
  }, true);
  fileHost.addEventListener("contextmenu", openMenu);
  menu.addEventListener("mousedown", (event) => event.preventDefault());
  menu.addEventListener("click", (event) => {
    const item = event.target.closest("button");
    if (!item || item.disabled) return;
    closeMenu();
    const action = item.dataset.do;
    if (action === "lines" || action === "file") askAbout(action === "lines");
    else document.execCommand(action);
  });
  document.addEventListener("mousedown", (event) => {
    if (!menu.hidden && !menu.contains(event.target)) closeMenu();
  }, true);
  document.addEventListener("keydown", (event) => {
    if (!menu.hidden && event.key === "Escape") {
      event.preventDefault();
      closeMenu();
    }
  }, true);
  window.addEventListener("blur", closeMenu);
  window.addEventListener("resize", closeMenu);
  document.addEventListener("scroll", closeMenu, true);
}

// Listeners and first steps, run by main.js once every part is loaded.
export function start() {
  startMenu();
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
