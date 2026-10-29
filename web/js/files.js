// The code editor's state: the open folder and file, the editor itself, and the helper's folders.
import { state } from "./state.js";
import { recallHandle } from "./local-folder.js";
import { restoreFile } from "./sections.js";
import { openFolder, openLocalFolder, showFolderDialog } from "./folder-view.js";
import { saveOpenFile } from "./file-actions.js";
import { renderMarkdown } from "./markdown.js";
import { reviewChanged, syncApplyButtons } from "./apply.js";

// Files section. The helper reads and writes the folder; the page only
// shows one file at a time. The file on disk is the only copy of the text.
export const FOLDER_KEY = "hikarilm-folder";
export const KIND_KEY = "hikarilm-folder-kind";
export const folderLabel = document.querySelector("#folder-label");
export const fileFilter = document.querySelector("#file-filter");
export const folderNote = document.querySelector("#folder-note");
export const fileList = document.querySelector("#file-list");
const fileTitle = document.querySelector("#file-title");
export const fileNote = document.querySelector("#file-note");
const fileHost = document.querySelector("#file-text");
// The editor is CodeMirror, one file loaded the first time the code editor
// is opened. Until it arrives there is simply an empty box.
export let editor = null;
let editorLoading = null;
export function loadEditor() {
  if (editor) return Promise.resolve(editor);
  if (!editorLoading) {
    editorLoading = import("/vendor/codemirror.js").then((module) => {
      editor = module.createEditor(fileHost, {
        onChange: () => {
          fileNote.textContent = "";
          refreshFileButtons();
        },
        onSelection: () => refreshFileButtons(),
        onSave: () => saveOpenFile(),
        onReview: reviewChanged,
      });
      return editor;
    }, (error) => {
      editorLoading = null;
      throw new Error("The code editor did not load (" + (error.message || error) + ").");
    });
  }
  return editorLoading;
}
export const editorText = () => (editor ? editor.getValue() : "");

// A Markdown file can be shown drawn, the way a reply is, instead of as text.
// The preview is of the text in the editor, saved or not, and cannot be edited.
const preview = document.querySelector("#file-preview");
export const previewButton = document.querySelector("#preview-file");
const isMarkdown = (path) => /\.(md|markdown|rmd)$/i.test(path || "");
export const previewShown = () => !preview.hidden;
export function showPreview(on) {
  const shown = on && isMarkdown(state.openPath);
  if (shown) renderMarkdown(preview, editorText());
  else preview.replaceChildren();
  preview.hidden = !shown;
  fileHost.hidden = shown;
  previewButton.textContent = shown ? "Raw" : "Preview";
  previewButton.setAttribute("aria-pressed", String(shown));
}
export const saveFile = document.querySelector("#save-file");
export const newFile = document.querySelector("#new-file");
export const deleteFile = document.querySelector("#delete-file");
const diskNote = document.querySelector("#disk-note");
export const conflictBox = document.querySelector("#conflict");
// Where the open folder lives. Either the helper's machine (read and
// written by the helper) or this computer (a folder picked with the
// browser's own folder picker, read and written by the page itself).

export const hasEdits = () => Boolean(state.openPath) && editorText() !== state.savedText;

export async function filesApi(url, options) {
  const response = await fetch(url, options);
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    const error = new Error(body.error || "The helper could not do that.");
    error.conflict = Boolean(body.conflict);
    throw error;
  }
  return body;
}

export function helperSource(root) {
  const file = (path) => "/api/workspace/file?root=" + encodeURIComponent(root) + "&path=" + encodeURIComponent(path);
  return {
    kind: "helper",
    root,
    label: root,
    list: () => filesApi("/api/workspace?root=" + encodeURIComponent(root)),
    read: (path) => filesApi(file(path)),
    write: (path, text, version) => filesApi("/api/workspace/file", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ root, path, text, version }),
    }),
    create: (path) => filesApi("/api/workspace/file", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ root, path }),
    }),
    remove: (path) => filesApi(file(path), { method: "DELETE" }),
  };
}

export function discardOk() {
  return !hasEdits() || confirm("This file has changes that are not saved. Leave them behind?");
}

export function refreshFileButtons() {
  const open = Boolean(state.openPath);
  fileTitle.textContent = open ? state.openPath + (hasEdits() ? " ●" : "") : "No file open";
  saveFile.disabled = !open || state.saving || !hasEdits();
  diskNote.hidden = !open;
  previewButton.hidden = !isMarkdown(state.openPath);
  newFile.disabled = !state.source || state.saving;
  deleteFile.disabled = !open || state.saving;
  if (applyPath !== state.openPath) {
    applyPath = state.openPath;
    syncApplyButtons();
  }
}
let applyPath = null;

// Open the folder used last time. A folder from this computer needs the
// user's permission again after a reload, so it may only be offered.
export async function reopenLast() {
  if (localStorage.getItem(KIND_KEY) === "local") {
    const handle = await recallHandle();
    if (handle && (await handle.queryPermission({ mode: "readwrite" }).catch(() => "")) === "granted") {
      if (await openLocalFolder(handle, true)) {
        restoreFile();
        return;
      }
    }
  } else {
    const last = localStorage.getItem(FOLDER_KEY) || "";
    if (last && (await openFolder(last, true))) {
      restoreFile();
      return;
    }
  }
  showFolderDialog();
}
