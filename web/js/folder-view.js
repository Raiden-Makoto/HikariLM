// The file list, opening a folder, and the folder chooser window.
import { state } from "./state.js";
import { plainError } from "./core.js";
import {
  conflictBox, discardOk, editor, fileFilter, fileList, fileNote,
  filesApi, FOLDER_KEY, folderLabel, folderNote, helperSource, KIND_KEY,
  refreshFileButtons, showPreview,
} from "./files.js";
import { localSource, recallHandle, rememberHandle } from "./local-folder.js";
import { openFileAt } from "./file-actions.js";

export function closeFile() {
  state.openToken += 1;
  state.openPath = "";
  state.openVersion = "";
  state.savedText = "";
  if (editor) editor.close();
  showPreview(false);
  conflictBox.hidden = true;
  fileNote.textContent = "";
  refreshFileButtons();
}

export function renderFiles() {
  const files = state.allFiles;
  const cut = state.filesCut;
  const needle = fileFilter.value.trim().toLowerCase();
  const shown = needle ? files.filter((path) => path.toLowerCase().includes(needle)) : files;
  fileList.replaceChildren();
  for (const path of shown) {
    const item = document.createElement("li");
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = path;
    button.title = path;
    if (path === state.openPath) button.setAttribute("aria-current", "true");
    button.addEventListener("click", () => openFileAt(path));
    item.append(button);
    fileList.append(item);
  }
  if (!files.length) folderNote.textContent = "No files of a kind this page opens in " + state.folderName + ".";
  else if (needle) folderNote.textContent = shown.length + " of " + files.length + " files match.";
  else if (cut) folderNote.textContent = "Showing the first " + files.length + " files. This is meant for a small folder.";
  else folderNote.textContent = files.length + (files.length === 1 ? " file in " : " files in ") + state.folderName + ".";
}

// Returns true when the folder was opened.
export async function openFolder(path, quiet = false) {
  path = path.trim();
  if (!path) return false;
  return openSource(helperSource(path), quiet);
}

export async function openLocalFolder(handle, quiet = false) {
  return openSource(localSource(handle), quiet);
}

async function openSource(next, quiet = false) {
  if (!discardOk()) return false;
  folderNote.textContent = "Opening…";
  try {
    const body = await next.list();
    closeFile();
    // The helper tidies the path it was given (for example ~), so keep its answer.
    state.source = next.kind === "helper" ? helperSource(body.root) : next;
    state.folderName = body.name;
    state.allFiles = body.files;
    state.filesCut = body.cut;
    fileFilter.value = "";
    fileFilter.disabled = false;
    folderLabel.textContent = state.source.label;
    folderLabel.title = state.source.label;
    localStorage.setItem(KIND_KEY, state.source.kind);
    if (state.source.kind === "helper") localStorage.setItem(FOLDER_KEY, body.root);
    else rememberHandle(state.source.root);
    renderFiles();
    refreshFileButtons();
    return true;
  } catch (error) {
    if (!quiet) closeFile();
    state.source = null;
    refreshFileButtons();
    state.allFiles = [];
    fileFilter.disabled = true;
    folderLabel.textContent = "No folder open";
    fileList.replaceChildren();
    folderNote.textContent = plainError(error.message);
    return false;
  }
}

// The folder chooser: click through folders, filter them by typing, or
// type a whole path and press Enter. Only folder names are listed.
const folderDialog = document.querySelector("#folder-dialog");
const browseInput = document.querySelector("#browse-input");
const browseList = document.querySelector("#browse-list");
const browseNote = document.querySelector("#browse-note");
let browsing = null;
let browseToken = 0;
const looksLikePath = (text) => /^(\/|~|\.\.?\/)/.test(text.trim());

function renderBrowse() {
  browseList.replaceChildren();
  if (!browsing) return;
  const needle = looksLikePath(browseInput.value) ? "" : browseInput.value.trim().toLowerCase();
  const add = (label, path, className) => {
    const item = document.createElement("li");
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = label;
    if (className) button.className = className;
    button.addEventListener("click", () => browseTo(path));
    item.append(button);
    browseList.append(item);
  };
  if (browsing.parent && !needle) add("↑  Up one folder", browsing.parent, "up");
  const join = (name) => (browsing.path === "/" ? "" : browsing.path) + "/" + name;
  const shown = browsing.folders.filter((name) => name.toLowerCase().includes(needle));
  for (const name of shown) add("📁  " + name, join(name));
  const bits = [];
  if (!browsing.folders.length) bits.push("No folders in here.");
  else if (needle) bits.push(shown.length + " of " + browsing.folders.length + " folders match.");
  if (browsing.cut) bits.push("Only the first 1000 folders are listed. Type a path to go further.");
  bits.push(browsing.files === 1
    ? "1 file of a kind this page opens is directly in this folder."
    : browsing.files + " files of a kind this page opens are directly in this folder.");
  browseNote.textContent = bits.join(" ");
}

async function browseTo(path) {
  const token = ++browseToken;
  try {
    const body = await filesApi("/api/workspace/browse?path=" + encodeURIComponent(path));
    if (token !== browseToken) return;
    browsing = body;
    browseInput.value = "";
    document.querySelector("#browse-path").textContent = body.path;
    renderBrowse();
  } catch (error) {
    if (token !== browseToken) return;
    browseNote.textContent = plainError(error.message);
  }
}

export function showFolderDialog() {
  if (!discardOk()) return;
  if (!folderDialog.open) folderDialog.showModal();
  browseInput.value = "";
  browseTo(state.source && state.source.kind === "helper" ? state.source.root : "");
  showLocalOptions();
  browseInput.focus();
}
const pickLocal = document.querySelector("#pick-local");
const reopenLocal = document.querySelector("#reopen-local");
const localNote = document.querySelector("#local-note");
const canPickLocal = typeof window.showDirectoryPicker === "function";

async function showLocalOptions() {
  localNote.textContent = canPickLocal
    ? "Your browser won't let a page open your home folder, Documents, Desktop, Downloads, a whole drive, or system folders. Pick a folder inside one of them, such as a project folder."
    : "This browser has no folder picker for web pages. Use the list below.";
  pickLocal.disabled = !canPickLocal;
  reopenLocal.hidden = true;
  if (!canPickLocal) return;
  const handle = await recallHandle();
  if (!handle || (state.source && state.source.kind === "local" && (await state.source.root.isSameEntry(handle).catch(() => false)))) return;
  reopenLocal.textContent = "Reopen " + handle.name;
  reopenLocal.hidden = false;
}

async function useLocalHandle(getHandle) {
  let handle;
  try {
    handle = await getHandle();
  } catch (error) {
    if (error.name !== "AbortError") localNote.textContent = plainError(error.message);
    return;
  }
  if (!handle) return;
  if (await openLocalFolder(handle)) folderDialog.close();
  else localNote.textContent = folderNote.textContent;
}

// Listeners and first steps, run by main.js once every part is loaded.
export function start() {
  browseInput.addEventListener("input", renderBrowse);
  browseInput.addEventListener("keydown", (event) => {
    if (event.key !== "Enter") return;
    event.preventDefault();
    const typed = browseInput.value.trim();
    if (looksLikePath(typed)) browseTo(typed);
    else if (browseList.querySelectorAll("button:not(.up)").length === 1) browseList.querySelector("button:not(.up)").click();
  });
  document.querySelector("#browse-home").addEventListener("click", () => browseTo(""));
  document.querySelector("#browse-open").addEventListener("click", async () => {
    if (!browsing) return;
    if (await openFolder(browsing.path)) folderDialog.close();
    else browseNote.textContent = folderNote.textContent;
  });

  pickLocal.addEventListener("click", () => useLocalHandle(() => window.showDirectoryPicker({ id: "lightusagi", mode: "readwrite" })));
  reopenLocal.addEventListener("click", () => useLocalHandle(async () => {
    const handle = await recallHandle();
    if (!handle) return null;
    const permission = await handle.requestPermission({ mode: "readwrite" });
    if (permission !== "granted") throw new Error("The browser did not give this page access to " + handle.name + ".");
    return handle;
  }));
  document.querySelector("#close-folder").addEventListener("click", () => folderDialog.close());
  document.querySelector("#choose-folder").addEventListener("click", showFolderDialog);
  fileFilter.addEventListener("input", renderFiles);
}
