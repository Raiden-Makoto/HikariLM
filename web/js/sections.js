// The two sections, Chat and Code editor: switching between them, their
// addresses, and which chat each one shows.
import { state } from "./state.js";
import { messageBox, welcomeHint } from "./core.js";
import { attachButton, fileInput } from "./attachments.js";
import { notify, TEXT_EXTENSIONS } from "./text-files.js";
import { renderLog } from "./turns.js";
import { fillModels, inCodeView } from "./models.js";
import { closeMenu, renderChats } from "./chat-list.js";
import { clearComposer, draftKey, isEditorId, lastChatKey, loadDraft, openChat, saveDraft } from "./chats.js";
import { discardOk, fileNote, loadEditor, refreshFileButtons, reopenLast } from "./files.js";
import { openFileAt } from "./file-actions.js";

// Each section has its own address, so a reload or a bookmark comes back
// to the same section: /chat and /editor. Returns false if the user chose
// to stay because of unsaved edits.
const SECTION_PATHS = { chat: "/chat", files: "/editor" };
const SECTION_KEY = "lightusagi-section";
export const FILE_KEY = "lightusagi-file";
const sectionFromPath = () => (/^\/editor\/?$/.test(location.pathname) ? "files" : "chat");
// A plain "/" has no section in it, so it opens the one used last.
export const sectionToOpen = () =>
  (/^\/(chat|editor)\/?$/.test(location.pathname) ? sectionFromPath()
    : localStorage.getItem(SECTION_KEY) === "files" ? "files" : "chat");
const sectionShown = () => (inCodeView() ? "files" : "chat");

// After a reload, put the file that was open back in the editor.
export function restoreFile() {
  let saved;
  try {
    saved = JSON.parse(localStorage.getItem(FILE_KEY) || "null");
  } catch (error) {
    return;
  }
  if (!saved || !state.source || saved.kind !== state.source.kind || state.openPath) return;
  const folder = state.source.kind === "helper" ? state.source.root : state.source.root.name;
  if (saved.folder === folder && state.allFiles.includes(saved.path)) openFileAt(saved.path, true);
}

let shownSection = null;
const parked = {};

// Open a chat from anywhere. A chat made in the code editor opens in the
// code editor, at /editor, and any other chat opens in Chat.
export async function openChatAnywhere(id) {
  if (isEditorId(id) !== inCodeView() && !setSection(isEditorId(id) ? "files" : "chat", "push", false)) return;
  await openChat(id);
}

export function setSection(name, how = "push", restore = true) {
  const files = name === "files";
  if (!files && !discardOk()) return false;
  if (how !== "url" && location.pathname !== SECTION_PATHS[name]) {
    history[how === "replace" ? "replaceState" : "pushState"](null, "", SECTION_PATHS[name]);
  }
  // Chat and Code editor each have their own conversation and unsent
  // text. Keep this one's, then show the other's.
  const swap = shownSection !== null && shownSection !== name;
  if (swap && state.activeRequest) {
    notify("Wait for the reply to finish, or stop it, before switching.");
    return false;
  }
  if (swap) {
    saveDraft(draftKey(), true);
    parked[shownSection] = { chatId: state.chatId, messages: state.messages };
  }
  document.body.classList.toggle("view-files", files);
  shownSection = name;
  if (swap) {
    const back = parked[name] || { chatId: null, messages: [] };
    state.chatId = back.chatId;
    state.messages = back.messages;
    clearComposer();
    renderLog();
    renderChats();
    if (state.chatId) loadDraft(draftKey());
    else if (restore && !parked[name] && localStorage.getItem(lastChatKey())) openChat(localStorage.getItem(lastChatKey()));
    else if (restore) loadDraft(draftKey());
  }
  const emptyHint = document.querySelector("#empty .hint");
  if (emptyHint) emptyHint.textContent = welcomeHint();
  localStorage.setItem(SECTION_KEY, name);
  fileInput.accept = files ? [...TEXT_EXTENSIONS].map((ext) => "." + ext).join(",") : "";
  attachButton.title = files ? "Attach text or code files" : "Attach text, code, PDF, Word, Excel or picture files";
  if (!state.activeRequest) fillModels();
  document.querySelector("#tab-chat").setAttribute("aria-selected", String(!files));
  document.querySelector("#tab-files").setAttribute("aria-selected", String(files));
  if (files) {
    closeMenu();
    loadEditor().catch((error) => { fileNote.textContent = error.message; });
    if (!state.filesStarted) {
      state.filesStarted = true;
      reopenLast();
    }
    refreshFileButtons();
  } else {
    messageBox.focus();
  }
  return true;
}

// Listeners and first steps, run by main.js once every part is loaded.
export function start() {
  document.querySelector("#tab-chat").addEventListener("click", () => setSection("chat"));
  document.querySelector("#tab-files").addEventListener("click", () => setSection("files"));
  window.addEventListener("popstate", () => {
    const want = sectionFromPath();
    const now = sectionShown();
    if (want === now) return;
    // Back or forward with unsaved edits: asked first, and refusing stays put.
    if (!setSection(want, "url")) history.pushState(null, "", SECTION_PATHS[now]);
  });
}
