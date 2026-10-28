// Keyboard shortcuts for the whole page, and the new chat one the user can change.
import { isMac, modKeys } from "./core.js";
import { inCodeView } from "./models.js";
import { newChat } from "./chats.js";
import { setSidebar } from "./page.js";
import { setFilesHidden } from "./layout.js";
import { askAbout } from "./file-actions.js";

// Cmd (Mac) or Ctrl (elsewhere) with B, K or L is fixed:
//   B hides and shows the side list: the chats in Chat, the files in the code editor.
//   K adds the open file to the message, L its highlighted lines (code editor only).
// New chat is Ctrl+Option+N (Ctrl+Alt+N) unless the user picks another in Settings.
const NEW_CHAT_KEY = "lightusagi-new-chat-keys";
const DEFAULT_NEW_CHAT = { ctrl: true, alt: true, shift: false, meta: false, code: "KeyN" };
const keysButton = document.querySelector("#new-chat-keys");
const resetButton = document.querySelector("#reset-new-chat-keys");
const keysNote = document.querySelector("#keys-note");
let recording = false;

function newChatKeys() {
  try {
    const saved = JSON.parse(localStorage.getItem(NEW_CHAT_KEY) || "null");
    if (saved && typeof saved.code === "string") return saved;
  } catch (error) {
    // A broken saved value falls back to the default.
  }
  return DEFAULT_NEW_CHAT;
}

const keyName = (code) => code.replace(/^(Key|Digit|Numpad)/, "").replace(/^Arrow/, "") || code;
function keysLabel(keys) {
  const name = keyName(keys.code);
  if (isMac) {
    return (keys.ctrl ? "⌃" : "") + (keys.alt ? "⌥" : "") + (keys.shift ? "⇧" : "") + (keys.meta ? "⌘" : "") + name;
  }
  return [keys.ctrl && "Ctrl", keys.alt && "Alt", keys.shift && "Shift", keys.meta && "Win", name].filter(Boolean).join("+");
}

const matches = (keys, event) => event.code === keys.code && event.ctrlKey === keys.ctrl &&
  event.altKey === keys.alt && event.shiftKey === keys.shift && event.metaKey === keys.meta;

// With Cmd/Ctrl alone: new window, new tab, close tab and quit belong to the
// browser; the rest are the page's and the editor's (side list, add file, add
// lines, save, find, go to line, next match, undo, redo, copy, paste, cut,
// select all, comment).
const BROWSER_KEYS = ["n", "t", "w", "q"];
const TAKEN_KEYS = ["b", "k", "l", "s", "f", "g", "d", "z", "y", "c", "v", "x", "a", "slash"];

// Why a combination cannot be the new chat shortcut, or "" if it can.
function refusal(keys) {
  if (!keys.ctrl && !keys.alt && !keys.meta) return "Use Ctrl, " + (isMac ? "Option or Cmd" : "or Alt") + " with a key, so typing is not caught.";
  const modOnly = (isMac ? keys.meta && !keys.ctrl : keys.ctrl && !keys.meta) && !keys.alt;
  const key = keyName(keys.code).toLowerCase();
  if (modOnly && BROWSER_KEYS.includes(key)) return "The browser keeps " + keysLabel(keys) + " for itself.";
  if (modOnly && !keys.shift && TAKEN_KEYS.includes(key)) return keysLabel(keys) + " is already used here.";
  return "";
}

function showKeys() {
  const label = keysLabel(newChatKeys());
  keysButton.textContent = recording ? "Press keys…" : label;
  keysButton.setAttribute("aria-pressed", String(recording));
  keysButton.setAttribute("aria-label", recording ? "Press the new shortcut, or Esc to keep " + label : "New chat shortcut: " + label + ". Change");
  resetButton.disabled = recording || JSON.stringify(newChatKeys()) === JSON.stringify(DEFAULT_NEW_CHAT);
  document.querySelector("#new-chat").title = "New chat (" + label + ")";
  document.querySelector("#new-editor-chat").title = "New chat (" + label + ")";
}

function record(event) {
  if (!recording) return;
  event.preventDefault();
  event.stopImmediatePropagation();
  if (["Control", "Alt", "Shift", "Meta", "OS"].includes(event.key)) return;
  recording = false;
  window.removeEventListener("keydown", record, true);
  if (event.key === "Escape" && !event.ctrlKey && !event.altKey && !event.metaKey) {
    keysNote.textContent = "";
    showKeys();
    return;
  }
  const keys = { ctrl: event.ctrlKey, alt: event.altKey, shift: event.shiftKey, meta: event.metaKey, code: event.code };
  const why = refusal(keys);
  if (why) {
    keysNote.textContent = why + " The shortcut is still " + keysLabel(newChatKeys()) + ".";
  } else {
    localStorage.setItem(NEW_CHAT_KEY, JSON.stringify(keys));
    keysNote.textContent = "New chat is now " + keysLabel(keys) + ".";
  }
  showKeys();
}

function onKey(event) {
  if (recording || event.repeat || document.querySelector("dialog[open]")) return;
  if (matches(newChatKeys(), event)) {
    event.preventDefault();
    newChat();
    return;
  }
  const mod = isMac ? event.metaKey && !event.ctrlKey : event.ctrlKey && !event.metaKey;
  if (!mod || event.altKey || event.shiftKey) return;
  const key = event.key.toLowerCase();
  if (key === "b") {
    if (inCodeView()) setFilesHidden(!document.body.classList.contains("files-hidden"));
    else setSidebar(document.body.classList.contains("sidebar-closed"));
  } else if ((key === "k" || key === "l") && inCodeView()) {
    askAbout(key === "l");
  } else {
    return;
  }
  event.preventDefault();
}

// Listeners and first steps, run by main.js once every part is loaded.
export function start() {
  document.addEventListener("keydown", onKey, true);
  keysButton.addEventListener("click", () => {
    if (recording) return;
    recording = true;
    keysNote.textContent = "Press the keys for a new chat, or Esc to keep the old ones.";
    window.addEventListener("keydown", record, true);
    showKeys();
  });
  keysButton.addEventListener("blur", () => {
    if (!recording) return;
    recording = false;
    window.removeEventListener("keydown", record, true);
    keysNote.textContent = "";
    showKeys();
  });
  resetButton.addEventListener("click", () => {
    localStorage.removeItem(NEW_CHAT_KEY);
    keysNote.textContent = "New chat is back to " + keysLabel(DEFAULT_NEW_CHAT) + ".";
    showKeys();
  });
  document.querySelector("#keys-fixed").textContent = modKeys("B") + " hides or shows the chats, or the files in the code editor. " +
    "In the code editor, " + modKeys("K") + " adds the open file to the message and " + modKeys("L") +
    " adds the highlighted lines (or the line the cursor is on).";
  document.querySelector("#toggle-sidebar").title = "Hide or show chats (" + modKeys("B") + ")";
  document.querySelector("#disk-note").textContent = "Modifying this file will edit its contents on disk. " +
    modKeys("K") + " adds the file to the message, " + modKeys("L") + " the highlighted lines.";
  showKeys();
}
