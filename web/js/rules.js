// Your rules for the models (kept by the helper in ~/.hikarilm/rules.md), and in the code editor
// the open folder's own rules file. The helper puts both first in the system message.
import { state } from "./state.js";
import { plainError } from "./core.js";
import { openFileAt } from "./file-actions.js";

const FOLDER_FILES = ["HIKARILM.md", "AGENTS.md", "CLAUDE.md", ".cursorrules"];
const MAX_CHARS = 8000;
let mine = { text: "", on: true };

const hasMine = () => mine.on && mine.text.trim() !== "";
const folderFile = () => (state.source ? FOLDER_FILES.find((name) => state.allFiles.includes(name)) || "" : "");

// The folder's rules file, read again each time so a change to it counts at once.
async function readFolderRules() {
  const name = folderFile();
  if (!name) return null;
  try {
    const body = await state.source.read(name);
    return body.text.trim() ? { name, text: body.text.slice(0, MAX_CHARS) } : null;
  } catch (error) {
    return null;
  }
}

// The line under the folder in the code editor: which rules its chats get, and the file to open.
export function showFolderRules() {
  const line = document.querySelector("#folder-rules");
  const name = folderFile();
  const parts = [hasMine() && "yours", name].filter(Boolean);
  line.hidden = !state.source || !parts.length;
  if (line.hidden) return;
  line.replaceChildren("Rules: " + (hasMine() ? "yours" : ""));
  if (!name) return;
  if (hasMine()) line.append(" + ");
  const open = document.createElement("button");
  open.type = "button";
  open.className = "link-button";
  open.textContent = name;
  open.title = "Open " + name + ". Chats in the code editor follow it" + (name === "HIKARILM.md" ? "." : "; a HIKARILM.md would come first.");
  open.addEventListener("click", () => openFileAt(name));
  line.append(open);
}

// What a chat send asks for: rules on, unless this chat turned them off, and in the code
// editor the folder's file.
export async function rulesRequest(codeView) {
  const chat = state.chats.find((item) => item.id === state.chatId);
  return { on: chat?.rules !== false, folder: codeView ? await readFolderRules() : null };
}

// Rules that take a large part of the context leave little for the chat.
export function rulesWarning(event) {
  if (!event.num_ctx || event.tokens * 4 <= event.num_ctx) return "";
  return "Your rules take about " + event.tokens.toLocaleString() + " of the model's " + event.num_ctx.toLocaleString() +
    " tokens of context, so less of the chat fits.";
}

function showNote(text) {
  const count = document.querySelector("#rules-text").value.length;
  document.querySelector("#rules-note").textContent =
    (text ? text + " " : "") + count.toLocaleString() + " of " + MAX_CHARS.toLocaleString() + " characters.";
}

async function loadRules() {
  try {
    mine = await (await fetch("/api/rules")).json();
  } catch (error) {
    document.querySelector("#rules-note").textContent = "Could not reach the helper.";
    return;
  }
  document.querySelector("#rules-text").value = mine.text;
  document.querySelector("#rules-toggle").checked = mine.on;
  document.querySelector("#rules-text").disabled = false;
  showNote(mine.cut ? "rules.md is longer than the limit; only the first " + MAX_CHARS.toLocaleString() + " characters are used." : "");
  showFolderRules();
}

async function saveRules(fields) {
  try {
    const response = await fetch("/api/rules", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(fields),
    });
    const body = await response.json();
    if (!response.ok) throw new Error(body.error || "the helper refused it");
    mine = body;
    showNote(mine.on ? "Saved." : "Saved. Off: no chat gets them until you turn them on.");
  } catch (error) {
    showNote("Not saved: " + plainError(error.message) + ".");
  }
  showFolderRules();
}

// Listeners and first steps, run by main.js once every part is loaded.
export function start() {
  const text = document.querySelector("#rules-text");
  text.addEventListener("input", () => showNote(""));
  text.addEventListener("change", () => saveRules({ text: text.value }));
  document.querySelector("#rules-toggle").addEventListener("change", (event) => saveRules({ on: event.target.checked }));
  document.querySelector("#open-settings").addEventListener("click", loadRules);
  loadRules();
}
