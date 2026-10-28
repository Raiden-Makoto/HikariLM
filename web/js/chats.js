// Unsent drafts, and starting, opening and deleting a chat.
import { state } from "./state.js";
import { messageBox, modelSelect, status } from "./core.js";
import { renderChips } from "./attachments.js";
import { renderLog } from "./turns.js";
import { CODE_MODEL_KEY, inCodeView, updatePictureMode } from "./models.js";
import { closeSidebarIfNarrow, refreshChats, renderChats } from "./chat-list.js";

// Drafts. What is typed and attached but not sent is kept by the helper in
// ~/.lightusagi/drafts, one per chat ("new" before a chat has started), so
// a reload or a switch to another chat does not lose it. Requests go
// through one queue so they land in the order they were made.
const DRAFT_NEW = "new";
const LAST_CHAT = "lightusagi-chat";
let draftTimer = null;
let draftQueue = Promise.resolve();
// The code editor has its own conversations, apart from the Chat page's.
// Their ids start with "editor-", so they and their drafts never mix with
// the Chat page's, and the chat list leaves them out.
export const isEditorId = (id) => typeof id === "string" && id.startsWith("editor-");
export const lastChatKey = () => (inCodeView() ? "lightusagi-chat-editor" : LAST_CHAT);
export const draftKey = () => state.chatId || (inCodeView() ? "editor-new" : DRAFT_NEW);

export function saveDraft(key = draftKey(), withFiles = false, keepalive = false) {
  clearTimeout(draftTimer);
  draftTimer = null;
  const empty = !messageBox.value && state.attached.length === 0;
  const body = { text: messageBox.value };
  if (withFiles) {
    body.files = state.attached.map((file) =>
      file.image ? { name: file.name, image: file.image } : { name: file.name, text: file.text });
  }
  const request = () => empty
    ? fetch("/api/drafts/" + key, { method: "DELETE", keepalive })
    : fetch("/api/drafts/" + key, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
        keepalive,
      });
  if (keepalive) request().catch(() => {});
  else draftQueue = draftQueue.then(request).catch(() => {});
}

export function scheduleDraft() {
  clearTimeout(draftTimer);
  draftTimer = setTimeout(() => saveDraft(), 400);
}

export function dropDraft(key) {
  clearTimeout(draftTimer);
  draftTimer = null;
  draftQueue = draftQueue.then(() => fetch("/api/drafts/" + key, { method: "DELETE" })).catch(() => {});
}

export function flushDraft(keepalive = false) {
  if (draftTimer !== null) saveDraft(draftKey(), false, keepalive);
}

export function clearComposer() {
  messageBox.value = "";
  messageBox.style.height = "";
  state.attached = [];
  renderChips();
}

export async function loadDraft(key) {
  let draft;
  try {
    await draftQueue;
    const response = await fetch("/api/drafts/" + key);
    if (!response.ok) return;
    draft = await response.json();
  } catch (error) {
    return;
  }
  // Ignore the answer if the user moved on, or has already started typing.
  if (draftKey() !== key || messageBox.value || state.attached.length) return;
  messageBox.value = draft.text || "";
  state.attached = draft.files || [];
  renderChips();
  messageBox.style.height = "auto";
  messageBox.style.height = Math.min(messageBox.scrollHeight, 160) + "px";
}

export function newChat() {
  if (state.activeRequest) return;
  flushDraft();
  state.chatId = null;
  state.messages = [];
  localStorage.removeItem(lastChatKey());
  clearComposer();
  renderLog();
  renderChats();
  closeSidebarIfNarrow();
  messageBox.focus();
  loadDraft(draftKey());
}

export async function openChat(id) {
  if (state.activeRequest || id === state.chatId) {
    closeSidebarIfNarrow();
    return;
  }
  let chat;
  try {
    const response = await fetch("/api/chats/" + id);
    chat = await response.json();
    if (!response.ok) throw new Error(chat.error || "could not open chat");
  } catch (error) {
    status.textContent = "Could not open that chat: " + error.message;
    refreshChats();
    return;
  }
  flushDraft();
  state.chatId = id;
  state.messages = chat.messages || [];
  localStorage.setItem(lastChatKey(), id);
  clearComposer();
  loadDraft(id);
  if ([...modelSelect.options].some((option) => option.value === chat.model)) {
    modelSelect.value = chat.model;
    localStorage.setItem(inCodeView() ? CODE_MODEL_KEY : "lightusagi-model", chat.model);
    updatePictureMode();
  }
  renderLog();
  renderChats();
  closeSidebarIfNarrow();
  messageBox.focus();
}

export async function deleteChat(id) {
  try {
    await fetch("/api/chats/" + id, { method: "DELETE" });
  } catch (error) {
    status.textContent = "Could not delete that chat.";
  }
  if (id === state.chatId) {
    state.chatId = null;
    state.messages = [];
    localStorage.removeItem(lastChatKey());
    clearComposer();
    renderLog();
    loadDraft(draftKey());
  }
  refreshChats();
}
