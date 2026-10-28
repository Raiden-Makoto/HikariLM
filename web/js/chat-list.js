// The chat list in the sidebar: folders, pins, the options menu, renaming, and saving chats.
import { state } from "./state.js";
import { chatList, modelSelect, narrow, status } from "./core.js";
import { deleteChat, lastChatKey } from "./chats.js";
import { setSidebar } from "./page.js";
import { openChatAnywhere } from "./sections.js";

export function renderChats() {
  chatList.replaceChildren();
  if (state.chats.length === 0) {
    const none = document.createElement("li");
    none.className = "side-empty";
    none.textContent = "No chats yet.";
    chatList.append(none);
    return;
  }
  // Every chat shows in one place. Pinned chats come first, then folders
  // (the folder with the newest chat first), then the rest. Inside each
  // group the most recently used chat is on top, which is the order the
  // helper already returns.
  const pinned = state.chats.filter((chat) => chat.pinned);
  const unpinned = state.chats.filter((chat) => !chat.pinned);
  const folders = new Map();
  for (const chat of unpinned) {
    if (!chat.folder) continue;
    if (!folders.has(chat.folder)) folders.set(chat.folder, []);
    folders.get(chat.folder).push(chat);
  }
  const loose = unpinned.filter((chat) => !chat.folder);
  const label = (text) => {
    const item = document.createElement("li");
    item.className = "side-group";
    item.textContent = text;
    chatList.append(item);
  };
  if (pinned.length) {
    label("Pinned");
    for (const chat of pinned) chatList.append(chatRow(chat, true));
  }
  for (const [name, inside] of folders) {
    const open = !foldersClosed.has(name) || inside.some((chat) => chat.id === state.chatId);
    const item = document.createElement("li");
    const head = document.createElement("button");
    head.type = "button";
    head.className = "folder-head";
    head.setAttribute("aria-expanded", String(open));
    head.innerHTML = '<svg class="chevron" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M6 9l6 6 6-6"/></svg><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round" aria-hidden="true"><path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/></svg>';
    const text = document.createElement("span");
    text.className = "folder-name";
    text.textContent = name;
    text.title = name;
    const count = document.createElement("span");
    count.className = "count";
    count.textContent = String(inside.length);
    head.append(text, count);
    head.addEventListener("click", () => {
      if (foldersClosed.has(name)) foldersClosed.delete(name);
      else foldersClosed.add(name);
      localStorage.setItem(FOLDERS_CLOSED, JSON.stringify([...foldersClosed]));
      renderChats();
    });
    const headRow = document.createElement("div");
    headRow.className = "folder-row";
    const more = document.createElement("button");
    more.className = "chat-more";
    more.type = "button";
    more.setAttribute("aria-label", "Options for folder " + name);
    more.setAttribute("aria-haspopup", "menu");
    more.setAttribute("aria-expanded", "false");
    more.innerHTML = '<svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><circle cx="5" cy="12" r="1.7"/><circle cx="12" cy="12" r="1.7"/><circle cx="19" cy="12" r="1.7"/></svg>';
    more.addEventListener("click", (event) => {
      event.stopPropagation();
      if (menuFor === "folder:" + name) closeMenu(true);
      else openFolderMenu(name, headRow, more);
    });
    headRow.append(head, more);
    item.append(headRow);
    if (open) {
      const list = document.createElement("ul");
      list.className = "folder-chats";
      for (const chat of inside) list.append(chatRow(chat, false));
      item.append(list);
    }
    chatList.append(item);
  }
  if (loose.length) {
    if (pinned.length || folders.size) label("Chats");
    for (const chat of loose) chatList.append(chatRow(chat, false));
  }
}

const FOLDERS_CLOSED = "lightusagi-folders-closed";
const foldersClosed = new Set(JSON.parse(localStorage.getItem(FOLDERS_CLOSED) || "[]"));
const PIN_ICON = '<svg class="pin-mark" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M14 3l7 7-2 1-3 3 .5 5-2 .5-3.5-3.5L6 21l-1-1 5.5-5.5L7 11l.5-2 5 .5 3-3z"/></svg>';

function chatRow(chat, showFolder) {
  const row = document.createElement("li");
  row.className = "chat-row" + (chat.id === state.chatId ? " active" : "");
  const open = document.createElement("button");
  open.className = "chat-open";
  open.type = "button";
  open.title = chat.title;
  if (chat.pinned) open.insertAdjacentHTML("beforeend", PIN_ICON);
  open.append(chat.title);
  if (showFolder && chat.folder) {
    const where = document.createElement("span");
    where.className = "in-folder";
    where.textContent = chat.folder;
    open.append(where);
  }
  if (chat.id === state.chatId) open.setAttribute("aria-current", "true");
  open.addEventListener("click", () => openChatAnywhere(chat.id));
  open.addEventListener("dblclick", () => startRename(chat, row));
  const more = document.createElement("button");
  more.className = "chat-more";
  more.type = "button";
  more.setAttribute("aria-label", "Options for " + chat.title);
  more.setAttribute("aria-haspopup", "menu");
  more.setAttribute("aria-expanded", "false");
  more.innerHTML = '<svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><circle cx="5" cy="12" r="1.7"/><circle cx="12" cy="12" r="1.7"/><circle cx="19" cy="12" r="1.7"/></svg>';
  more.addEventListener("click", (event) => {
    event.stopPropagation();
    if (menuFor === chat.id) closeMenu(true);
    else openMenu(chat, row, more);
  });
  row.append(open, more);
  return row;
}

// The options menu: pin, rename, move to a folder, delete.
const menu = document.querySelector("#chat-menu");
let menuFor = null;
let menuButton = null;

export function closeMenu(returnFocus = false) {
  menu.hidden = true;
  menu.replaceChildren();
  menuButton?.setAttribute("aria-expanded", "false");
  if (returnFocus) menuButton?.focus();
  menuFor = null;
  menuButton = null;
}

function menuItem(text, onClick, className = "") {
  const item = document.createElement("button");
  item.type = "button";
  item.setAttribute("role", "menuitem");
  item.textContent = text;
  if (className) item.className = className;
  item.addEventListener("click", onClick);
  return item;
}

function placeMenu(button) {
  const box = button.getBoundingClientRect();
  menu.hidden = false;
  const width = menu.offsetWidth;
  const height = menu.offsetHeight;
  menu.style.left = Math.max(8, Math.min(box.right - width, window.innerWidth - width - 8)) + "px";
  const below = box.bottom + 4;
  menu.style.top = (below + height > window.innerHeight - 8 ? Math.max(8, box.top - height - 4) : below) + "px";
}

function openMenu(chat, row, button) {
  if (state.activeRequest) return;
  closeMenu();
  menuFor = chat.id;
  menuButton = button;
  button.setAttribute("aria-expanded", "true");
  showMainMenu(chat, row);
  placeMenu(button);
  menu.querySelector("button")?.focus();
}

function showMainMenu(chat, row) {
  menu.replaceChildren(
    menuItem(chat.pinned ? "Unpin" : "Pin to top", () => changeChat(chat.id, { pinned: !chat.pinned })),
    menuItem("Rename", () => {
      const target = row;
      closeMenu();
      startRename(chat, target);
    }),
    menuItem(chat.folder ? "Move to another folder…" : "Move to folder…", () => {
      showFolderMenu(chat, row);
      placeMenu(menuButton);
    }),
    document.createElement("hr"),
  );
  let armed = false;
  let timer = null;
  const remove = menuItem("Delete", () => {
    if (!armed) {
      armed = true;
      remove.textContent = "Click again to delete";
      timer = setTimeout(() => {
        armed = false;
        remove.textContent = "Delete";
      }, 3000);
      return;
    }
    clearTimeout(timer);
    closeMenu();
    deleteChat(chat.id);
  }, "danger");
  menu.append(remove);
}

function showFolderMenu(chat, row) {
  const names = [...new Set(state.chats.map((item) => item.folder).filter(Boolean))].sort((a, b) => a.localeCompare(b));
  const back = menuItem("‹ Back", () => {
    showMainMenu(chat, row);
    placeMenu(menuButton);
  });
  const input = document.createElement("input");
  input.type = "text";
  input.maxLength = 40;
  input.placeholder = "New folder name";
  input.setAttribute("aria-label", "New folder name");
  input.addEventListener("keydown", (event) => {
    event.stopPropagation();
    if (event.key === "Enter" && input.value.trim()) changeChat(chat.id, { folder: input.value.trim() });
    else if (event.key === "Escape") closeMenu(true);
  });
  const items = [back, document.createElement("hr")];
  if (chat.folder) items.push(menuItem("No folder", () => changeChat(chat.id, { folder: null })));
  for (const name of names) {
    if (name === chat.folder) continue;
    items.push(menuItem(name, () => changeChat(chat.id, { folder: name })));
  }
  menu.replaceChildren(...items, input);
}

function openFolderMenu(name, headRow, button) {
  if (state.activeRequest) return;
  closeMenu();
  menuFor = "folder:" + name;
  menuButton = button;
  button.setAttribute("aria-expanded", "true");
  const inside = state.chats.filter((chat) => chat.folder === name).length;
  let armed = false;
  let timer = null;
  const remove = menuItem("Delete folder", () => {
    if (!armed) {
      armed = true;
      remove.textContent = "Click again. " + (inside === 1 ? "The chat stays." : "All " + inside + " chats stay.");
      timer = setTimeout(() => {
        armed = false;
        remove.textContent = "Delete folder";
      }, 4000);
      return;
    }
    clearTimeout(timer);
    changeFolder(name, null);
  }, "danger");
  menu.replaceChildren(
    menuItem("Rename folder", () => {
      closeMenu();
      startFolderRename(name, headRow);
    }),
    remove,
    Object.assign(document.createElement("p"), {
      className: "menu-note",
      textContent: "Deleting a folder keeps its " + (inside === 1 ? "chat" : "chats") + ".",
    }),
  );
  placeMenu(button);
  menu.querySelector("button")?.focus();
}

function startFolderRename(name, headRow) {
  const input = document.createElement("input");
  input.className = "chat-rename-input";
  input.type = "text";
  input.value = name;
  input.maxLength = 40;
  input.setAttribute("aria-label", "Folder name");
  headRow.replaceChildren(input);
  input.focus();
  input.select();
  let done = false;
  const finish = (commit) => {
    if (done) return;
    done = true;
    const next = input.value.trim();
    if (commit && next && next !== name) changeFolder(name, next);
    else renderChats();
  };
  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter") finish(true);
    else if (event.key === "Escape") finish(false);
  });
  input.addEventListener("blur", () => finish(true));
}

async function changeFolder(from, to) {
  closeMenu();
  const count = state.chats.filter((chat) => chat.folder === from).length;
  try {
    const response = await fetch("/api/folders", {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ from, to }),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "change failed");
    foldersClosed.delete(from);
    localStorage.setItem(FOLDERS_CLOSED, JSON.stringify([...foldersClosed]));
    const plural = (n) => n + (n === 1 ? " chat" : " chats");
    if (!to) status.textContent = "Folder \u201c" + from + "\u201d deleted. " + plural(result.changed) + " kept, now outside any folder.";
    else if (state.chats.some((chat) => chat.folder === result.to && chat.folder !== from)) status.textContent = "Merged \u201c" + from + "\u201d into \u201c" + result.to + "\u201d.";
    else status.textContent = "Folder renamed to \u201c" + result.to + "\u201d.";
    if (result.failed) status.textContent += " " + plural(result.failed) + " could not be updated.";
    if (count === 0 && !result.changed) status.textContent = "That folder was already gone.";
  } catch (error) {
    status.textContent = "Could not change that folder: " + error.message;
  }
  refreshChats();
}

async function changeChat(id, fields) {
  closeMenu();
  try {
    const response = await fetch("/api/chats/" + id, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(fields),
    });
    if (!response.ok) throw new Error((await response.json()).error || "change failed");
  } catch (error) {
    status.textContent = "Could not change that chat: " + error.message;
  }
  refreshChats();
}

function startRename(chat, row) {
  const input = document.createElement("input");
  input.className = "chat-rename-input";
  input.type = "text";
  input.value = chat.title;
  input.maxLength = 120;
  input.setAttribute("aria-label", "Chat name");
  row.replaceChildren(input);
  input.focus();
  input.select();
  let done = false;
  const finish = async (commit) => {
    if (done) return;
    done = true;
    const title = input.value.trim();
    if (commit && title && title !== chat.title) {
      try {
        const response = await fetch("/api/chats/" + chat.id, {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ title }),
        });
        if (!response.ok) throw new Error((await response.json()).error || "rename failed");
      } catch (error) {
        status.textContent = "Could not rename that chat: " + error.message;
      }
    }
    refreshChats();
  };
  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter") finish(true);
    else if (event.key === "Escape") finish(false);
  });
  input.addEventListener("blur", () => finish(true));
}

export async function refreshChats() {
  try {
    const response = await fetch("/api/chats");
    const body = await response.json();
    state.chats = body.chats || [];
  } catch (error) {
    state.chats = [];
  }
  renderChats();
}

export async function saveChat() {
  try {
    const response = await fetch("/api/chats/" + state.chatId, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: modelSelect.value,
        messages: state.messages.map((message) => ({
          ...message,
          files: message.files?.map((file) =>
            file.image ? { name: file.name, kind: "image" } : file),
        })),
      }),
    });
    if (!response.ok) throw new Error((await response.json()).error || "save failed");
  } catch (error) {
    status.textContent = "This chat could not be saved: " + error.message;
    return;
  }
  localStorage.setItem(lastChatKey(), state.chatId);
  await refreshChats();
}

export function closeSidebarIfNarrow() {
  if (narrow.matches) setSidebar(false);
}

// Listeners and first steps, run by main.js once every part is loaded.
export function start() {
  document.addEventListener("click", (event) => {
    // A menu item may replace itself while handling this click, so ask the
    // event path rather than the (by now detached) target.
    if (!menu.hidden && !event.composedPath().includes(menu)) closeMenu();
  });
  document.addEventListener("keydown", (event) => {
    if (menu.hidden) return;
    if (event.key === "Escape") closeMenu(true);
    else if (event.key === "ArrowDown" || event.key === "ArrowUp") {
      const items = [...menu.querySelectorAll("button")];
      const at = items.indexOf(document.activeElement);
      const next = event.key === "ArrowDown" ? (at + 1) % items.length : (at - 1 + items.length) % items.length;
      items[next]?.focus();
      event.preventDefault();
    }
  });
  window.addEventListener("resize", () => closeMenu());
  chatList.addEventListener("scroll", () => closeMenu());
}
