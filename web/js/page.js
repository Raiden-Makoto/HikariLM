// Dropping files on the page, the sidebar, notifications, and the theme and background.
import { state } from "./state.js";
import { messageBox, modelSelect, narrow } from "./core.js";
import { addFiles, attachButton, fileInput } from "./attachments.js";
import { CODE_MODEL_KEY, inCodeView, loadModels, SETUP_PICTURES, updatePictureMode } from "./models.js";
import { isEditorId, newChat } from "./chats.js";
import { openPictureSetup } from "./settings.js";
import { openChatAnywhere } from "./sections.js";

let dragDepth = 0;
const hasFiles = (event) => [...(event.dataTransfer?.types || [])].includes("Files");

export function setSidebar(open) {
  document.body.classList.toggle("sidebar-closed", !open);
  const toggle = document.querySelector("#toggle-sidebar");
  toggle.setAttribute("aria-expanded", String(open));
  toggle.setAttribute("aria-label", open ? "Hide chats" : "Show chats");
  if (!narrow.matches) localStorage.setItem("hikarilm-sidebar", open ? "open" : "closed");
}

// Notifications. Only when you are away from the page (another tab or
// window): the tab title changes, and the browser shows a notification if
// it was allowed. Clicking the notification brings you back to that chat.
const NOTIFY_KEY = "hikarilm-notify";
const BASE_TITLE = "HikariLM";
const notifyNote = document.querySelector("#notify-note");
const canNotify = "Notification" in window;
const notifyOn = () => localStorage.getItem(NOTIFY_KEY) !== "off";
const userIsAway = () => document.hidden || !document.hasFocus();

function updateNotifySettings() {
  document.querySelector("#notify-toggle").checked = notifyOn();
  if (!notifyOn()) notifyNote.textContent = "";
  else if (!canNotify) notifyNote.textContent = "This browser cannot show notifications here. The tab title will change instead.";
  else if (Notification.permission === "denied") notifyNote.textContent = "Notifications are blocked for this site in the browser settings. The tab title will still change.";
  else if (Notification.permission === "default") notifyNote.textContent = "The browser will ask for permission the first time.";
  else notifyNote.textContent = "You will be told when a reply finishes while you are in another tab or window.";
}

export function askNotifyPermission() {
  if (!notifyOn() || !canNotify || Notification.permission !== "default") return;
  Promise.resolve(Notification.requestPermission()).then(updateNotifySettings, () => {});
}

export function announce(heading, chatName, detail, id) {
  if (!notifyOn() || !userIsAway()) return;
  document.title = "\u25cf " + heading + " \u00b7 " + BASE_TITLE;
  if (!canNotify || Notification.permission !== "granted") return;
  try {
    const note = new Notification(heading, {
      body: (chatName ? chatName + "\n" : "") + detail.replace(/\s+/g, " ").trim().slice(0, 120),
      tag: "hikarilm-reply",
    });
    note.onclick = () => {
      window.focus();
      note.close();
      if (id && (id !== state.chatId || isEditorId(id) !== inCodeView())) openChatAnywhere(id);
    };
    setTimeout(() => note.close(), 10000);
  } catch (error) {
    // Some browsers only allow notifications from a service worker; the title still changed.
  }
}

function clearAnnouncement() {
  if (!userIsAway()) document.title = BASE_TITLE;
}

export function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  localStorage.setItem("hikarilm-theme", theme);
  document.querySelector("#dark-toggle").checked = theme === "dark";
}

export function applyBackground(url) {
  const veil = "color-mix(in srgb, var(--bg) 88%, transparent)";
  document.body.style.backgroundImage = url
    ? "linear-gradient(" + veil + ", " + veil + "), url(\"" + url.replaceAll("\"", "%22") + "\")"
    : "";
  document.body.classList.toggle("has-bg", Boolean(url));
  document.querySelector("#clear-bg").disabled = !url;
}

// Listeners and first steps, run by main.js once every part is loaded.
export function start() {
  window.addEventListener("dragenter", (event) => {
    if (!hasFiles(event) || state.activeRequest) return;
    dragDepth += 1;
    document.body.classList.add("dragging");
  });
  window.addEventListener("dragleave", (event) => {
    if (!hasFiles(event)) return;
    dragDepth = Math.max(0, dragDepth - 1);
    if (dragDepth === 0) document.body.classList.remove("dragging");
  });
  window.addEventListener("dragover", (event) => {
    if (hasFiles(event)) event.preventDefault();
  });
  window.addEventListener("drop", async (event) => {
    if (!hasFiles(event)) return;
    event.preventDefault();
    dragDepth = 0;
    document.body.classList.remove("dragging");
    if (state.activeRequest) return;
    await addFiles([...event.dataTransfer.files]);
    messageBox.focus();
  });
  attachButton.addEventListener("click", () => fileInput.click());
  fileInput.addEventListener("change", async () => {
    const picked = [...fileInput.files];
    fileInput.value = "";
    await addFiles(picked);
    messageBox.focus();
  });
  modelSelect.addEventListener("change", () => {
    if (modelSelect.value === SETUP_PICTURES) {
      const saved = localStorage.getItem(inCodeView() ? CODE_MODEL_KEY : "hikarilm-model");
      modelSelect.value = [...modelSelect.options].some((option) => option.value === saved) ? saved : modelSelect.options[0].value;
      openPictureSetup();
      return;
    }
    localStorage.setItem(inCodeView() ? CODE_MODEL_KEY : "hikarilm-model", modelSelect.value);
    updatePictureMode();
  });
  window.addEventListener("focus", () => {
    if (!state.activeRequest && !modelSelect.value) loadModels();
  });
  document.querySelector("#new-chat").addEventListener("click", newChat);
  document.querySelector("#new-editor-chat").addEventListener("click", newChat);
  document.querySelector("#toggle-sidebar").addEventListener("click", () => {
    setSidebar(document.body.classList.contains("sidebar-closed"));
  });
  document.querySelector(".main").addEventListener("click", (event) => {
    if (event.target.closest("#toggle-sidebar")) return;
    if (narrow.matches && !document.body.classList.contains("sidebar-closed")) setSidebar(false);
  });

  document.querySelector("#notify-toggle").addEventListener("change", (event) => {
    localStorage.setItem(NOTIFY_KEY, event.target.checked ? "on" : "off");
    updateNotifySettings();
    askNotifyPermission();
  });
  window.addEventListener("focus", clearAnnouncement);
  document.addEventListener("visibilitychange", clearAnnouncement);
  updateNotifySettings();

  document.querySelector("#dark-toggle").addEventListener("change", (event) => {
    applyTheme(event.target.checked ? "dark" : "light");
  });
}
