// The code editor's layout: hiding the file list, and the chat pane's width.
import { modKeys } from "./core.js";

// Ctrl+B (Cmd+B on a Mac) hides and shows the file list, as in Cursor.
const FILES_HIDDEN_KEY = "hikarilm-files-hidden";
const toggleFiles = document.querySelector("#toggle-files");

export function setFilesHidden(hidden) {
  document.body.classList.toggle("files-hidden", hidden);
  toggleFiles.setAttribute("aria-expanded", String(!hidden));
  toggleFiles.setAttribute("aria-label", hidden ? "Show files" : "Hide files");
  toggleFiles.title = (hidden ? "Show files (" : "Hide files (") + modKeys("B") + ")";
  localStorage.setItem(FILES_HIDDEN_KEY, hidden ? "1" : "0");
}

// The chat pane's left edge can be dragged (or moved with the arrow keys
// while it has focus). Double-click puts it back. The width is remembered.
const PANE_KEY = "hikarilm-pane";
const paneResize = document.querySelector("#pane-resize");
const chatPane = document.querySelector("#chat-pane");
const clampPane = (width) => Math.max(320, Math.min(width, window.innerWidth * 0.75, window.innerWidth - 360));

function setPane(width, keep = true) {
  if (width === null) {
    document.documentElement.style.removeProperty("--pane-w");
    localStorage.removeItem(PANE_KEY);
    return;
  }
  const value = Math.round(clampPane(width));
  document.documentElement.style.setProperty("--pane-w", value + "px");
  if (keep) localStorage.setItem(PANE_KEY, String(value));
}
const endResize = (event) => {
  if (!paneResize.hasPointerCapture(event.pointerId)) return;
  paneResize.releasePointerCapture(event.pointerId);
  document.body.classList.remove("resizing");
  setPane(chatPane.getBoundingClientRect().width);
};
const savedPane = Number(localStorage.getItem(PANE_KEY));

// Listeners and first steps, run by main.js once every part is loaded.
export function start() {
  toggleFiles.addEventListener("click", () => setFilesHidden(!document.body.classList.contains("files-hidden")));
  setFilesHidden(localStorage.getItem(FILES_HIDDEN_KEY) === "1");
  paneResize.addEventListener("pointerdown", (event) => {
    event.preventDefault();
    paneResize.setPointerCapture(event.pointerId);
    document.body.classList.add("resizing");
  });
  paneResize.addEventListener("pointermove", (event) => {
    if (!paneResize.hasPointerCapture(event.pointerId)) return;
    setPane(chatPane.parentElement.getBoundingClientRect().right - event.clientX, false);
  });
  paneResize.addEventListener("pointerup", endResize);
  paneResize.addEventListener("pointercancel", endResize);
  paneResize.addEventListener("dblclick", () => setPane(null));
  paneResize.addEventListener("keydown", (event) => {
    const step = event.shiftKey ? 80 : 24;
    const now = chatPane.getBoundingClientRect().width;
    if (event.key === "ArrowLeft") setPane(now + step);
    else if (event.key === "ArrowRight") setPane(now - step);
    else if (event.key === "Home" || event.key === "Escape") setPane(null);
    else return;
    event.preventDefault();
  });
  if (savedPane > 0) setPane(savedPane, false);
}
