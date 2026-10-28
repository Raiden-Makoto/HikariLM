// The page's main elements, the error wording, and the empty chat's greeting.
export const modelSelect = document.querySelector("#model");
export const messageBox = document.querySelector("#message");
export const sendButton = document.querySelector("#send");
export const status = document.querySelector("#status");
export const log = document.querySelector("#log");
export const sidebar = document.querySelector("#sidebar");
export const chatList = document.querySelector("#chats");
export const settings = document.querySelector("#settings");
export const bgNote = document.querySelector("#bg-note");
export const narrow = window.matchMedia("(max-width: 720px)");
export const isMac = /mac|iphone|ipad/i.test(navigator.platform || "");
// How a Cmd (Mac) or Ctrl (elsewhere) shortcut is written on the page.
export const modKeys = (key) => (isMac ? "⌘" : "Ctrl+") + key;

export function plainError(message) {
  if (message.includes("not reachable")) return "Ollama is not running.";
  // What Chrome, Safari and Firefox say when the helper closed the connection or is not running.
  if (/^(Failed to fetch|Load failed|NetworkError|network error)/i.test(message)) {
    return "The connection to the LightUsagi helper was lost. If its window is closed, start it again; " +
      "if it is open, the last lines there say what went wrong.";
  }
  return message;
}

export function scrollLog() {
  log.scrollTop = log.scrollHeight;
}

export function showEmpty() {
  const empty = document.createElement("p");
  empty.className = "empty";
  empty.id = "empty";
  const title = document.createElement("strong");
  title.id = "welcome-title";
  title.textContent = welcomeLine();
  const hint = document.createElement("span");
  hint.className = "hint";
  hint.textContent = welcomeHint();
  const keys = document.createElement("span");
  keys.className = "keys";
  keys.textContent = "Enter sends. Shift+Enter starts a new line.";
  empty.append(title, hint, keys);
  log.replaceChildren(empty);
}

// By the clock on this computer: morning 5:00-11:59, afternoon 12:00-16:59,
// evening 17:00-23:59, and "Late one" in the small hours (0:00-4:59).
function greeting(now = new Date()) {
  const hour = now.getHours();
  if (hour >= 5 && hour < 12) return "Good morning";
  if (hour >= 12 && hour < 17) return "Good afternoon";
  if (hour >= 17) return "Good evening";
  return "Late one";
}

// The code editor takes text and code only, so it does not offer PDFs or pictures.
export function welcomeHint() {
  return document.body.classList.contains("view-files")
    ? "Ask about the open file: " + modKeys("K") + " adds it to the message, " + modKeys("L") +
      " adds the highlighted lines. Or paste some code, or attach a text or code file."
    : "Ask a question, paste some code, or drop in a file, PDF, or picture.";
}

function welcomeLine(now = new Date()) {
  return greeting(now) + ", what are we working on?";
}

// A tab left open keeps its old greeting, so check again now and then
// and whenever the user comes back to it.
function refreshWelcome() {
  const title = document.querySelector("#welcome-title");
  if (title && title.textContent !== welcomeLine()) title.textContent = welcomeLine();
}

// Listeners and first steps, run by main.js once every part is loaded.
export function start() {
  setInterval(refreshWelcome, 60_000);
  window.addEventListener("focus", refreshWelcome);
  document.addEventListener("visibilitychange", refreshWelcome);
}
