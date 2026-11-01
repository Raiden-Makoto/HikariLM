// The Settings window.
import { bgNote, messageBox, modelSelect, plainError, settings } from "./core.js";
import {
  READER_KEY, REWRITE_KEY, isPictureModel, modelLabel, readers, rewriteModel, rewriters, updateChangeNote,
} from "./models.js";
import { applyBackground } from "./page.js";

// Earlier chats and pictures: the helper keeps small notes of what was
// answered and read, and a few close to a new question go along with it.
const NOTES_KEY = "hikarilm-notes";
export const notesOn = () => localStorage.getItem(NOTES_KEY) !== "off";

async function showNotesSetting() {
  document.querySelector("#notes-toggle").checked = notesOn();
  const note = document.querySelector("#notes-note");
  note.textContent = notesOn()
    ? "A few notes from earlier chats and pictures that match your question are sent along with it. Not used in the code editor."
    : "Off. Each question is sent on its own.";
  try {
    const body = await (await fetch("/api/notes")).json();
    note.textContent += " " + body.notes + (body.notes === 1 ? " note" : " notes") + " kept so far." +
      (body.error ? " Not indexed yet: " + plainError(body.error) : "");
  } catch (error) {
    // The count is only a courtesy.
  }
}

// Context length and compaction live in the helper, which enforces them.
let serverSettings = null;
const numberOrNull = (input) => (input.value.trim() === "" ? null : Number(input.value));

// Context length belongs to chat models; a picture model has none.
const contextModel = () => (isPictureModel(modelSelect.value) ? "" : modelSelect.value);

async function showContextSetting() {
  const model = contextModel();
  try {
    serverSettings = await (await fetch("/api/settings")).json();
  } catch (error) {
    document.querySelector("#context-note").textContent = "Could not reach the helper.";
    return;
  }
  const { context: ctx, compaction } = serverSettings;
  document.querySelector("#ctx-all").value = ctx.all ?? "";
  document.querySelector("#ctx-model").value = (model && ctx.models[model]) ?? "";
  document.querySelector("#ctx-model").disabled = !model;
  document.querySelector("#ctx-model-label").textContent = model ? "Only " + modelLabel(model) : "This model";
  document.querySelector("#compact-toggle").checked = compaction.enabled;
  document.querySelector("#compact-threshold").value = compaction.threshold ?? "";
  document.querySelector("#compact-retention").value = compaction.retention;
  document.querySelector("#compact-max").value = compaction.max_tokens;
  document.querySelector("#compact-prompt").value = compaction.prompt;
  document.querySelector("#compact-note").textContent = "";
  document.querySelector("#think-toggle").checked = serverSettings.thinking.enabled;
  document.querySelector("#think-limit").value = serverSettings.thinking.budget ?? "";
  document.querySelector("#kv-bits").value = String(serverSettings.kv_cache.bits);
  showKvWarning();
  showPictureSetting();
  await showContext();
}

// Where picture making stands: what is downloaded, whether mflux is installed, and what
// was left out.
function pictureStanding(body) {
  const parts = [];
  if (body.models.length) {
    parts.push("Downloaded: " + body.models.join(", ") + ".");
    parts.push(body.ready ? "Ready: pick it under Make pictures in the model list."
      : "Next, run  hikarilm restart  in Terminal to install mflux.");
  } else {
    parts.push("No FLUX.2 klein is downloaded yet.");
  }
  parts.push(...body.notes);
  return parts.join(" ");
}

async function showPictureSetting() {
  showReaderSetting();
  showRewriteSetting();
  let body = null;
  try {
    body = await (await fetch("/api/pictures")).json();
  } catch (error) {}
  document.querySelector("#picture-note").textContent = body ? pictureStanding(body) : "Could not reach the helper.";
}

function showReaderSetting() {
  const select = document.querySelector("#reader-model");
  const choices = readers();
  select.disabled = !choices.length;
  if (!choices.length) {
    select.replaceChildren(new Option("None installed", ""));
    return;
  }
  const chosen = localStorage.getItem(READER_KEY);
  select.replaceChildren(...choices.map((name) => new Option(modelLabel(name), name)));
  select.value = choices.includes(chosen) ? chosen : choices[0];
}

function showRewriteSetting() {
  const select = document.querySelector("#rewrite-model");
  const writer = rewriteModel();
  select.replaceChildren(new Option("Don't rewrite", "off"),
    ...rewriters().map((name) => new Option(modelLabel(name), name)));
  select.value = writer || "off";
}

// Steps of 1,024 up to 8K, then a quarter of each doubling, so there is
// something between 8,192 and 16,384 and every larger pair.
const CONTEXT_CHOICES = [2048, 3072, 4096, 5120, 6144, 7168];

function fillContextChoices(body) {
  const list = document.querySelector("#context-choices");
  list.replaceChildren(...CONTEXT_CHOICES.filter((size) => !body.trained || size <= body.trained).map((size) => {
    const option = document.createElement("option");
    option.value = size;
    if (size === body.default) option.label = "Default";
    return option;
  }));
}

async function showContext() {
  const note = document.querySelector("#context-note");
  const model = contextModel();
  try {
    const body = await (await fetch("/api/context?model=" + encodeURIComponent(model))).json();
    const parts = [];
    if (body.default) {
      const own = body.default.toLocaleString();
      document.querySelector("#ctx-model").placeholder = serverSettings?.context.all
        ? "Same: " + serverSettings.context.all.toLocaleString() : "Default: " + own;
      document.querySelector("#ctx-all").placeholder = "Default: " + own;
      parts.push("The default for " + modelLabel(model) + " is " + own + " tokens.");
    }
    if (body.error) parts.push(plainError(body.error));
    else if (model) {
      parts.push(modelLabel(model) + " will use " + body.num_ctx.toLocaleString() + " tokens.");
    }
    fillContextChoices(body);
    note.textContent = parts.join(" ") || "Blank uses each model's default.";
  } catch (error) {
    note.textContent = "";
  }
}

function showKvWarning() {
  document.querySelector("#kv-warning").hidden = document.querySelector("#kv-bits").value !== "4";
}

async function saveContextSetting() {
  const model = contextModel();
  const models = { ...serverSettings.context.models };
  const own = numberOrNull(document.querySelector("#ctx-model"));
  if (model) {
    if (own) models[model] = own;
    else delete models[model];
  }
  const wanted = {
    context: { all: numberOrNull(document.querySelector("#ctx-all")), models },
    compaction: {
      enabled: document.querySelector("#compact-toggle").checked,
      threshold: numberOrNull(document.querySelector("#compact-threshold")),
      retention: numberOrNull(document.querySelector("#compact-retention")) ?? 40,
      max_tokens: numberOrNull(document.querySelector("#compact-max")) ?? 1000,
      prompt: document.querySelector("#compact-prompt").value,
    },
    thinking: {
      enabled: document.querySelector("#think-toggle").checked,
      budget: numberOrNull(document.querySelector("#think-limit")),
    },
    kv_cache: { bits: Number(document.querySelector("#kv-bits").value) },
  };
  showKvWarning();
  const note = document.querySelector("#compact-note");
  const response = await fetch("/api/settings", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(wanted),
  });
  const body = await response.json();
  if (!response.ok) {
    note.textContent = "Not saved: " + plainError(body.error || "the helper refused it");
    return;
  }
  serverSettings = body;
  note.textContent = "Saved.";
  document.querySelector("#compact-retention").value = body.compaction.retention;
  document.querySelector("#think-limit").value = body.thinking.budget ?? "";
  await showContext();
}

const ISSUES = "https://github.com/Raiden-Makoto/HikariLM/issues/new";

// Read ahead of the click: Safari blocks a window opened after an await.
let version = "unknown";

// Updates: how far behind GitHub this copy is, and the button that pulls and restarts it.
async function showUpdateSetting() {
  const note = document.querySelector("#update-note");
  const button = document.querySelector("#update-button");
  if (button.dataset.restarting) return;
  note.textContent = "HikariLM " + version + ". Checking GitHub…";
  let body;
  try {
    body = await (await fetch("/api/update")).json();
  } catch (error) {
    note.textContent = "HikariLM " + version + ". Could not reach the helper.";
    return;
  }
  button.disabled = !body.git;
  const here = "HikariLM " + version + (body.head ? " (" + body.head + ")" : "") + ". ";
  if (!body.git) note.textContent = here + "This copy was not made with git clone, so it cannot update itself. Download the new release from GitHub.";
  else if (body.error) note.textContent = here + body.error;
  else if (body.behind) note.textContent = here + body.behind + (body.behind === 1 ? " new change" : " new changes") + " on GitHub.";
  else note.textContent = here + "Up to date.";
}

async function runUpdate() {
  const note = document.querySelector("#update-note");
  const button = document.querySelector("#update-button");
  button.disabled = true;
  note.textContent = "Getting the new code…";
  let body;
  let before;
  try {
    before = await (await fetch("/api/state")).json();
    const response = await fetch("/api/update", { method: "POST" });
    body = await response.json();
    if (!response.ok) throw new Error(body.error || "The update failed.");
  } catch (error) {
    note.textContent = plainError(error.message);
    button.disabled = false;
    return;
  }
  if (!body.restarting) {
    note.textContent = "HikariLM " + version + ". Already up to date.";
    button.disabled = false;
    return;
  }
  // The new copy runs the first-start steps again, which can take minutes when an update
  // brings new packages. The page reloads once it answers with its new code.
  button.dataset.restarting = "1";
  note.textContent = "Updated from " + body.from + " to " + body.to + ". Restarting…";
  const started = Date.now();
  while (Date.now() - started < 15 * 60 * 1000) {
    await new Promise((resolve) => setTimeout(resolve, 1500));
    try {
      const now = await (await fetch("/api/state", { cache: "no-store" })).json();
      if (now.pid !== before.pid || now.code !== before.code) {
        window.location.reload();
        return;
      }
    } catch (error) {
      note.textContent = "Updated from " + body.from + " to " + body.to + ". Restarting… (" +
        Math.round((Date.now() - started) / 1000) + " s)";
    }
  }
  delete button.dataset.restarting;
  note.textContent = "It has not come back yet. In Terminal, hikarilm log shows what it is doing, and hikarilm start starts it.";
}

function reportIssue() {
  const model = modelSelect.value;
  const body = [
    "**What happened**", "", "", "**What you expected**", "", "",
    "**Details**",
    "- HikariLM: " + version,
    "- Model: " + (model ? modelLabel(model) + " (`" + model + "`)" : "none chosen"),
    "- Browser: " + navigator.userAgent,
  ].join("\n");
  window.open(ISSUES + "?" + new URLSearchParams({ body }), "_blank", "noopener");
}

// Listeners and first steps, run by main.js once every part is loaded.
export function start() {
  document.querySelector("#notes-toggle").addEventListener("change", (event) => {
    localStorage.setItem(NOTES_KEY, event.target.checked ? "on" : "off");
    showNotesSetting();
  });
  for (let size = 8192; size <= 262144; size *= 2) {
    for (let step = 0; step < 4 && size + step * size / 4 <= 262144; step++) CONTEXT_CHOICES.push(size + step * size / 4);
  }

  document.querySelectorAll("#settings input:not(#dark-toggle):not(#notify-toggle):not(#notes-toggle):not(#bg-file):not(.lora-input), #compact-prompt, #kv-bits")
    .forEach((input) => input.addEventListener("change", () => { if (serverSettings) saveContextSetting(); }));

  document.querySelector("#open-settings").addEventListener("click", () => {
    showNotesSetting();
    showContextSetting();
    showUpdateSetting();
    settings.showModal();
  });
  fetch("/api/state").then((response) => response.json())
    .then((body) => { version = body.version || version; }, () => {});
  document.querySelector("#report-issue").addEventListener("click", reportIssue);
  document.querySelector("#update-button").addEventListener("click", runUpdate);
  document.querySelector("#rewrite-model").addEventListener("change", (event) => {
    localStorage.setItem(REWRITE_KEY, event.target.value);
    updateChangeNote();
  });
  document.querySelector("#reader-model").addEventListener("change", (event) => {
    localStorage.setItem(READER_KEY, event.target.value);
  });
  document.querySelector("#close-settings").addEventListener("click", () => settings.close());
  settings.addEventListener("click", (event) => {
    if (event.target === settings) settings.close();
  });
  settings.addEventListener("close", () => messageBox.focus());
  document.querySelector("#pick-bg").addEventListener("click", () => {
    document.querySelector("#bg-file").click();
  });
  document.querySelector("#bg-file").addEventListener("change", () => {
    const file = document.querySelector("#bg-file").files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      const url = String(reader.result);
      bgNote.textContent = "";
      try {
        localStorage.setItem("hikarilm-bg", url);
      } catch (error) {
        bgNote.textContent = "That image is too large to keep after a refresh.";
      }
      applyBackground(url);
    };
    reader.readAsDataURL(file);
  });
  document.querySelector("#clear-bg").addEventListener("click", () => {
    localStorage.removeItem("hikarilm-bg");
    document.querySelector("#bg-file").value = "";
    bgNote.textContent = "";
    applyBackground("");
  });
}
