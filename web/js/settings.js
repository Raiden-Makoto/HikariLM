// The Settings window.
import { bgNote, messageBox, modelSelect, plainError, settings } from "./core.js";
import { isPictureModel, modelLabel } from "./models.js";
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

async function showPictureSetting() {
  const note = document.querySelector("#picture-note");
  let body;
  try {
    body = await (await fetch("/api/pictures")).json();
  } catch (error) {
    note.textContent = "Could not reach the helper.";
    return;
  }
  document.querySelector("#picture-get").textContent = body.get;
  const parts = [];
  if (body.models.length) {
    parts.push("Downloaded: " + body.models.join(", ") + ".");
    if (!body.ready) parts.push("mflux is not set up yet. Run ./start.sh again to install it.");
  } else {
    parts.push("No FLUX.2 model is downloaded yet.");
  }
  parts.push(...body.notes);
  note.textContent = parts.join(" ");
}

// From the model list when no picture model is downloaded: Settings says how to get one.
export function openPictureSetup() {
  showNotesSetting();
  showContextSetting();
  settings.showModal();
  document.querySelector("#picture-setting").scrollIntoView({ block: "center" });
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

  document.querySelectorAll("#settings input:not(#dark-toggle):not(#notify-toggle):not(#notes-toggle):not(#bg-file), #compact-prompt, #kv-bits")
    .forEach((input) => input.addEventListener("change", () => { if (serverSettings) saveContextSetting(); }));

  document.querySelector("#open-settings").addEventListener("click", () => {
    showNotesSetting();
    showContextSetting();
    settings.showModal();
  });
  fetch("/api/state").then((response) => response.json())
    .then((body) => { version = body.version || version; }, () => {});
  document.querySelector("#report-issue").addEventListener("click", reportIssue);
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
