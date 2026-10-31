// The Settings window, and the one-time picture setup window.
import { state } from "./state.js";
import { bgNote, messageBox, modelSelect, plainError, settings } from "./core.js";
import { allPictures, isPictureModel, loadModels, modelLabel, pictureNotes } from "./models.js";
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
const gigabytes = (bytes) => (bytes / 1024 ** 3).toFixed(1) + " GB";
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
  const pictures = serverSettings.pictures;
  document.querySelector("#pic-program").value = pictures.program;
  document.querySelector("#pic-models").value = pictures.models;
  document.querySelector("#pic-address").value = pictures.address;
  showPictureSetup();
  await showContext();
}

async function showPictureSetup(saved = false) {
  const note = document.querySelector("#picture-note");
  let body;
  try {
    body = await (await fetch("/api/pictures")).json();
  } catch (error) {
    note.textContent = "Could not reach the helper.";
    return;
  }
  const parts = saved ? ["Saved."] : [];
  if (body.models.length) {
    parts.push(body.program
      ? "sd-server: " + body.program + ". Models: " + body.models.join(", ") + "."
      : (serverSettings.pictures.program
        ? "sd-server is not at the place written in \"Where sd-server is\". Empty that box, or use Set up picture making."
        : "sd-server is not in " + body.program_dir + " yet. Use Set up picture making."));
  } else if (!body.server) {
    parts.push("No picture model is set up yet.");
  }
  if (body.server) parts.push(body.server.error ? plainError(body.server.error) : "Picture server: " + body.server.model + ".");
  parts.push(...body.problems);
  // Said only when an Ollama image model is installed but cannot run here.
  const hidden = pictureNotes.find((line) => line.includes("picture models are hidden") || line.endsWith("image models are hidden."));
  if (hidden) parts.push(hidden);
  note.textContent = parts.join(" ");
}

// The one-time picture setup: the helper downloads sd-server and FLUX.2 klein
// into the app folder and adds the model; this window shows how far it is.
const pictureSetup = document.querySelector("#picture-setup");
const KLEIN_ID = "sd:flux2-klein";
let setupTimer = null;
let setupReloaded = false;

async function refreshSetup() {
  const note = document.querySelector("#setup-note");
  let body;
  try {
    body = await (await fetch("/api/pictures/setup")).json();
  } catch (error) {
    note.textContent = "Could not reach the HikariLM helper.";
    return;
  }
  document.querySelector("#setup-folder").textContent = body.klein.folder.replace(/[\\/]models$/, "");
  const parts = [{ key: "sd-server", label: "sd-server, the program that draws", ready: Boolean(body.program) },
    ...body.klein.files.map((file) => ({ key: file.name, label: file.name, ready: file.found }))];
  const list = document.querySelector("#setup-list");
  list.replaceChildren();
  for (const part of parts) {
    const item = document.createElement("li");
    const mark = document.createElement("span");
    mark.className = "mark";
    const now = body.running && body.file === part.key;
    mark.textContent = part.ready ? "✓" : now ? "↓" : "·";
    const label = document.createElement("span");
    label.textContent = part.label + (now && body.total ? " (" + Math.round((100 * body.done) / body.total) + "%)" : "");
    item.append(mark, label);
    list.append(item);
  }
  const allReady = parts.every((part) => part.ready);
  const usable = allPictures.some((item) => item.id === KLEIN_ID);
  if (body.finished && !usable && !setupReloaded) {
    setupReloaded = true;
    loadModels();
  }
  const bar = document.querySelector("#setup-bar");
  bar.hidden = !(body.running && body.total);
  if (!bar.hidden) bar.firstElementChild.style.width = ((100 * body.done) / body.total).toFixed(1) + "%";
  note.textContent = body.running
    ? "Downloading " + body.file + (body.total ? ": " + gigabytes(body.done) + " of " + gigabytes(body.total) : "…") +
      ". You can close this window; the download keeps going."
    : body.error || (usable ? "Done. Pick flux2-klein under Make pictures in the model list, then describe a picture." : "");
  const start = document.querySelector("#setup-start");
  start.hidden = body.running || usable;
  start.textContent = allReady ? "Finish setup" : parts.some((part) => part.ready) ? "Carry on" : "Download and set up";
  document.querySelector("#setup-stop").hidden = !body.running;
  document.querySelector("#setup-use").hidden = !usable || body.running;
}

export function openPictureSetup() {
  setupReloaded = false;
  pictureSetup.showModal();
  refreshSetup();
  clearInterval(setupTimer);
  setupTimer = setInterval(refreshSetup, 1000);
}
const setupAction = async (action) => {
  await fetch("/api/pictures/setup", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ action }),
  }).catch(() => {});
  refreshSetup();
};

async function savePictureSetting() {
  const wanted = {
    context: serverSettings.context,
    compaction: serverSettings.compaction,
    thinking: serverSettings.thinking,
    kv_cache: serverSettings.kv_cache,
    pictures: {
      program: document.querySelector("#pic-program").value,
      models: document.querySelector("#pic-models").value,
      address: document.querySelector("#pic-address").value,
    },
  };
  const response = await fetch("/api/settings", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(wanted),
  });
  const body = await response.json();
  if (!response.ok) {
    document.querySelector("#picture-note").textContent = "Not saved: " + plainError(body.error || "the helper refused it");
    return;
  }
  serverSettings = body;
  document.querySelector("#pic-address").value = body.pictures.address;
  await showPictureSetup(true);
  if (!state.activeRequest) loadModels();
}

// Steps of 1,024 up to 8K, then a quarter of each doubling, so there is
// something between 8,192 and 16,384 and every larger pair.
const CONTEXT_CHOICES = [2048, 3072, 4096, 5120, 6144, 7168];

function fillContextChoices(body) {
  const list = document.querySelector("#context-choices");
  list.replaceChildren(...CONTEXT_CHOICES.filter((size) => !body.trained || size <= body.trained).map((size) => {
    const option = document.createElement("option");
    option.value = size;
    if (size === body.default) option.label = "Ollama's default";
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
      const from = { model: "set by the model", OLLAMA_CONTEXT_LENGTH: "from OLLAMA_CONTEXT_LENGTH", ollama: "Ollama's built-in" }[body.default_from];
      const own = body.default.toLocaleString();
      document.querySelector("#ctx-model").placeholder = serverSettings?.context.all
        ? "Same: " + serverSettings.context.all.toLocaleString() : "Ollama's: " + own;
      document.querySelector("#ctx-all").placeholder = "Ollama's: " + own;
      parts.push("Ollama's default for " + modelLabel(model) + " is " + own + " tokens (" + from + ").");
    }
    if (body.error) parts.push(plainError(body.error));
    else if (model) {
      parts.push(modelLabel(model) + " will use " + body.num_ctx.toLocaleString() + " tokens.");
    }
    fillContextChoices(body);
    note.textContent = parts.join(" ") || "Blank uses Ollama's own setting.";
  } catch (error) {
    note.textContent = "";
  }
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
    pictures: serverSettings.pictures,
  };
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

// Listeners and first steps, run by main.js once every part is loaded.
export function start() {
  document.querySelector("#notes-toggle").addEventListener("change", (event) => {
    localStorage.setItem(NOTES_KEY, event.target.checked ? "on" : "off");
    showNotesSetting();
  });
  pictureSetup.addEventListener("close", () => clearInterval(setupTimer));
  document.querySelector("#close-picture-setup").addEventListener("click", () => pictureSetup.close());
  document.querySelector("#setup-start").addEventListener("click", () => { setupReloaded = false; setupAction("start"); });
  document.querySelector("#setup-stop").addEventListener("click", () => setupAction("stop"));
  document.querySelector("#setup-use").addEventListener("click", () => {
    pictureSetup.close();
    modelSelect.value = KLEIN_ID;
    modelSelect.dispatchEvent(new Event("change"));
    messageBox.focus();
  });
  document.querySelector("#open-picture-setup").addEventListener("click", () => {
    settings.close();
    openPictureSetup();
  });
  for (let size = 8192; size <= 262144; size *= 2) {
    for (let step = 0; step < 4 && size + step * size / 4 <= 262144; step++) CONTEXT_CHOICES.push(size + step * size / 4);
  }

  document.querySelectorAll("#settings input:not(#dark-toggle):not(#notify-toggle):not(#notes-toggle):not(#bg-file):not([id^=pic-]), #compact-prompt, #kv-bits")
    .forEach((input) => input.addEventListener("change", () => { if (serverSettings) saveContextSetting(); }));
  document.querySelectorAll("#picture-setting input, #pic-models")
    .forEach((input) => input.addEventListener("change", () => { if (serverSettings) savePictureSetting(); }));

  document.querySelector("#open-settings").addEventListener("click", () => {
    showNotesSetting();
    showContextSetting();
    settings.showModal();
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
