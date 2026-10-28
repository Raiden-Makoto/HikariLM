// The model picker, the header's memory line, and picture models.
import { state } from "./state.js";
import { messageBox, modelSelect, plainError, status } from "./core.js";
import { attachButton } from "./attachments.js";

// What the last send used, from the helper: the context length it ran
// with, where compaction starts, and (after the reply) the tokens used.

export async function refreshState() {
  const response = await fetch("/api/state");
  const body = await response.json();
  let text = body.large_model
    ? "Kept loaded: " + (isPictureModel(body.large_model) ? pictureLabel(body.large_model) + " (pictures)" : modelLabel(body.large_model))
    : "No large model loaded.";
  if (body.large_model && body.unloads_at && !state.activeRequest) {
    const left = body.unloads_at - Date.now() / 1000;
    if (left > 86400) text += " (pinned by another program)";
    else if (left > 90) text += " · unloads in " + Math.round(left / 60) + " min";
    else text += " · unloads in under a minute";
  }
  if (body.also_loaded) text += " · " + body.also_loaded + " more model" + (body.also_loaded > 1 ? "s" : "") + " in memory";
  if (state.lastContext && state.lastContext.used) {
    text += " · Context used: " + state.lastContext.used.toLocaleString() + " of " + state.lastContext.num_ctx.toLocaleString() +
      " tokens (compacts past " + state.lastContext.threshold.toLocaleString() + ")";
  }
  status.textContent = text;
}

export async function loadModels() {
  let models = [];
  let problem = "";
  try {
    const response = await fetch("/api/tags");
    const body = await response.json();
    models = body.models || [];
    allPictures = body.pictures || [];
    pictureNotes = body.picture_notes || [];
    fillPictureSizes(body.sizes || {});
    if (!response.ok) problem = plainError(body.error || "Could not list models.");
    else if (models.length === 0 && allPictures.length === 0) problem = "No chat models installed in Ollama.";
  } catch (error) {
    problem = "The helper is not responding.";
  }
  allModels = models;
  fillModels(problem);
}

// Picture models answer a message with a picture. They are listed apart in
// the picker, only in Chat, and never get the chat's context settings.
export let allPictures = [];
export let pictureNotes = [];
const SIZE_NAMES = { square: "Square", portrait: "Portrait", landscape: "Landscape", wide: "Wide", small: "Small square" };
export const pictureSize = document.querySelector("#picture-size");
export const pictureSeed = document.querySelector("#picture-seed");
export const pictureSteps = document.querySelector("#picture-steps");
export const pictureToggle = document.querySelector("#picture-toggle");
export const isPictureModel = (value) => allPictures.some((item) => item.id === value);
export const pictureLabel = (id) => allPictures.find((item) => item.id === id)?.label || id.replace(/^(sd|remote):/, "");

function fillPictureSizes(sizes) {
  if (pictureSize.options.length || !Object.keys(sizes).length) return;
  for (const [key, [width, height]] of Object.entries(sizes)) {
    pictureSize.append(new Option((SIZE_NAMES[key] || key) + ", " + width + " × " + height, key));
  }
}

function setPictureOptions(open) {
  document.querySelector("#picture-options").hidden = !open;
  pictureToggle.setAttribute("aria-expanded", String(open));
}

export function updatePictureMode() {
  const on = isPictureModel(modelSelect.value);
  pictureToggle.hidden = !on;
  if (!on) setPictureOptions(false);
  attachButton.hidden = on;
  messageBox.placeholder = on ? "Describe the picture" : "Message";
}

// The picker shows every chat model in Chat. In the code editor it shows
// only text-only models, with coder models first, because the editor has
// no pictures or PDFs to read.
let allModels = [];
let modelProblem = "";
export const CODE_MODEL_KEY = "lightusagi-model-code";
export const inCodeView = () => document.body.classList.contains("view-files");
const looksVision = (name) =>
  /gemma3|llava|vision|minicpm-v|moondream|pixtral|mllama|qwen[\d.]*-?vl|[-_.:]vl\b/i.test(name);
export const codeRank = (name) => (/qwen.*coder|coder.*qwen/i.test(name) ? 0 : /code|devstral/i.test(name) ? 1 : 2);

// Models from LM Studio's llmster carry an lms: prefix; the picker shows their own name.
const isLmsModel = (name) => name.startsWith("lms:");
const modelLabel = (name) => (isLmsModel(name) ? name.slice(4) + " (LM Studio)" : name);
export const SETUP_PICTURES = "__setup_pictures";

export function modelsForView() {
  if (!inCodeView()) return allModels;
  return allModels
    .filter((name) => !looksVision(name))
    .map((name, index) => ({ name, index }))
    .sort((a, b) => codeRank(a.name) - codeRank(b.name) || a.index - b.index)
    .map((item) => item.name);
}

export function fillModels(problem = modelProblem) {
  modelProblem = problem;
  const models = modelsForView();
  const pictures = inCodeView() ? [] : allPictures;
  modelSelect.replaceChildren();
  if (models.length === 0 && pictures.length === 0) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = allModels.length ? "No text-only model" : "No model selected";
    modelSelect.append(option);
    modelSelect.disabled = true;
    status.textContent = problem || (allModels.length
      ? "The code editor needs a text-only model, and the installed ones also read pictures."
      : "");
    updatePictureMode();
    return;
  }
  const saved = localStorage.getItem(inCodeView() ? CODE_MODEL_KEY : "lightusagi-model");
  const group = (label) => {
    const element = document.createElement("optgroup");
    element.label = label;
    modelSelect.append(element);
    return element;
  };
  const fromOllama = models.filter((name) => !isLmsModel(name));
  const fromLms = models.filter(isLmsModel);
  const grouped = pictures.length || fromLms.length || !inCodeView();
  const chatPlace = grouped && fromOllama.length ? group(fromLms.length ? "Ollama" : "Chat") : modelSelect;
  for (const name of fromOllama) chatPlace.append(new Option(name, name));
  if (fromLms.length) {
    const lmsPlace = group("LM Studio");
    for (const name of fromLms) lmsPlace.append(new Option(modelLabel(name), name));
  }
  if (pictures.length) {
    const picturePlace = group("Make pictures");
    for (const item of pictures) picturePlace.append(new Option(item.label, item.id));
  } else if (!inCodeView()) {
    group("Make pictures").append(new Option("Set up picture making…", SETUP_PICTURES));
  }
  const values = [...models, ...pictures.map((item) => item.id)];
  modelSelect.value = values.includes(saved) ? saved : values[0];
  modelSelect.disabled = Boolean(state.activeRequest);
  updatePictureMode();
  refreshState();
}

// Listeners and first steps, run by main.js once every part is loaded.
export function start() {
  // Models leave memory on their own, so the header asks again now and then.
  setInterval(() => {
    if (document.visibilityState === "visible" && !state.activeRequest) refreshState().catch(() => {});
  }, 30000);
  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "visible" && !state.activeRequest) refreshState().catch(() => {});
  });
  pictureToggle.addEventListener("click", () => setPictureOptions(document.querySelector("#picture-options").hidden));
}
