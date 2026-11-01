// LoRAs: the list in Settings, added by their Hugging Face repo, and the picker in Options,
// which shows only while the picture model is FLUX.2 klein base 4B.
import { plainError } from "./core.js";
import { copyButton } from "./copy.js";

const BASE_4B = "flux2-klein-base-4b";
const CHOSEN_KEY = "hikarilm-lora";
const STRENGTH_KEY = "hikarilm-lora-strength";
let loras = [];
let variant = "";

const picker = () => document.querySelector("#picture-lora");
const strengthBox = () => document.querySelector("#lora-strength");
export const findLora = (id) => loras.find((item) => item.id === id && item.ready) || null;

function megabytes(bytes) {
  return bytes >= 1e9 ? (bytes / 1e9).toFixed(1) + " GB" : Math.max(1, Math.round(bytes / 1e6)) + " MB";
}

export async function loadLoras() {
  try {
    loras = (await (await fetch("/api/loras")).json()).loras || [];
  } catch (error) {
    return;
  }
  fillPicker();
  showList();
}

function fillPicker() {
  const ready = loras.filter((item) => item.ready);
  const chosen = localStorage.getItem(CHOSEN_KEY) || "";
  picker().replaceChildren(new Option("None", ""), ...ready.map((item) => new Option(item.label, item.id)));
  picker().value = ready.some((item) => item.id === chosen) ? chosen : "";
  strengthBox().value = localStorage.getItem(STRENGTH_KEY) || "1";
  updateLoraPicker(variant);
}

// Called by models.js whenever the model changes, with the picture model's variant ("" for a chat model).
export function updateLoraPicker(next) {
  variant = next || "";
  const anyReady = loras.some((item) => item.ready);
  const onBase = variant === BASE_4B;
  const chosen = onBase ? findLora(picker().value) : null;
  document.querySelector("#lora-field").hidden = !(onBase && anyReady);
  document.querySelector("#strength-field").hidden = !chosen;
  document.querySelector("#lora-off").hidden = !(variant && !onBase && anyReady);
  const toggle = document.querySelector("#picture-toggle");
  toggle.textContent = chosen ? "Options · LoRA" : "Options";
  toggle.title = chosen ? "Drawing with the LoRA " + chosen.label : "";
}

function strength() {
  const value = Number(strengthBox().value);
  return value > 0 && value <= 2 ? Math.round(value * 100) / 100 : 1;
}

// What a new picture asks for: the LoRA picked in Options, when the model is the base 4B.
export function loraRequest() {
  const chosen = variant === BASE_4B ? findLora(picker().value) : null;
  return chosen ? { lora: chosen.id, lora_strength: strength() } : {};
}

// --- Settings -------------------------------------------------------------------

function commandLine(command) {
  const line = document.createElement("div");
  line.className = "lora-command";
  const code = document.createElement("code");
  code.textContent = command;
  line.append(code, copyButton(() => command, code));
  return line;
}

function paragraph(text, className = "") {
  const element = document.createElement("p");
  element.textContent = text;
  if (className) element.className = className;
  return element;
}

function loraItem(item) {
  const box = document.createElement("li");
  box.className = "lora-item" + (item.ready ? " ready" : "");
  const head = document.createElement("div");
  head.className = "lora-head";
  const name = document.createElement("strong");
  name.textContent = item.repo;
  const size = document.createElement("span");
  size.textContent = item.size ? megabytes(item.size) : "";
  head.append(name, size);
  box.append(head, paragraph(item.file, "lora-file"));
  if (item.error) box.append(paragraph(item.error, "lora-bad"));
  else if (item.ready) box.append(paragraph("Downloaded and ready. Pick it under Options when drawing with FLUX.2 klein base 4B."));
  else box.append(paragraph("Not downloaded yet. In Terminal:"), commandLine(item.command));
  if (item.note) box.append(paragraph(item.note, "note"));
  const trigger = document.createElement("label");
  trigger.className = "lora-trigger";
  const triggerName = document.createElement("span");
  triggerName.textContent = "Trigger word";
  const input = document.createElement("input");
  input.className = "lora-input";
  input.value = item.trigger;
  input.placeholder = "None";
  input.maxLength = 100;
  input.title = "Put in front of every prompt drawn with this LoRA, unless the prompt has it already.";
  input.addEventListener("change", () => saveTrigger(item, input));
  trigger.append(triggerName, input);
  const actions = document.createElement("div");
  actions.className = "lora-actions";
  const remove = document.createElement("button");
  remove.type = "button";
  remove.className = "choice";
  remove.textContent = "Remove";
  remove.title = "Take it off the list. The download stays in the Hugging Face cache.";
  remove.addEventListener("click", () => removeLora(item));
  actions.append(remove);
  box.append(trigger, actions);
  return box;
}

function showList() {
  document.querySelector("#lora-list").replaceChildren(...loras.map(loraItem));
  document.querySelector("#lora-empty").hidden = loras.length > 0;
  document.querySelector("#lora-again").hidden = !loras.some((item) => !item.ready && !item.error);
}

async function post(path, body) {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const answer = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(answer.error || "The helper could not do that.");
  return answer;
}

async function saveTrigger(item, input) {
  try {
    const answer = await post("/api/loras/trigger", { id: item.id, trigger: input.value });
    item.trigger = answer.lora.trigger;
    input.value = item.trigger;
  } catch (error) {
    input.value = item.trigger;
    document.querySelector("#lora-found").replaceChildren(paragraph(plainError(error.message), "lora-bad"));
  }
}

async function removeLora(item) {
  try {
    await post("/api/loras/remove", { id: item.id });
  } catch (error) {}
  if (localStorage.getItem(CHOSEN_KEY) === item.id) localStorage.removeItem(CHOSEN_KEY);
  await loadLoras();
}

// Look a repo up and show what it holds, with Add, or why it cannot be drawn.
async function lookUp(text) {
  const found = document.querySelector("#lora-found");
  found.replaceChildren(paragraph("Looking up " + text + " on Hugging Face…"));
  let answer;
  try {
    answer = await post("/api/loras", { repo: text });
  } catch (error) {
    found.replaceChildren(paragraph(plainError(error.message), "lora-bad"));
    return;
  }
  const lora = answer.found;
  const head = document.createElement("div");
  head.className = "lora-head";
  const name = document.createElement("strong");
  name.textContent = lora.repo;
  const size = document.createElement("span");
  size.textContent = megabytes(lora.size);
  head.append(name, size);
  const parts = [head];
  if (lora.files.length > 1) {
    const select = document.createElement("select");
    select.setAttribute("aria-label", "File in the repo");
    for (const file of lora.files) select.append(new Option(file.name + " (" + megabytes(file.size) + ")", file.name));
    select.value = lora.file;
    select.addEventListener("change", () => lookUp(lora.repo + ":" + select.value));
    parts.push(paragraph("It has " + lora.files.length + " files. This one is the LoRA; pick another if its page says so."), select);
  } else {
    parts.push(paragraph(lora.file, "lora-file"));
  }
  if (lora.error) {
    parts.push(paragraph(lora.error, "lora-bad"));
  } else {
    parts.push(paragraph(lora.trigger ? "Trigger word: " + lora.trigger + ". It goes in front of your prompt." :
      "Its page names no trigger word. If it has one, add it after."));
    if (lora.note) parts.push(paragraph(lora.note, "note"));
    if (lora.gated) parts.push(paragraph("Its makers ask you to accept their terms on its Hugging Face page first, and to sign in with  ~/.hikarilm/venv/bin/hf auth login.", "note"));
    const actions = document.createElement("div");
    actions.className = "lora-actions";
    const add = document.createElement("button");
    add.type = "button";
    add.className = "choice";
    add.textContent = "Add";
    add.addEventListener("click", () => {
      add.disabled = true;
      add.textContent = "Adding…";
      addLora(lora.id);
    });
    actions.append(add);
    parts.push(actions);
  }
  found.replaceChildren(...parts);
}

async function addLora(id) {
  const found = document.querySelector("#lora-found");
  try {
    const answer = await post("/api/loras", { repo: id, add: true });
    loras = answer.loras || loras;
  } catch (error) {
    found.replaceChildren(paragraph(plainError(error.message), "lora-bad"));
    return;
  }
  document.querySelector("#lora-repo").value = "";
  const added = loras.find((item) => item.id === id);
  found.replaceChildren(paragraph(added?.ready ? "Added, and already downloaded."
    : "Added. Download it with the command in the list; it is ready once the download is done."));
  fillPicker();
  showList();
}

// Listeners and first steps, run by main.js once every part is loaded.
export function start() {
  document.querySelector("#lora-add").addEventListener("submit", (event) => {
    event.preventDefault();
    const text = document.querySelector("#lora-repo").value.trim();
    if (text) lookUp(text);
  });
  document.querySelector("#lora-check").addEventListener("click", loadLoras);
  document.querySelector("#open-settings").addEventListener("click", loadLoras);
  document.querySelector("#picture-toggle").addEventListener("click", loadLoras);
  picker().addEventListener("change", () => {
    localStorage.setItem(CHOSEN_KEY, picker().value);
    updateLoraPicker(variant);
  });
  strengthBox().addEventListener("change", () => {
    strengthBox().value = String(strength());
    localStorage.setItem(STRENGTH_KEY, strengthBox().value);
  });
  loadLoras();
}
