// One message on screen: its text, thinking, files, pictures and Apply buttons, and the busy state.
import { state } from "./state.js";
import { log, messageBox, modelSelect, scrollLog, sendButton, showEmpty, sidebar } from "./core.js";
import { attachButton } from "./attachments.js";
import { renderMarkdown, showThought, splitStrayThinking } from "./markdown.js";
import { pictureLabel, pictureSeed, pictureSize, pictureSteps, pictureToggle, updateChangeNote } from "./models.js";
import { closeMenu } from "./chat-list.js";
import { varyPicture } from "./send.js";
import { applyCode, syncApplyButtons } from "./apply.js";
import { deletePrompt, editPrompt } from "./prompt-actions.js";

export function addTurn(role, text, thought, files, message, question) {
  if (role === "assistant" && text) ({ text, thought } = splitStrayThinking(text, thought));
  document.querySelector("#empty")?.remove();
  const block = document.createElement("article");
  block.className = "turn " + role;
  const who = document.createElement("span");
  who.className = "who";
  who.textContent = role === "user" ? "You" : "Model";
  // Progress lines (reading a picture, drawing one) show here, apart from the model's thinking.
  const thinking = document.createElement("div");
  thinking.className = "thinking";
  thinking.hidden = true;
  const box = document.createElement("details");
  box.className = "thought";
  const label = document.createElement("summary");
  const thoughtText = document.createElement("div");
  thoughtText.className = "thought-text";
  box.append(label, thoughtText);
  const view = { box, label, text: thoughtText, full: "" };
  box.addEventListener("toggle", () => { if (box.open) thoughtText.textContent = view.full; });
  showThought(view, thought || "", { seconds: message?.thoughtSeconds ?? null, cut: Boolean(message?.thinkingCut) });
  const calls = document.createElement("div");
  calls.className = "tool-calls";
  for (const call of message?.tools || []) showToolCall(calls, call);
  const answer = document.createElement("div");
  if (role === "assistant" && text) renderMarkdown(answer, text);
  else answer.textContent = text;
  if (message?.picture) answer.append(pictureFigure(message));
  block.append(who, thinking, box, calls, answer);
  if (files?.length) {
    const row = document.createElement("div");
    row.className = "chips";
    for (const file of files) {
      const chip = document.createElement("div");
      chip.className = "chip";
      const name = document.createElement("span");
      name.textContent = file.name;
      chip.append(name);
      row.append(chip);
    }
    who.after(row);
  }
  if (role === "user") {
    thinking.remove();
    box.remove();
    calls.remove();
    if (message) {
      const actions = document.createElement("div");
      actions.className = "prompt-actions";
      for (const [label, act] of [["Edit", () => editPrompt(message, block, answer)], ["Delete", () => deletePrompt(message)]]) {
        const button = document.createElement("button");
        button.type = "button";
        button.textContent = label;
        button.addEventListener("click", act);
        actions.append(button);
      }
      block.append(actions);
    }
  } else if (text) {
    updateApply(answer, question);
  }
  log.append(block);
  scrollLog();
  return { block, thinking, thought: view, calls, answer };
}

const SHORT = 40;
const short = (text) => (text.length > SHORT ? text.slice(0, SHORT) + "…" : text);

const part = (label, text) => {
  const head = document.createElement("div");
  head.className = "tool-part";
  head.textContent = label;
  const body = document.createElement("pre");
  body.textContent = text;
  return [head, body];
};

// One use of the tool: "Ran Python", opening to the code, what it printed and its plots. While
// it runs (call.running) the line says so. Answers from 4.8 have the calculator's
// "Calculated: sqrt(4761) = 69". Returns the line, for the running one to be swapped out.
export function showToolCall(container, call, replacing) {
  const box = document.createElement("details");
  box.className = "tool-call" + (call.error ? " failed" : "") + (call.running ? " running" : "");
  const line = document.createElement("summary");
  if (call.code === undefined) {
    line.textContent = call.error
      ? "Calculator: " + short(call.expression) + " could not be worked out"
      : "Calculated: " + short(call.expression) + " = " + short(call.result);
    const full = document.createElement("pre");
    full.textContent = call.expression + "\n= " + (call.error ? "Error: " + call.error : call.result);
    box.append(line, full);
  } else {
    const plots = (call.pictures || []).length;
    line.textContent = call.running ? "Running Python…"
      : "Ran Python" + (plots ? " · " + plots + (plots === 1 ? " plot" : " plots") : "") + (call.error ? " · " + short(call.error) : "");
    box.append(line, ...part("Code", call.code));
    if (!call.running) {
      const output = call.output || "";
      if (output.trim() || !call.error) box.append(...part("Output", output.trim() ? output : "(nothing printed)"));
      if (call.error && !output.includes(call.error)) box.append(...part("Error", call.error));
      for (const src of call.pictures || []) {
        const img = document.createElement("img");
        img.className = "tool-plot";
        img.src = src;
        img.alt = "A plot the Python code made";
        box.append(img);
      }
    }
  }
  if (replacing) replacing.replaceWith(box);
  else container.append(box);
  return box;
}

// In the code editor, a finished reply's code blocks get Apply, as in Cursor. Which
// ones show is applyTarget's call (edit-blocks.js), from the question this reply
// answers: { text, files } of the user's message. Drawn again whenever the reply is.
export function updateApply(answer, question) {
  answer.afterRender = () => {
    for (const box of answer.querySelectorAll(".code-block:not([data-quote])")) {
      const path = box.dataset.path || "";
      const button = document.createElement("button");
      button.type = "button";
      button.className = "code-apply";
      button.textContent = "Apply";
      button.hidden = true;
      button.dataset.lang = box.dataset.lang || "";
      button.dataset.path = path;
      button.question = question;
      button.addEventListener("click", () => applyCode(box.querySelector("code").textContent, path));
      box.querySelector(".code-head").append(button);
    }
    syncApplyButtons();
  };
  if (answer.classList.contains("md")) answer.afterRender();
}

export function renderLog() {
  if (state.messages.length === 0) {
    showEmpty();
    return;
  }
  log.replaceChildren();
  let question = null;
  for (const message of state.messages) {
    if (message.role === "user") question = { text: message.content, files: message.files };
    addTurn(message.role, message.content, message.thinking, message.files, message, question);
  }
  updateChangeNote();
}

// A picture reply is kept in the chat as its notes; the file is shown by
// address, so its bytes never sit in the page or in the saved chat.
export function pictureFigure(message) {
  const record = message.picture;
  const figure = document.createElement("figure");
  figure.className = "picture";
  const img = document.createElement("img");
  img.src = "/api/images/" + record.id + ".png";
  img.alt = record.prompt;
  img.loading = "lazy";
  if (record.width && record.height) {
    img.width = record.width;
    img.height = record.height;
  }
  const caption = document.createElement("figcaption");
  const references = (record.references || []).length;
  caption.textContent = [record.width && record.width + " × " + record.height, "seed " + record.seed, pictureLabel(record.model),
    record.lora && "LoRA " + record.lora.label + (record.lora.strength !== 1 ? " at " + record.lora.strength : ""),
    record.changed && "a change to the picture before",
    references && "drawn from " + references + (references === 1 ? " reference picture" : " reference pictures")]
    .filter(Boolean).join(" · ");
  img.addEventListener("error", () => { caption.textContent = "The picture file is missing from the helper's images folder."; });
  const actions = document.createElement("div");
  actions.className = "actions";
  const download = document.createElement("a");
  download.className = "choice";
  download.href = img.src;
  download.download = "hikarilm-" + record.id + ".png";
  download.textContent = "Download";
  const vary = document.createElement("button");
  vary.type = "button";
  vary.className = "choice";
  vary.textContent = "Vary";
  vary.title = "Same prompt with a new seed. The new picture replaces this one.";
  vary.addEventListener("click", () => varyPicture(message, figure));
  actions.append(download, vary);
  figure.append(img, caption);
  if (record.request) {
    const drawn = document.createElement("details");
    drawn.className = "drawn-prompt";
    const summary = document.createElement("summary");
    summary.textContent = "Drawn from a rewritten prompt";
    const text = document.createElement("p");
    text.textContent = record.prompt;
    drawn.append(summary, text);
    if (record.reference) {
      const seen = document.createElement("p");
      seen.textContent = "The reference picture, as the vision model saw it: " + record.reference;
      drawn.append(seen);
    }
    figure.append(drawn);
  }
  figure.append(actions);
  return figure;
}

export function setBusy(busy) {
  sendButton.classList.toggle("stop", busy);
  sendButton.setAttribute("aria-label", busy ? "Stop" : "Send");
  messageBox.disabled = busy;
  attachButton.disabled = busy;
  if (busy) closeMenu();
  modelSelect.disabled = busy || !modelSelect.value;
  for (const input of [pictureSize, pictureSeed, pictureSteps, pictureToggle]) input.disabled = busy;
  sidebar.toggleAttribute("inert", busy);
  document.dispatchEvent(new Event("busychange"));
}
