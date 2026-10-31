// One message on screen: its text, thinking, files, pictures and Apply buttons, and the busy state.
import { state } from "./state.js";
import { log, messageBox, modelSelect, scrollLog, sendButton, showEmpty, sidebar } from "./core.js";
import { attachButton } from "./attachments.js";
import { renderMarkdown, showThought, splitStrayThinking } from "./markdown.js";
import { pictureLabel, pictureSeed, pictureSize, pictureSteps, pictureToggle } from "./models.js";
import { closeMenu } from "./chat-list.js";
import { varyPicture } from "./send.js";
import { applyCode, syncApplyButtons } from "./apply.js";

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
  const answer = document.createElement("div");
  if (role === "assistant" && text) renderMarkdown(answer, text);
  else answer.textContent = text;
  if (message?.picture) answer.append(pictureFigure(message));
  block.append(who, thinking, box, answer);
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
  } else if (text) {
    updateApply(answer, question);
  }
  log.append(block);
  scrollLog();
  return { block, thinking, thought: view, answer };
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
  caption.textContent = [record.width && record.width + " × " + record.height, "seed " + record.seed, pictureLabel(record.model)]
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
  figure.append(img, caption, actions);
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
