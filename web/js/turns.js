// One message on screen: its text, thinking, files, pictures and Apply buttons, and the busy state.
import { state } from "./state.js";
import { log, messageBox, modelSelect, scrollLog, sendButton, showEmpty, sidebar } from "./core.js";
import { attachButton } from "./attachments.js";
import { renderMarkdown, showThought, splitStrayThinking } from "./markdown.js";
import { pictureLabel, pictureSeed, pictureSize, pictureSteps, pictureToggle } from "./models.js";
import { closeMenu } from "./chat-list.js";
import { varyPicture } from "./send.js";
import { applyBlock } from "./file-actions.js";

export function addTurn(role, text, thought, files, message) {
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
    updateApply(block, text);
  }
  log.append(block);
  scrollLog();
  return { block, thinking, thought: view, answer };
}

// Complete code blocks (``` ... ```) in a reply. In the code editor each one
// gets an Apply button. This is only the page copying text into the editor,
// so it costs the model nothing.
function codeBlocks(text) {
  const found = [];
  const fence = /^```[^\n]*\n([\s\S]*?)\n?```[ \t]*$/gm;
  let match;
  while ((match = fence.exec(text))) found.push(match[1]);
  return found;
}

export function updateApply(block, text) {
  block.querySelector(".apply-row")?.remove();
  const blocks = codeBlocks(text);
  if (!blocks.length) return;
  const row = document.createElement("div");
  row.className = "apply-row";
  blocks.forEach((code, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "choice";
    button.textContent = blocks.length === 1 ? "Apply code" : "Apply code block " + (index + 1);
    button.title = "Put this code into the open file as an unsaved edit";
    button.addEventListener("click", () => applyBlock(code));
    row.append(button);
  });
  block.append(row);
}

export function renderLog() {
  if (state.messages.length === 0) {
    showEmpty();
    return;
  }
  log.replaceChildren();
  for (const message of state.messages) addTurn(message.role, message.content, message.thinking, message.files, message);
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
  download.download = "lightusagi-" + record.id + ".png";
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
