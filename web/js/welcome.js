// The welcome page (/welcome): what HikariLM is, and the models to start with, marked when downloaded.

import { copyButton } from "./copy.js";

function modelBox(model) {
  const item = document.createElement("div");
  item.className = "setup-choice" + (model.have ? " have" : "");
  const head = document.createElement("div");
  head.className = "setup-choice-head";
  const name = document.createElement("strong");
  name.textContent = model.name + " · " + model.for;
  const size = document.createElement("span");
  size.textContent = model.have ? "Downloaded" : model.download + " download";
  head.append(name, size);
  const note = document.createElement("p");
  note.textContent = model.note;
  item.append(head, note);
  if (!model.have) {
    const line = document.createElement("div");
    line.className = "setup-command";
    const command = document.createElement("code");
    command.textContent = model.command;
    line.append(command, copyButton(() => model.command, command));
    item.append(line);
  }
  return item;
}

let timer = null;

async function check() {
  const text = document.querySelector("#standing-text");
  let body;
  try {
    body = await (await fetch("/api/welcome")).json();
  } catch (error) {
    text.textContent = "Could not reach the HikariLM helper. Is it running? hikarilm start starts it.";
    return;
  }
  document.querySelector("#checked").textContent = "Checked at " +
    new Date().toLocaleTimeString([], { hour: "numeric", minute: "2-digit", second: "2-digit" }) + ". This page notices a download by itself.";
  const groups = body.groups.map((group) => {
    const heading = document.createElement("h3");
    heading.textContent = group;
    return [heading, ...body.models.filter((model) => model.group === group).map(modelBox)];
  });
  document.querySelector("#models").replaceChildren(...groups.flat());
  document.querySelector("#standing").classList.toggle("ready", body.chat > 0);
  document.querySelector("#start-chatting").hidden = body.chat === 0;
  text.textContent = "HikariLM " + body.version + ". " + (body.chat
    ? "Ready: " + body.chat + (body.chat === 1 ? " chat model is" : " chat models are") + " downloaded."
    : "No chat model is downloaded yet: start with Qwen3 8B below.");
  // Keep looking while the page is open, so it follows along with a download.
  clearTimeout(timer);
  timer = setTimeout(check, 10000);
}

document.addEventListener("visibilitychange", () => {
  if (document.visibilityState === "visible") check();
});
check();
