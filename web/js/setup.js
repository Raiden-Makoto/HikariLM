// The picture setup page (/setup/pictures): the steps, and where they stand on this Mac.

function choiceBox(choice, have) {
  const item = document.createElement("div");
  item.className = "setup-choice" + (have ? " have" : "");
  const head = document.createElement("div");
  head.className = "setup-choice-head";
  const name = document.createElement("strong");
  name.textContent = choice.name;
  const size = document.createElement("span");
  size.textContent = have ? "Downloaded" : choice.download + " download";
  head.append(name, size);
  const note = document.createElement("p");
  note.textContent = choice.note;
  const line = document.createElement("div");
  line.className = "setup-command";
  const command = document.createElement("code");
  command.textContent = choice.command;
  const copy = document.createElement("button");
  copy.type = "button";
  copy.textContent = "Copy";
  copy.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(choice.command);
      copy.textContent = "Copied";
    } catch (error) {
      window.getSelection().selectAllChildren(command);
      copy.textContent = "Press Cmd+C";
    }
    setTimeout(() => { copy.textContent = "Copy"; }, 2000);
  });
  line.append(command, copy);
  item.append(head, note, line);
  return item;
}

function mark(step, done) {
  document.querySelector(step).classList.toggle("done", done);
}

let timer = null;

async function check() {
  const text = document.querySelector("#standing-text");
  const standing = document.querySelector("#standing");
  let body;
  try {
    body = await (await fetch("/api/pictures")).json();
  } catch (error) {
    text.textContent = "Could not reach the HikariLM helper. Is it running? hikarilm start starts it.";
    return;
  }
  document.querySelector("#checked").textContent = "Checked at " + new Date().toLocaleTimeString([], { hour: "numeric", minute: "2-digit", second: "2-digit" }) + ".";
  document.querySelector("#choices").replaceChildren(
    ...body.choices.map((choice) => choiceBox(choice, body.ids.includes(choice.repo))));
  const downloaded = body.models.length > 0;
  const ready = downloaded && body.ready;
  mark("#step-download", downloaded);
  mark("#step-start", ready);
  standing.classList.toggle("ready", ready);
  const parts = [];
  if (ready) parts.push("Ready. " + body.models.join(", ") + (body.models.length > 1 ? " are" : " is") + " set up.");
  else if (downloaded) parts.push("Downloaded: " + body.models.join(", ") + ". Next, step 2: run  hikarilm restart  to install mflux.");
  else parts.push("Nothing downloaded yet. Start with step 1. This page notices the download by itself.");
  parts.push(...body.notes);
  text.textContent = parts.join(" ");
  const go = document.querySelector("#start-drawing");
  go.hidden = !ready;
  go.dataset.model = ready ? body.ids[0] : "";
  // Keep looking until it is done, so the page follows along with the download.
  clearTimeout(timer);
  if (!ready) timer = setTimeout(check, 5000);
}

document.querySelector("#check").addEventListener("click", check);
document.querySelector("#start-drawing").addEventListener("click", () => {
  const model = document.querySelector("#start-drawing").dataset.model;
  if (model) localStorage.setItem("hikarilm-model", model);
});
document.addEventListener("visibilitychange", () => {
  if (document.visibilityState === "visible") check();
});
check();
