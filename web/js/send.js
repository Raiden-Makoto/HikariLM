// Sending a message and streaming the reply, and making, describing and varying pictures.
import { state } from "./state.js";
import { messageBox, modelSelect, plainError, scrollLog, showEmpty, status } from "./core.js";
import { renderChips } from "./attachments.js";
import { notify } from "./text-files.js";
import { finishMarkdown, showThought, streamMarkdown } from "./markdown.js";
import { addTurn, pictureFigure, setBusy, updateApply } from "./turns.js";
import {
  changeTarget, clearRewrite, inCodeView, isPictureModel, isVision, loadModels, modelLabel, pictureLabel, pictureSeed,
  pictureSize, pictureSteps, refreshState, rewriteModel, updateChangeNote, visionModel,
} from "./models.js";
import { saveChat } from "./chat-list.js";
import { draftKey, dropDraft, flushDraft, saveDraft, scheduleDraft } from "./chats.js";
import { announce, askNotifyPermission } from "./page.js";
import { notesOn } from "./settings.js";
import { editorPrompt } from "./edit-blocks.js";
import { findLora, loraRequest } from "./loras.js";
import { rulesRequest, rulesWarning } from "./rules.js";

// Ask the helper to have a vision model write one picture down as text. It
// answers from the saved description when it has one, so the vision model
// is not loaded again for a picture it has already read.
async function describePicture(file, model, signal, onText, purpose = "code", onLoaded = () => {}) {
  const response = await fetch("/api/describe", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model, image: file.image, name: file.name, purpose }),
    signal,
  });
  if (!response.ok) {
    const failure = await response.json().catch(() => ({}));
    throw new Error(failure.error || "The picture could not be read.");
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop();
    for (const line of lines) {
      if (!line.trim()) continue;
      const event = JSON.parse(line);
      if (event.error) throw new Error(event.error);
      if (event.hikarilm === "loaded") onLoaded(event);
      if (event.text) onText(event.text);
      if (typeof event.description === "string") {
        return { text: event.description, cached: Boolean(event.cached), model: event.model || model };
      }
    }
  }
  throw new Error("The picture was not described.");
}

async function findNotes(text, signal) {
  if (!notesOn() || inCodeView() || text.length < 15) return [];
  try {
    const response = await fetch("/api/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: text, exclude: state.chatId || "" }),
      signal,
    });
    if (!response.ok) return [];
    return (await response.json()).notes || [];
  } catch (error) {
    if (error.name === "AbortError") throw error;
    return [];
  }
}

async function send() {
  const content = messageBox.value.trim();
  if (!content || state.activeRequest) return;
  const model = modelSelect.value;
  if (!model) {
    loadModels();
    return;
  }
  if (isPictureModel(model)) return sendPicture(content, model);
  if (inCodeView() && state.attached.some((file) => file.image)) {
    notify("The code editor cannot send pictures. Remove the picture, or use the Chat tab.");
    return;
  }
  // A model that cannot see (Qwen, R1) is given the picture as text: a
  // vision model writes it down first, once, and that text goes with the message.
  let describer = "";
  if (state.attached.some((file) => file.image) && !isVision(model)) {
    describer = visionModel();
    if (!describer) {
      notify("Pictures need a model that can read them, such as Gemma 3 or Qwen2.5-VL. Remove the picture or download one.");
      return;
    }
  }
  const draftOwner = draftKey();
  askNotifyPermission();
  if (!state.chatId) state.chatId = (inCodeView() ? "editor-" : "") + crypto.randomUUID();
  const thisChat = state.chatId;
  let outcome = "";
  let failure = "";
  const files = state.attached;
  state.attached = [];
  renderChips();
  const asked = { role: "user", content, files: files.length ? files : undefined };
  state.messages.push(asked);
  const sent = addTurn("user", content, "", files, asked);
  messageBox.value = "";
  messageBox.style.height = "";
  dropDraft(draftOwner);
  let restored = false;
  const reply = addTurn("assistant", "");
  setBusy(true);
  status.textContent = "Streaming from " + modelLabel(model) + "…";
  state.activeRequest = new AbortController();
  let saved = false;
  let answer = "";
  let thought = "";
  let usage;
  let thinkStart = 0;
  let thinkEnd = 0;
  let answerStart = 0;
  let thinkingCut = false;
  const thinkSeconds = () => (thinkStart ? ((thinkEnd || Date.now()) - thinkStart) / 1000 : null);
  const drawThought = () => showThought(reply.thought, thought,
    { seconds: thinkSeconds(), live: Boolean(thinkStart && !thinkEnd), cut: thinkingCut });
  // The seconds keep counting while the model thinks, even between tokens.
  const thinkClock = setInterval(() => { if (thinkStart && !thinkEnd) drawThought(); }, 1000);
  state.lastContext = null;
  try {
    // A few earlier notes close to the question go along with it, just for
    // this send. They are not saved into the chat.
    let notesPrefix = "";
    const found = await findNotes(content, state.activeRequest.signal);
    if (found.length) {
      notesPrefix = "Notes from earlier chats and pictures (use them only if they help):\n" +
        found.map((note) => "[" + note.title + "]\n" + note.text.slice(0, 800)).join("\n\n") + "\n\nMy question:\n";
      const used = document.createElement("div");
      used.className = "used-notes";
      used.textContent = "Used earlier notes: " + found.map((note) => note.title).join(" · ");
      sent.block.append(used);
    }
    if (describer) {
      const converted = [];
      for (const file of files) {
        if (!file.image) {
          converted.push(file);
          continue;
        }
        status.textContent = "Reading " + file.name + " with " + modelLabel(describer) + "…";
        let live = "";
        const result = await describePicture(file, describer, state.activeRequest.signal, (piece) => {
          live += piece;
          reply.thinking.textContent = live;
          reply.thinking.hidden = false;
          scrollLog();
        }, "code", (event) => {
          status.textContent = "Loaded: " + modelLabel(event.model) +
            (event.unloaded ? ", after unloading " + modelLabel(event.unloaded) : "") + ". Reading " + file.name + "…";
        });
        converted.push({ name: file.name + " (read by " + modelLabel(result.model || describer) + ")", text: result.text });
      }
      reply.thinking.textContent = "";
      reply.thinking.hidden = true;
      state.messages[state.messages.length - 1].files = converted;
      status.textContent = "Streaming from " + modelLabel(model) + "…";
    }
    // In the code editor the model is told how to write code for a file, so Apply knows where it goes.
    const system = inCodeView() ? [{ role: "system", content: editorPrompt(state.folderName, state.openPath) }] : [];
    const rules = await rulesRequest(inCodeView());
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model,
        // The helper puts your rules (and the folder's) first in the system message.
        rules,
        // What the documents are searched with, without the earlier notes in front.
        question: content,
        messages: [...system, ...state.messages.map((message, index) => {
          const out = {
            role: message.role,
            // A chat model cannot see a picture made earlier, only that it was made.
            content: message.picture
              ? "[A picture was made here from the prompt above, " + message.picture.width + " × " + message.picture.height + ".]"
              : (index === state.messages.length - 1 ? notesPrefix : "") + message.content,
          };
          // The helper fits file text into the context: whole, as notes, or the closest passages.
          const texts = (message.files || []).filter((file) => typeof file.text === "string");
          if (texts.length && !message.picture) out.files = texts.map((file) => ({ name: file.name, text: file.text }));
          const pictures = (message.files || []).filter((file) => file.image).map((file) => file.image);
          if (pictures.length && index === state.messages.length - 1) out.images = pictures;
          // The helper needs these to compact the way Open WebUI does.
          if (message.contextSummary) out.contextSummary = message.contextSummary;
          if (message.usage) out.usage = message.usage;
          return out;
        })],
      }),
      signal: state.activeRequest.signal,
    });
    if (!response.ok) {
      const failure = await response.json();
      throw new Error(failure.error || "chat failed");
    }
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop();
      for (const line of lines) {
        if (!line.trim()) continue;
        const event = JSON.parse(line);
        if (event.error) throw new Error(event.error);
        if (event.hikarilm === "context") {
          state.lastContext = event;
          continue;
        }
        if (event.hikarilm === "rules") {
          const warning = rulesWarning(event);
          if (warning) notify(warning);
          continue;
        }
        if (event.hikarilm === "compaction") {
          if (!event.done) status.textContent = "Compacting context…";
          else if (event.error) status.textContent = "Context compaction failed, so the chat is sent as it is: " + plainError(event.error);
          else {
            // The summary is kept on the first message that stays, as a checkpoint.
            const kept = state.messages[event.checkpoint - system.length];
            if (kept) kept.contextSummary = event.summary;
            status.textContent = "Context compacted: " + event.dropped + " older messages summarized. Streaming from " + modelLabel(model) + "…";
          }
          continue;
        }
        if (event.hikarilm === "reading") {
          status.textContent = event.error
            ? "Reading the document failed, so the closest passages are sent instead: " + plainError(event.error)
            : "Reading " + event.name + ", part " + event.part + " of " + event.parts + "…";
          continue;
        }
        if (event.hikarilm === "documents") {
          const used = document.createElement("div");
          used.className = "used-notes";
          used.textContent = event.mode === "notes"
            ? "Read all of " + event.files.map((file) => file.name + " (" + file.parts + (file.parts === 1 ? " part" : " parts") + ")").join(", ") + ", then answered from notes"
            : "Too long to send whole, so the closest passages were sent: " +
              event.files.map((file) => file.name + " (" + file.used + " of " + file.total + ")").join(", ") +
              (event.search === "words" ? ", matched by words since nomic-embed-text is not installed" : "");
          sent.block.append(used);
          status.textContent = "Streaming from " + modelLabel(model) + "…";
          scrollLog();
          continue;
        }
        if (event.hikarilm === "answer_was_thinking") {
          // A </think> arrived, so what streamed as the answer was the model's reasoning.
          thought += answer;
          answer = "";
          thinkStart ||= answerStart || Date.now();
          thinkEnd = 0;
          finishMarkdown(reply.answer, "");
          drawThought();
          continue;
        }
        if (event.hikarilm === "thinking_cut") {
          thinkingCut = true;
          if (!event.final) status.textContent = "Thinking reached " + event.budget.toLocaleString() + " tokens, so " + modelLabel(model) + " is answering now…";
          drawThought();
          continue;
        }
        if (event.done) usage = { prompt_eval_count: event.prompt_eval_count || 0, eval_count: event.eval_count || 0 };
        const reasoning = event.message?.thinking || "";
        const piece = event.message?.content || "";
        if (reasoning) {
          thinkStart ||= Date.now();
          thought += reasoning;
        }
        if (piece) {
          answerStart ||= Date.now();
          if (thinkStart && !thinkEnd && piece.trim()) thinkEnd = Date.now();
          answer += piece;
          streamMarkdown(reply.answer, answer);
        }
        if (reasoning || piece) drawThought();
        scrollLog();
      }
    }
    if (!answer && !thought) {
      state.messages.pop();
      outcome = "failed";
      failure = "The model sent no reply.";
    } else {
      outcome = "done";
    }
  } catch (error) {
    if (error.name === "AbortError" && !answer && !thought) {
      sent.block.remove();
      reply.block.remove();
      messageBox.value = content;
      state.attached = files;
      renderChips();
      restored = true;
      if (state.messages.length <= 1) showEmpty();
    }
    if (error.name !== "AbortError") {
      outcome = "failed";
      failure = plainError(error.message);
      if (answer || thought) {
        const note = document.createElement("div");
        note.className = "thinking";
        note.textContent = "The reply was cut off: " + plainError(error.message);
        reply.answer.after(note);
      } else {
        reply.answer.textContent = plainError(error.message);
      }
    }
    if (!answer && !thought) state.messages.pop();
  } finally {
    clearInterval(thinkClock);
    if (thinkStart && !thinkEnd) thinkEnd = Date.now();
    if (answer) finishMarkdown(reply.answer, answer);
    if (thought) drawThought();
    if (answer || thought) {
      if (answer) updateApply(reply.answer, { text: content, files });
      state.messages.push({
        role: "assistant", content: answer, thinking: thought || undefined, model, usage,
        thoughtSeconds: thought ? Math.round(thinkSeconds()) : undefined, thinkingCut: thinkingCut || undefined,
      });
      if (usage && state.lastContext) state.lastContext.used = usage.prompt_eval_count + usage.eval_count;
      saved = true;
    }
    state.activeRequest = null;
    if (state.messages.length === 0) state.chatId = null;
    if (restored) saveDraft(draftKey(), true);
    setBusy(false);
    refreshState();
    messageBox.focus();
  }
  if (saved) await saveChat();
  const name = state.chats.find((chat) => chat.id === thisChat)?.title || "";
  if (outcome === "done") announce("Reply ready", name, answer || thought, thisChat);
  else if (outcome === "failed") announce("Reply stopped", name, failure, thisChat);
}

// Ask the helper for one picture. It first moves the chat model (or the
// last picture model) out of memory; show() is told each step.
async function makePicture(payload, signal, show) {
  const lora = payload.lora ? findLora(payload.lora) : null;
  const label = pictureLabel(payload.model) + (lora ? " with the LoRA " + lora.label : "");
  show("Loading " + label + "…");
  const response = await fetch("/api/generate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
    signal,
  });
  if (!response.ok) throw new Error((await response.json()).error || "The picture could not be made.");
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffered = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffered += decoder.decode(value, { stream: true });
    const lines = buffered.split("\n");
    buffered = lines.pop();
    for (const line of lines) {
      if (!line.trim()) continue;
      const event = JSON.parse(line);
      if (event.error) throw new Error(event.error);
      if (event.image) return event.image;
      if (event.hikarilm === "loading") {
        show("Loading " + label + (event.unloading ? ", after unloading " + event.unloading : "") + "…");
      } else if (event.progress < 0) show("Saving the picture…");
      else if (event.total) show("Drawing… step " + event.progress + " of " + event.total);
      else if (event.progress !== undefined) show("Drawing…");
    }
  }
  throw new Error("The helper stopped before the picture was finished.");
}

function pictureRequest(prompt, model, request, reference, pictures) {
  const number = (input) => (input.value.trim() === "" ? null : Number(input.value));
  return { model, prompt, request: request || undefined, reference: reference || undefined,
    references: pictures.length ? pictures : undefined, size: pictureSize.value || "square",
    seed: number(pictureSeed), steps: number(pictureSteps), ...loraRequest() };
}

// A picture drawn with a LoRA is varied or changed with it again, if it is still ready.
function sameLora(picture) {
  return picture.lora && findLora(picture.lora.id) ? { lora: picture.lora.id, lora_strength: picture.lora.strength } : {};
}

function fitBox() {
  messageBox.style.height = "auto";
  messageBox.style.height = Math.min(messageBox.scrollHeight, 160) + "px";
}

async function readStream(response, onEvent) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) return null;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop();
    for (const line of lines) {
      if (!line.trim()) continue;
      const event = JSON.parse(line);
      if (event.error) throw new Error(event.error);
      const result = onEvent(event);
      if (result !== undefined) return result;
    }
  }
}

// A picture model is asked in words a chat model wrote: what a name looks like, and
// what an attached reference picture shows. The rewrite goes into the box to be read,
// edited or undone; the next send draws it. `asked` is what the user first wrote, when
// `request` is an earlier rewrite being redone for a picture attached since.
async function rewritePrompt(request, writer, describer, references, asked = request) {
  const note = document.querySelector("#rewrite-note");
  const noteText = document.querySelector("#rewrite-text");
  const undo = document.querySelector("#rewrite-undo");
  const seenBox = document.querySelector("#rewrite-seen");
  const say = (text) => {
    noteText.textContent = text;
    undo.hidden = true;
    note.hidden = false;
    status.textContent = text;
  };
  const loadedLine = (event) => "Loaded: " + modelLabel(event.model) +
    (event.unloaded ? ", after unloading " + (isPictureModel(event.unloaded) ? pictureLabel(event.unloaded) : modelLabel(event.unloaded)) : "");
  // What the vision model saw stays on the note, so a rewrite that ignored it shows.
  const showSeen = (title, text, open) => {
    seenBox.querySelector("summary").textContent = title;
    seenBox.querySelector("p").textContent = text;
    seenBox.open = open;
    seenBox.hidden = false;
  };
  seenBox.hidden = true;
  setBusy(true);
  state.activeRequest = new AbortController();
  state.lastContext = null;
  const signal = state.activeRequest.signal;
  let prompt = "";
  const seen = [];
  try {
    for (const file of describer ? references : []) {
      const reader = modelLabel(describer);
      say("Loading " + reader + " to read " + file.name + "…");
      let live = "";
      const result = await describePicture(file, describer, signal, (piece) => {
        live += piece;
        showSeen(reader + " is reading " + file.name, live, true);
      }, "look", (event) => say(loadedLine(event) + ". Reading " + file.name + "…"));
      seen.push((references.length > 1 ? file.name + ": " : "") + result.text);
      showSeen("What " + modelLabel(result.model || describer) + " saw in " +
        references.map((item) => item.name).slice(0, seen.length).join(", ") +
        (result.cached ? " (read before, so not read again)" : ""), seen.join("\n\n"), false);
    }
    say("Loading " + modelLabel(writer) + " to rewrite the prompt…");
    const response = await fetch("/api/rewrite", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model: writer, request, reference: seen.join("\n\n"), shown: references.length > 0 }),
      signal,
    });
    if (!response.ok) throw new Error((await response.json().catch(() => ({}))).error || "The prompt could not be rewritten.");
    let live = "";
    prompt = await readStream(response, (event) => {
      if (event.hikarilm === "loaded") say(loadedLine(event) + ". Rewriting the prompt…");
      if (event.text) {
        live += event.text;
        messageBox.value = live;
        fitBox();
      }
      if (typeof event.prompt === "string") return event.prompt;
    }) || "";
    if (!prompt) throw new Error("The helper stopped before the prompt was rewritten.");
  } catch (error) {
    note.hidden = true;
    seenBox.hidden = true;
    if (error.name !== "AbortError") notify("Not rewritten: " + plainError(error.message));
  } finally {
    state.activeRequest = null;
    setBusy(false);
  }
  messageBox.value = prompt || request;
  fitBox();
  if (prompt) {
    state.rewrite = { request: asked, writer, pictures: references.map((file) => file.image), seen: seen.join("\n\n") };
    noteText.textContent = "Rewritten by " + modelLabel(writer) + " from “" + asked + "”" +
      (references.length ? " and " + references.map((file) => file.name).join(", ") : "") +
      ". Edit it if you like, then send to draw.";
    undo.hidden = false;
    note.hidden = false;
    status.textContent = "";
  }
  saveDraft(draftKey(), true);
  refreshState().catch(() => {});
  messageBox.focus();
}

async function sendPicture(prompt, model) {
  if (state.attached.some((file) => !file.image)) {
    notify("A picture model takes reference pictures only. Remove the other files, or pick a chat model.");
    return;
  }
  // What is in the box is drawn; a rewrite happens only when Rewrite is pressed.
  const pending = state.rewrite;
  return drawPicture(prompt, model, pending && !pending.asIs ? pending.request : "");
}

// Rewrite, by Options: a chat model rewrites the box into how things look, and a model
// that can see reads the attached pictures for it first.
function rewriteBox() {
  const text = messageBox.value.trim();
  if (!text || state.activeRequest) return;
  const references = state.attached.filter((file) => file.image);
  const describer = references.length ? visionModel() : "";
  const writer = rewriteModel() || describer;
  if (!writer) {
    notify("No chat model to rewrite with. Pick one under Settings, Rewrite prompts with.");
    return;
  }
  const pending = state.rewrite;
  rewritePrompt(text, writer, describer, references, pending && !pending.asIs ? pending.request : text);
}

async function drawPicture(prompt, model, request) {
  const draftOwner = draftKey();
  askNotifyPermission();
  if (!state.chatId) state.chatId = crypto.randomUUID();
  const thisChat = state.chatId;
  const files = state.attached;
  const asked = request || prompt;
  const pending = state.rewrite;
  const target = changeTarget()?.picture;
  const choice = state.changeChoice;
  state.attached = [];
  state.changeChoice = "";
  renderChips();
  clearRewrite();
  const question = { role: "user", content: asked, files: files.length ? files : undefined };
  state.messages.push(question);
  const sent = addTurn("user", asked, "", files, question);
  messageBox.value = "";
  messageBox.style.height = "";
  dropDraft(draftOwner);
  const reply = addTurn("assistant", "");
  const show = (text) => {
    reply.thinking.textContent = text;
    reply.thinking.hidden = false;
    status.textContent = text;
    scrollLog();
  };
  setBusy(true);
  state.activeRequest = new AbortController();
  // The chat model's context figures stop describing what is loaded.
  state.lastContext = null;
  let record = null;
  let failure = "";
  try {
    const asking = pictureRequest(prompt, model, request, pending?.asIs ? "" : pending?.seen,
      files.filter((file) => file.image).map((file) => file.image));
    // A change keeps the picture's size, so the model is not stretching it.
    if (target) Object.assign(asking, { change: target.id, size: target.size });
    if (target?.lora) {
      delete asking.lora;
      delete asking.lora_strength;
      Object.assign(asking, sameLora(target));
    }
    record = await makePicture(asking, state.activeRequest.signal, show);
  } catch (error) {
    if (error.name !== "AbortError") failure = plainError(error.message);
  } finally {
    state.activeRequest = null;
    setBusy(false);
  }
  const name = () => state.chats.find((chat) => chat.id === thisChat)?.title || "";
  if (record) {
    const message = { role: "assistant", content: "", model, picture: record };
    state.messages.push(message);
    reply.block.remove();
    addTurn("assistant", "", "", null, message);
    await saveChat();
    announce("Picture ready", name(), prompt, thisChat);
  } else if (failure) {
    state.messages.pop();
    reply.thinking.hidden = true;
    reply.answer.textContent = failure;
    announce("Picture failed", name(), failure, thisChat);
  } else {
    // Stopped: nothing was made, so the prompt and pictures go back, still rewritten.
    state.messages.pop();
    sent.block.remove();
    reply.block.remove();
    messageBox.value = prompt;
    fitBox();
    state.attached = files;
    renderChips();
    state.rewrite = pending;
    state.changeChoice = choice;
    document.querySelector("#rewrite-note").hidden = !pending || Boolean(pending.asIs);
    saveDraft(draftKey(), true);
    if (state.messages.length === 0) showEmpty();
  }
  if (state.messages.length === 0) state.chatId = null;
  updateChangeNote();
  refreshState().catch(() => {});
  messageBox.focus();
}

// Vary keeps the prompt, size, steps and reference pictures and draws with a new seed.
// The new picture takes the old one's place, and the old file is deleted.
export async function varyPicture(message, figure) {
  if (state.activeRequest) return;
  const old = message.picture;
  if (!isPictureModel(old.model)) {
    notify(pictureLabel(old.model) + " is not set up for pictures any more, so this one cannot be varied.");
    return;
  }
  // The references are in the message that asked for the picture. Chats are saved
  // without their pictures, so after a reload they are gone.
  const asked = state.messages[state.messages.indexOf(message) - 1];
  const pictures = (asked?.files || []).filter((file) => file.image).map((file) => file.image);
  if ((old.references || []).length && pictures.length !== old.references.length) {
    notify("This picture was drawn from reference pictures, which are not kept after a reload. " +
      "Attach them again and send to draw a new one.");
    return;
  }
  const lora = sameLora(old);
  let prompt = old.prompt;
  if (old.lora && !lora.lora) {
    notify("The LoRA " + old.lora.label + " is not ready any more, so this one is drawn without it.");
    const front = old.lora.trigger ? old.lora.trigger + ", " : "";
    if (front && prompt.toLowerCase().startsWith(front.toLowerCase())) prompt = prompt.slice(front.length);
  }
  const line = document.createElement("div");
  line.className = "thinking";
  figure.before(line);
  const show = (text) => {
    line.textContent = text;
    status.textContent = text;
  };
  setBusy(true);
  state.activeRequest = new AbortController();
  state.lastContext = null;
  let record = null;
  try {
    record = await makePicture({ model: old.model, prompt, request: old.request, reference: old.reference,
      references: old.references?.length ? pictures : undefined, change: old.changed, size: old.size, steps: old.steps, seed: null, ...lora },
      state.activeRequest.signal, show);
  } catch (error) {
    if (error.name !== "AbortError") notify("No new picture: " + plainError(error.message));
  } finally {
    state.activeRequest = null;
    setBusy(false);
    line.remove();
  }
  if (record) {
    message.picture = record;
    figure.replaceWith(pictureFigure(message));
    await saveChat();
    fetch("/api/images/" + old.id, { method: "DELETE" }).catch(() => {});
  }
  refreshState().catch(() => {});
}

// Listeners and first steps, run by main.js once every part is loaded.
export function start() {
  document.querySelector("#composer").addEventListener("submit", (event) => {
    event.preventDefault();
    if (state.activeRequest) {
      state.activeRequest.abort();
      return;
    }
    send();
  });
  messageBox.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      send();
    }
  });
  messageBox.addEventListener("input", () => {
    fitBox();
    if (!messageBox.value.trim()) {
      clearRewrite();
      state.changeChoice = "";
    }
    updateChangeNote();
    scheduleDraft();
  });
  document.querySelector("#picture-rewrite").addEventListener("click", rewriteBox);
  document.querySelector("#change-toggle").addEventListener("click", () => {
    state.changeChoice = changeTarget() ? "new" : "change";
    updateChangeNote();
    messageBox.focus();
  });
  document.querySelector("#rewrite-undo").addEventListener("click", () => {
    if (!state.rewrite) return;
    messageBox.value = state.rewrite.request;
    fitBox();
    state.rewrite = { request: state.rewrite.request, asIs: true, pictures: state.rewrite.pictures };
    document.querySelector("#rewrite-note").hidden = true;
    saveDraft(draftKey(), true);
    messageBox.focus();
  });
  window.addEventListener("pagehide", () => flushDraft(true));
  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "hidden") flushDraft(true);
  });
}
