// Sending a message and streaming the reply, and making, describing and varying pictures.
import { state } from "./state.js";
import { messageBox, modelSelect, plainError, scrollLog, showEmpty, status } from "./core.js";
import { renderChips } from "./attachments.js";
import { isVision, notify, visionModel } from "./text-files.js";
import { finishMarkdown, showThought, streamMarkdown } from "./markdown.js";
import { addTurn, pictureFigure, setBusy, updateApply } from "./turns.js";
import {
  inCodeView, isPictureModel, loadModels, pictureLabel, pictureSeed, pictureSize,
  pictureSteps, refreshState,
} from "./models.js";
import { saveChat } from "./chat-list.js";
import { draftKey, dropDraft, flushDraft, saveDraft, scheduleDraft } from "./chats.js";
import { announce, askNotifyPermission } from "./page.js";
import { notesOn } from "./settings.js";

// Ask the helper to have a vision model write one picture down as text. It
// answers from the saved description when it has one, so the vision model
// is not loaded again for a picture it has already read.
async function describePicture(file, model, signal, onText) {
  const response = await fetch("/api/describe", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model, image: file.image, name: file.name }),
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
  // Gemma model writes it down first, once, and that text goes with the message.
  let describer = "";
  if (state.attached.some((file) => file.image) && !isVision(model)) {
    describer = visionModel();
    if (!describer) {
      notify("Images need a Gemma model. Remove the image or install gemma3.");
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
  state.messages.push({ role: "user", content, files: files.length ? files : undefined });
  const sent = addTurn("user", content, "", files);
  messageBox.value = "";
  messageBox.style.height = "";
  dropDraft(draftOwner);
  let restored = false;
  const reply = addTurn("assistant", "");
  setBusy(true);
  status.textContent = "Streaming from " + model + "…";
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
        status.textContent = "Reading " + file.name + " with " + describer + "…";
        let live = "";
        const result = await describePicture(file, describer, state.activeRequest.signal, (piece) => {
          live += piece;
          reply.thinking.textContent = live;
          reply.thinking.hidden = false;
          scrollLog();
        });
        converted.push({ name: file.name + " (read by " + (result.model || describer) + ")", text: result.text });
      }
      reply.thinking.textContent = "";
      reply.thinking.hidden = true;
      state.messages[state.messages.length - 1].files = converted;
      status.textContent = "Streaming from " + model + "…";
    }
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model,
        // What the documents are searched with, without the earlier notes in front.
        question: content,
        messages: state.messages.map((message, index) => {
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
        }),
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
        if (event.lightusagi === "context") {
          state.lastContext = event;
          if (event.clamped) {
            notify("Context cut to " + event.num_ctx.toLocaleString() + " tokens so " + model + " fits in this computer's memory.");
          }
          continue;
        }
        if (event.lightusagi === "compaction") {
          if (!event.done) status.textContent = "Compacting context…";
          else if (event.error) status.textContent = "Context compaction failed, so the chat is sent as it is: " + plainError(event.error);
          else {
            // The summary is kept on the first message that stays, as a checkpoint.
            if (state.messages[event.checkpoint]) state.messages[event.checkpoint].contextSummary = event.summary;
            status.textContent = "Context compacted: " + event.dropped + " older messages summarized. Streaming from " + model + "…";
          }
          continue;
        }
        if (event.lightusagi === "reading") {
          status.textContent = event.error
            ? "Reading the document failed, so the closest passages are sent instead: " + plainError(event.error)
            : "Reading " + event.name + ", part " + event.part + " of " + event.parts + "…";
          continue;
        }
        if (event.lightusagi === "documents") {
          const used = document.createElement("div");
          used.className = "used-notes";
          used.textContent = event.mode === "notes"
            ? "Read all of " + event.files.map((file) => file.name + " (" + file.parts + (file.parts === 1 ? " part" : " parts") + ")").join(", ") + ", then answered from notes"
            : "Too long to send whole, so the closest passages were sent: " +
              event.files.map((file) => file.name + " (" + file.used + " of " + file.total + ")").join(", ") +
              (event.search === "words" ? ", matched by words since nomic-embed-text is not installed" : "");
          sent.block.append(used);
          status.textContent = "Streaming from " + model + "…";
          scrollLog();
          continue;
        }
        if (event.lightusagi === "answer_was_thinking") {
          // A </think> arrived, so what streamed as the answer was the model's reasoning.
          thought += answer;
          answer = "";
          thinkStart ||= answerStart || Date.now();
          thinkEnd = 0;
          finishMarkdown(reply.answer, "");
          drawThought();
          continue;
        }
        if (event.lightusagi === "thinking_cut") {
          thinkingCut = true;
          if (!event.final) status.textContent = "Thinking reached " + event.budget.toLocaleString() + " tokens, so " + model + " is answering now…";
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
      if (answer) updateApply(reply.block, answer);
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
  const label = pictureLabel(payload.model);
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
      if (event.lightusagi === "loading") {
        show("Loading " + label + (event.unloading ? ", after unloading " + event.unloading : "") + "…");
      } else if (event.progress < 0) show("Saving the picture…");
      else if (event.total) show("Drawing… step " + event.progress + " of " + event.total);
      else if (event.progress !== undefined) show("Drawing…");
    }
  }
  throw new Error("The helper stopped before the picture was finished.");
}

function pictureRequest(prompt, model) {
  const number = (input) => (input.value.trim() === "" ? null : Number(input.value));
  return { model, prompt, size: pictureSize.value || "square", seed: number(pictureSeed), steps: number(pictureSteps) };
}

async function sendPicture(prompt, model) {
  if (state.attached.length) {
    notify("A picture model cannot read attached files. Remove them, or pick a chat model.");
    return;
  }
  const draftOwner = draftKey();
  askNotifyPermission();
  if (!state.chatId) state.chatId = crypto.randomUUID();
  const thisChat = state.chatId;
  state.messages.push({ role: "user", content: prompt });
  const sent = addTurn("user", prompt, "");
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
    record = await makePicture(pictureRequest(prompt, model), state.activeRequest.signal, show);
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
    // Stopped: nothing was made, so the prompt goes back into the box.
    state.messages.pop();
    sent.block.remove();
    reply.block.remove();
    messageBox.value = prompt;
    saveDraft(draftKey(), true);
    if (state.messages.length === 0) showEmpty();
  }
  if (state.messages.length === 0) state.chatId = null;
  refreshState().catch(() => {});
  messageBox.focus();
}

// Vary keeps the prompt, size and steps and draws with a new seed. The new
// picture takes the old one's place, and the old file is deleted.
export async function varyPicture(message, figure) {
  if (state.activeRequest) return;
  const old = message.picture;
  if (!isPictureModel(old.model)) {
    notify(pictureLabel(old.model) + " is not set up for pictures any more, so this one cannot be varied.");
    return;
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
    record = await makePicture({ model: old.model, prompt: old.prompt, size: old.size, steps: old.steps, seed: null },
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
    messageBox.style.height = "auto";
    messageBox.style.height = Math.min(messageBox.scrollHeight, 160) + "px";
    scheduleDraft();
  });
  window.addEventListener("pagehide", () => flushDraft(true));
  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "hidden") flushDraft(true);
  });
}
