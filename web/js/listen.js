// Answers read aloud by Kokoro: the Listen button under each answer, the player, and the
// Reading aloud settings. The helper makes the sound a line at a time (/api/speak); each
// line is played as soon as it comes, so the first one is heard while the rest are made.
// One answer is read at a time: Listen on another stops the first.
import { status } from "./core.js";
import { answerTokens } from "./markdown.js";
import { answerText, sentences, spokenLines } from "./spoken.js";

export const VOICE_KEY = "hikarilm-voice";
export const SPEED_KEY = "hikarilm-voice-speed";
export const AUTO_KEY = "hikarilm-read-aloud";
const SAMPLE = "Hello! This is how answers sound when they are read aloud. The square root of 4761 is 69.";

let voice = null; // what /api/voice said: downloaded, ready, voices, default, get
let current = null; // the reading under way

export async function loadVoice() {
  try {
    voice = await (await fetch("/api/voice")).json();
  } catch (error) {
    voice = null;
  }
  for (const button of document.querySelectorAll(".listen-button")) button.hidden = !canListen();
  return voice;
}

export const canListen = () => Boolean(voice?.downloaded && voice?.ready && voice.voices?.length);
const chosenVoice = () => {
  const saved = localStorage.getItem(VOICE_KEY);
  return voice?.voices?.includes(saved) ? saved : voice?.default;
};
const chosenSpeed = () => Number(localStorage.getItem(SPEED_KEY)) || 1;
export const readsAloud = () => localStorage.getItem(AUTO_KEY) === "on" && canListen();

// What an answer says: its words (thinking is kept apart, never read), with the plots the
// Python tool drew named first.
export function linesFor(message) {
  const text = answerText(message?.content);
  const plots = (message?.tools || []).reduce((count, call) => count + (call.pictures?.length || 0), 0);
  const tokens = answerTokens(text);
  if (tokens) return spokenLines(tokens, { plots });
  return [...(plots ? ["There is a plot above."] : []), ...text.split(/\n+/).flatMap(sentences)];
}

function setButton(button, reading) {
  if (!button) return;
  button.textContent = reading ? "Stop" : button.dataset.idle || "Listen";
  button.setAttribute("aria-pressed", String(reading));
  if (!button.dataset.idle) button.title = reading ? "Stop reading aloud" : "Read this answer aloud";
}

export function stopListening() {
  const reading = current;
  if (!reading) return;
  current = null;
  reading.controller.abort();
  for (const source of reading.sources) {
    try {
      source.stop();
    } catch (error) {}
  }
  reading.context.close().catch(() => {});
  setButton(reading.button, false);
}

function play(reading, event) {
  const bytes = atob(event.audio);
  const samples = new Float32Array(bytes.length >> 1);
  for (let i = 0; i < samples.length; i++) {
    const value = bytes.charCodeAt(2 * i) | (bytes.charCodeAt(2 * i + 1) << 8);
    samples[i] = (value > 32767 ? value - 65536 : value) / 32768;
  }
  if (!samples.length) return;
  const buffer = reading.context.createBuffer(1, samples.length, event.rate || 24000);
  buffer.copyToChannel(samples, 0);
  const source = reading.context.createBufferSource();
  source.buffer = buffer;
  source.connect(reading.context.destination);
  const at = Math.max(reading.context.currentTime + 0.05, reading.next);
  source.start(at);
  reading.next = at + buffer.duration;
  reading.sources.add(source);
  source.onended = () => {
    reading.sources.delete(source);
    if (reading.finished && !reading.sources.size && current === reading) stopListening();
  };
}

// Reads lines aloud. The AudioContext is made here, in the click (or the send) that asked
// for it, since browsers only let a page play sound after the user did something.
export async function listen(lines, button = null) {
  stopListening();
  if (!canListen()) {
    status.textContent = voice?.downloaded ? "Reading aloud is not set up yet. Run  hikarilm restart  in Terminal."
      : "Reading aloud needs Kokoro. Get it with:  " + (voice?.get || "~/.hikarilm/venv/bin/hf download mlx-community/Kokoro-82M-bf16");
    return;
  }
  if (!lines.length) {
    status.textContent = "That answer has no words to read aloud.";
    return;
  }
  const reading = { controller: new AbortController(), context: new AudioContext(), sources: new Set(), button, next: 0, finished: false };
  current = reading;
  setButton(button, true);
  try {
    const response = await fetch("/api/speak", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ lines, voice: chosenVoice(), speed: chosenSpeed() }),
      signal: reading.controller.signal,
    });
    if (!response.ok) throw new Error((await response.json().catch(() => ({}))).error || "the helper could not read it");
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let waiting = "";
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      waiting += decoder.decode(value, { stream: true });
      let end;
      while ((end = waiting.indexOf("\n")) >= 0) {
        const line = waiting.slice(0, end).trim();
        waiting = waiting.slice(end + 1);
        if (!line) continue;
        const event = JSON.parse(line);
        if (event.error) throw new Error(event.error);
        if (event.audio && current === reading) play(reading, event);
      }
    }
    reading.finished = true;
    if (!reading.sources.size && current === reading) stopListening();
  } catch (error) {
    if (error.name === "AbortError" || current !== reading) return;
    stopListening();
    status.textContent = "Could not read aloud: " + error.message;
  }
}

// The Listen button under an answer. Hidden until Kokoro is ready.
export function addListenButton(block, message) {
  if (!String(message?.content || "").trim()) return;
  const row = document.createElement("div");
  row.className = "answer-actions";
  const button = document.createElement("button");
  button.type = "button";
  button.className = "listen-button";
  setButton(button, false);
  button.hidden = !canListen();
  button.addEventListener("click", () => {
    if (current?.button === button) stopListening();
    else listen(linesFor(message), button);
  });
  row.append(button);
  block.append(row);
  return button;
}

// After a reply, when Settings says to read answers aloud.
export function readWhenDone(button, message) {
  if (readsAloud() && button) listen(linesFor(message), button);
}

function showVoiceSetting() {
  const select = document.querySelector("#voice-choice");
  const note = document.querySelector("#voice-note");
  const ready = canListen();
  for (const id of ["#voice-choice", "#voice-speed", "#read-toggle", "#voice-sample"]) {
    document.querySelector(id).disabled = !ready;
  }
  if (!voice) note.textContent = "Could not reach the helper.";
  else if (!voice.downloaded) note.textContent = "Get Kokoro (0.4 GB) to have answers read aloud:  " + voice.get + "  then run  hikarilm restart.";
  else if (!voice.ready) note.textContent = "Kokoro is downloaded. Run  hikarilm restart  in Terminal to set up reading aloud.";
  else note.textContent = "Kokoro runs here, beside the chat model. Code is named rather than read, math is said in words, and thinking is never read.";
  const voices = voice?.voices || [];
  select.replaceChildren(...(voices.length ? voices.map((name) => new Option(voiceLabel(name), name)) : [new Option("None", "")]));
  select.value = chosenVoice() || "";
  document.querySelector("#voice-speed").value = String(chosenSpeed());
  document.querySelector("#voice-speed-value").textContent = chosenSpeed().toFixed(1) + "×";
  document.querySelector("#read-toggle").checked = localStorage.getItem(AUTO_KEY) === "on";
}

// af_heart is "Heart (American, female)".
export function voiceLabel(name) {
  const [kind, ...rest] = String(name).split("_");
  const accent = { a: "American", b: "British" }[kind?.[0]] || "";
  const who = { f: "female", m: "male" }[kind?.[1]] || "";
  const word = rest.join(" ");
  return (word ? word[0].toUpperCase() + word.slice(1) : name) + (accent || who ? " (" + [accent, who].filter(Boolean).join(", ") + ")" : "");
}

// Listeners and first steps, run by main.js once every part is loaded.
export function start() {
  loadVoice();
  document.querySelector("#open-settings").addEventListener("click", () => loadVoice().then(showVoiceSetting));
  document.querySelector("#voice-choice").addEventListener("change", (event) => localStorage.setItem(VOICE_KEY, event.target.value));
  document.querySelector("#voice-speed").addEventListener("input", (event) => {
    localStorage.setItem(SPEED_KEY, event.target.value);
    document.querySelector("#voice-speed-value").textContent = Number(event.target.value).toFixed(1) + "×";
  });
  document.querySelector("#read-toggle").addEventListener("change", (event) => {
    localStorage.setItem(AUTO_KEY, event.target.checked ? "on" : "off");
  });
  const sample = document.querySelector("#voice-sample");
  sample.addEventListener("click", () => {
    if (current?.button === sample) stopListening();
    else listen(sentences(SAMPLE), sample);
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && current && !document.querySelector("dialog[open]")) stopListening();
  });
}
