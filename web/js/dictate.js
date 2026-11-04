// Speaking instead of typing: the microphone button by Attach. Click to start, click again to
// stop and have Whisper write it down on this Mac; Esc cancels. The words go into the message
// box where the cursor was, to read and fix; Send stays yours. The browser asks for the
// microphone the first time; when it is refused, the note says how to allow it.
//
// No ffmpeg: MediaRecorder records, Web Audio decodes it, and an OfflineAudioContext makes it
// 16 kHz mono, sent as 16-bit PCM to /api/transcribe, which streams back each part as written.
import { messageBox, status } from "./core.js";

export const WHISPER_KEY = "hikarilm-whisper";
export const LANGUAGE_KEY = "hikarilm-whisper-language";
const RATE = 16000;
// Whisper's languages people ask for most; Detect lets Whisper tell.
export const LANGUAGES = [["", "Detect"], ["en", "English"], ["es", "Spanish"], ["fr", "French"], ["de", "German"],
  ["it", "Italian"], ["pt", "Portuguese"], ["nl", "Dutch"], ["pl", "Polish"], ["tr", "Turkish"], ["ru", "Russian"],
  ["uk", "Ukrainian"], ["ar", "Arabic"], ["hi", "Hindi"], ["ja", "Japanese"], ["ko", "Korean"], ["zh", "Chinese"],
  ["vi", "Vietnamese"], ["id", "Indonesian"], ["sv", "Swedish"], ["he", "Hebrew"], ["el", "Greek"]];

const button = document.querySelector("#mic");
const meter = document.querySelector("#mic-state");
const level = document.querySelector("#mic-level-bar");
const clock = document.querySelector("#mic-time");
const note = document.querySelector("#mic-note");
const noteText = document.querySelector("#mic-text");

let voice = null;      // /api/voice
let session = null;    // the recording or the writing down under way

export async function loadWhisper() {
  try {
    voice = await (await fetch("/api/voice")).json();
  } catch (error) {
    voice = null;
  }
  const whispers = voice?.whisper || [];
  button.title = whispers.length ? "Speak instead of typing" : "Get a voice model…";
  button.classList.toggle("unready", !whispers.length || !voice?.ready);
  return voice;
}

export const whispers = () => voice?.whisper || [];
const chosenWhisper = () => {
  const saved = localStorage.getItem(WHISPER_KEY);
  return whispers().some((item) => item.id === saved) ? saved : whispers()[0]?.id || "";
};

function say(text, { command = "", link = "" } = {}) {
  noteText.replaceChildren(text);
  if (command) {
    const code = document.createElement("code");
    code.textContent = command;
    noteText.append(" ", code);
  }
  if (link) {
    const a = document.createElement("a");
    a.href = link;
    a.target = "_blank";
    a.rel = "noopener";
    a.textContent = "See which to get";
    noteText.append(" ", a);
  }
  note.hidden = false;
}
const quiet = () => { note.hidden = true; };

// How to let the page use the microphone again, for the browser it is in.
function howToAllow(system = false) {
  const agent = navigator.userAgent;
  const where = /Firefox\//.test(agent) ? "Click the microphone icon at the left of the address bar and remove the block, then reload."
    : /Edg\//.test(agent) || /Chrome\//.test(agent) ? "Click the icon at the left of the address bar, turn Microphone on, and reload."
    : /Safari\//.test(agent) ? "In Safari, choose Safari > Settings for 127.0.0.1, set Microphone to Allow, and reload."
    : "Allow the microphone for this page in the browser's site settings, and reload.";
  if (system) return "macOS has not let the browser use it: System Settings > Privacy & Security > Microphone, turn the browser on, then quit and reopen the browser.";
  return where + " If the browser is allowed, macOS may not be: System Settings > Privacy & Security > Microphone, turn the browser on.";
}

async function permission() {
  try {
    return (await navigator.permissions.query({ name: "microphone" })).state;
  } catch (error) {
    return "";
  }
}

function why(error) {
  switch (error?.name) {
    case "NotAllowedError":
    case "PermissionDeniedError":
      return "The microphone was not allowed. " + howToAllow(/system/i.test(error.message || ""));
    case "NotFoundError":
    case "OverconstrainedError":
      return "No microphone was found. Plug one in, or pick one in System Settings > Sound > Input.";
    case "NotReadableError":
    case "TrackStartError":
      return "The microphone could not be opened: another app may be using it, or macOS blocked it " +
        "(System Settings > Privacy & Security > Microphone).";
    case "SecurityError":
      return "The browser only allows the microphone on http://127.0.0.1 or https pages.";
    default:
      return "The microphone could not be opened: " + (error?.message || error);
  }
}

function setButton(state) {
  button.dataset.state = state;
  button.setAttribute("aria-pressed", String(state === "recording"));
  button.setAttribute("aria-label", state === "recording" ? "Stop and write it down" : state === "writing" ? "Stop writing it down" : "Speak instead of typing");
  button.title = button.getAttribute("aria-label") + (state === "recording" ? " (Esc cancels)" : "");
  meter.hidden = state !== "recording" && state !== "writing";
  if (state === "writing") clock.textContent = "Writing…";
}

const clockText = (seconds) => Math.floor(seconds / 60) + ":" + String(Math.floor(seconds % 60)).padStart(2, "0");

async function begin() {
  quiet();
  await loadWhisper();
  if (!voice) return say("Could not reach the helper.");
  if (!whispers().length) {
    const notes = voice.whisper_notes || [];
    return say((notes[0] ? notes[0] + " " : "") + "Speaking needs a Whisper. Get one with", { command: voice.whisper_get, link: "/welcome" });
  }
  if (!voice.ready) return say("Writing down speech is not set up yet. Run", { command: "hikarilm restart" });
  if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
    return say("The browser only allows the microphone on http://127.0.0.1:8787 (or localhost), not at " + location.host + ".");
  }
  if (typeof MediaRecorder === "undefined") return say("This browser cannot record sound. Safari, Chrome and Firefox can.");
  const allowed = await permission();
  if (allowed === "denied") return say("The microphone is blocked for this page. " + howToAllow());
  if (allowed === "prompt") say("The browser will ask to use the microphone. What you say is written down here, on this Mac; nothing is sent anywhere.");
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true } });
  } catch (error) {
    return say(why(error));
  }
  quiet();
  const type = ["audio/webm;codecs=opus", "audio/mp4", "audio/webm", ""].find((kind) => !kind || MediaRecorder.isTypeSupported(kind));
  const recorder = new MediaRecorder(stream, type ? { mimeType: type } : undefined);
  const context = new AudioContext();
  const analyser = context.createAnalyser();
  analyser.fftSize = 1024;
  context.createMediaStreamSource(stream).connect(analyser);
  const me = { stream, recorder, context, chunks: [], began: performance.now(), cancelled: false, frame: 0,
    at: [messageBox.selectionStart ?? messageBox.value.length, messageBox.selectionEnd ?? messageBox.value.length] };
  session = me;
  recorder.ondataavailable = (event) => { if (event.data.size) me.chunks.push(event.data); };
  recorder.onstop = () => finishRecording(me);
  for (const track of stream.getAudioTracks()) track.addEventListener("ended", () => stopRecording(me));
  recorder.start(1000);
  setButton("recording");
  const samples = new Float32Array(analyser.fftSize);
  const tick = () => {
    if (session !== me || me.recorder.state !== "recording") return;
    analyser.getFloatTimeDomainData(samples);
    const loud = Math.sqrt(samples.reduce((sum, v) => sum + v * v, 0) / samples.length);
    level.style.width = Math.min(100, Math.round(loud * 400)) + "%";
    const seconds = (performance.now() - me.began) / 1000;
    clock.textContent = clockText(seconds);
    if (seconds >= (voice.max_seconds || 600)) {
      stopRecording(me);
      say("Stopped at " + clockText(voice.max_seconds || 600) + ", the longest a recording can be.");
      return;
    }
    me.frame = requestAnimationFrame(tick);
  };
  tick();
}

function release(me) {
  cancelAnimationFrame(me.frame);
  for (const track of me.stream.getTracks()) track.stop();
  me.context.close().catch(() => {});
}

function stopRecording(me) {
  if (me.recorder.state !== "inactive") me.recorder.stop();
}

export function cancel() {
  const me = session;
  if (!me) return false;
  if (me.recorder?.state === "recording") {
    me.cancelled = true;
    stopRecording(me);
    status.textContent = "Recording cancelled.";
  } else if (me.controller) {
    me.controller.abort();
  }
  return true;
}

// 16 kHz mono 16-bit PCM, made in the page: no ffmpeg.
async function toPcm(blob) {
  const decoding = new AudioContext();
  try {
    const sound = await decoding.decodeAudioData(await blob.arrayBuffer());
    const frames = Math.ceil(sound.duration * RATE);
    if (!frames) return new Int16Array(0);
    const offline = new OfflineAudioContext(1, frames, RATE);
    const source = offline.createBufferSource();
    source.buffer = sound;
    source.connect(offline.destination);
    source.start();
    const mono = (await offline.startRendering()).getChannelData(0);
    const pcm = new Int16Array(mono.length);
    for (let i = 0; i < mono.length; i++) pcm[i] = Math.max(-32768, Math.min(32767, Math.round(mono[i] * 32767)));
    return pcm;
  } finally {
    decoding.close().catch(() => {});
  }
}

// Puts a part where the cursor was, with a space between it and the words around it.
function insert(me, text) {
  const [from, to] = me.at;
  const before = messageBox.value.slice(0, from);
  const after = messageBox.value.slice(to);
  const lead = before && !/\s$/.test(before) ? " " : "";
  const trail = after && !/^\s/.test(after) ? " " : "";
  messageBox.value = before + lead + text + trail + after;
  const end = before.length + lead.length + text.length;
  me.at = [end, end];
  messageBox.setSelectionRange(end, end);
  messageBox.dispatchEvent(new Event("input", { bubbles: true }));
}

async function finishRecording(me) {
  release(me);
  if (me.cancelled || session !== me) {
    if (session === me) session = null;
    setButton("idle");
    return;
  }
  setButton("writing");
  me.controller = new AbortController();
  let wrote = false;
  try {
    const pcm = (await toPcm(new Blob(me.chunks, { type: me.recorder.mimeType }))).slice(0, (voice.max_seconds || 600) * RATE);
    if (pcm.length < RATE / 4) throw new Error("Nothing was recorded.");
    const query = new URLSearchParams({ model: chosenWhisper(), language: localStorage.getItem(LANGUAGE_KEY) || "" });
    status.textContent = "Writing down what you said with " + (whispers().find((item) => item.id === chosenWhisper())?.label || "Whisper") + "…";
    const response = await fetch("/api/transcribe?" + query, {
      method: "POST", headers: { "Content-Type": "application/octet-stream" }, body: pcm.buffer, signal: me.controller.signal,
    });
    if (!response.ok) throw new Error((await response.json().catch(() => ({}))).error || "the helper could not write it down");
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
        if (event.text) {
          insert(me, event.text);
          wrote = true;
        }
      }
    }
    status.textContent = "";
    if (!wrote) say("Nothing was heard. Speak closer to the microphone, or check the input in System Settings > Sound.");
  } catch (error) {
    if (error.name !== "AbortError") say(error.message === "Nothing was recorded." ? error.message : "Could not write it down: " + error.message);
    status.textContent = "";
  } finally {
    if (session === me) session = null;
    setButton("idle");
    messageBox.focus();
  }
}

function showWhisperSetting() {
  const choice = document.querySelector("#whisper-choice");
  const language = document.querySelector("#whisper-language");
  const shown = document.querySelector("#whisper-note");
  const listed = whispers();
  choice.replaceChildren(...(listed.length ? listed.map((item) => new Option(item.label, item.id)) : [new Option("None downloaded", "")]));
  choice.value = chosenWhisper();
  choice.disabled = !listed.length;
  language.replaceChildren(...LANGUAGES.map(([code, name]) => new Option(name, code)));
  language.value = localStorage.getItem(LANGUAGE_KEY) || "";
  language.disabled = !listed.length;
  if (!voice) shown.textContent = "Could not reach the helper.";
  else if (!listed.length) shown.textContent = "Get Whisper to speak instead of typing:  " + voice.whisper_get + "  (1.6 GB; " +
    voice.whisper_small_get.split(" ").pop() + " is 0.5 GB), then run  hikarilm restart.";
  else if (!voice.ready) shown.textContent = "Whisper is downloaded. Run  hikarilm restart  in Terminal to set it up.";
  else shown.textContent = "Press the microphone by Attach and speak; press it again to have it written into the message box. " +
    "Whisper runs here; the sound never leaves this Mac. The browser asks for the microphone the first time.";
}

// Listeners and first steps, run by main.js once every part is loaded.
export function start() {
  setButton("idle");
  loadWhisper();
  document.querySelector("#open-settings").addEventListener("click", () => loadWhisper().then(showWhisperSetting));
  document.querySelector("#whisper-choice").addEventListener("change", (event) => localStorage.setItem(WHISPER_KEY, event.target.value));
  document.querySelector("#whisper-language").addEventListener("change", (event) => localStorage.setItem(LANGUAGE_KEY, event.target.value));
  button.addEventListener("click", () => {
    if (!session) begin();
    else if (session.recorder?.state === "recording") stopRecording(session);
    else session.controller?.abort();
  });
  document.querySelector("#mic-close").addEventListener("click", quiet);
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && session && !document.querySelector("dialog[open]")) {
      event.preventDefault();
      event.stopImmediatePropagation();
      cancel();
    }
  }, true);
}
