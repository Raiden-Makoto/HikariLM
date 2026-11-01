// Editing and deleting a message you sent, and what came after it.
import { state } from "./state.js";
import { messageBox, showEmpty } from "./core.js";
import { renderChips } from "./attachments.js";
import { notify } from "./text-files.js";
import { clearRewrite } from "./models.js";
import { saveChat } from "./chat-list.js";
import { deleteChat } from "./chats.js";
import { renderLog } from "./turns.js";

// Drop messages from `index` on. Pictures they made are deleted from the helper too.
async function dropFrom(index, count = state.messages.length - index) {
  const gone = state.messages.splice(index, count);
  for (const message of gone) {
    if (message.picture?.id) fetch("/api/images/" + message.picture.id, { method: "DELETE" }).catch(() => {});
  }
  if (state.messages.length) {
    renderLog();
    await saveChat();
  } else if (state.chatId) {
    await deleteChat(state.chatId);
    showEmpty();
  }
}

// The message and its reply; the reply is the next message when it is the model's.
const replyCount = (index) => (state.messages[index + 1]?.role === "assistant" ? 2 : 1);

export async function deletePrompt(message) {
  if (state.activeRequest) return;
  const index = state.messages.indexOf(message);
  if (index < 0) return;
  const count = replyCount(index);
  const what = state.messages[index + 1]?.picture ? "this message and its picture" : count === 2 ? "this message and its reply" : "this message";
  if (!confirm("Delete " + what + "?")) return;
  await dropFrom(index, count);
}

// The message becomes a box in place. Sending it replaces the message and everything
// after it, the way ChatGPT does, and goes out with the model picked now.
export function editPrompt(message, block, answer) {
  if (state.activeRequest || block.querySelector(".prompt-edit")) return;
  const box = document.createElement("div");
  box.className = "prompt-edit";
  const field = document.createElement("textarea");
  field.value = message.content;
  field.setAttribute("aria-label", "Edit your message");
  const row = document.createElement("div");
  row.className = "prompt-actions";
  const resend = document.createElement("button");
  resend.type = "button";
  resend.textContent = "Send";
  const cancel = document.createElement("button");
  cancel.type = "button";
  cancel.textContent = "Cancel";
  row.append(resend, cancel);
  box.append(field, row);
  answer.hidden = true;
  block.querySelector(".prompt-actions").hidden = true;
  answer.after(box);
  const fit = () => {
    field.style.height = "auto";
    field.style.height = Math.min(field.scrollHeight, 320) + "px";
  };
  fit();
  field.focus();
  field.setSelectionRange(field.value.length, field.value.length);
  const close = () => {
    box.remove();
    answer.hidden = false;
    block.querySelector(".prompt-actions").hidden = false;
  };
  const submit = async () => {
    const text = field.value.trim();
    if (!text || state.activeRequest) return;
    const index = state.messages.indexOf(message);
    if (index < 0) return close();
    const later = state.messages.length - index - replyCount(index);
    if (later > 0 && !confirm("This also removes the " + later + (later === 1 ? " message" : " messages") +
        " after it. Send the edited message?")) return;
    // Chats are saved without their pictures, so a reloaded chat has only their names.
    const files = (message.files || []).filter((file) => file.image || typeof file.text === "string");
    const lost = (message.files || []).length - files.length;
    await dropFrom(index);
    clearRewrite();
    messageBox.value = text;
    state.attached = files;
    renderChips();
    if (lost) {
      notify("Pictures are not kept after a reload, so " + (lost === 1 ? "one was" : lost + " were") +
        " left out. Attach them again and send.");
      messageBox.focus();
      return;
    }
    document.querySelector("#composer").requestSubmit();
  };
  resend.addEventListener("click", submit);
  cancel.addEventListener("click", close);
  field.addEventListener("input", fit);
  field.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      submit();
    } else if (event.key === "Escape") {
      event.preventDefault();
      close();
    }
  });
}
