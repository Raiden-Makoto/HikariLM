// Files attached to a message: the limits, the chips under the box, and reading each kind.
import { state } from "./state.js";
import { modelSelect, status } from "./core.js";
import { readDocx, readXlsx } from "./office.js";
import { decodeText, isTextKind, notify, readBase64, readPdf, visionModel } from "./text-files.js";
import { inCodeView, updatePictureMode } from "./models.js";
import { draftKey, saveDraft } from "./chats.js";

export const MAX_FILE = 500_000;
export const MAX_FILES_TOTAL = 2_000_000;
const MAX_IMAGE = 10_000_000;
const MAX_IMAGES_TOTAL = 30_000_000;
const MAX_OFFICE = 50_000_000;
const MAX_PDF = 50_000_000;
export const MAX_PDF_PAGES = 150;
const PDF_IMAGE_PAGES = 4;
const IMAGE_TYPES = ["image/png", "image/jpeg", "image/webp", "image/gif"];
const chips = document.querySelector("#chips");
export const fileInput = document.querySelector("#file-input");
export const attachButton = document.querySelector("#attach");

export function renderChips() {
  chips.replaceChildren();
  state.attached.forEach((file, index) => {
    const chip = document.createElement("div");
    chip.className = "chip";
    const name = document.createElement("span");
    name.textContent = file.name;
    name.title = file.name;
    const remove = document.createElement("button");
    remove.type = "button";
    remove.textContent = "×";
    remove.setAttribute("aria-label", "Remove " + file.name);
    remove.addEventListener("click", () => {
      state.attached.splice(index, 1);
      renderChips();
      saveDraft(draftKey(), true);
    });
    chip.append(name, remove);
    chips.append(chip);
  });
}

export async function addFiles(list) {
  let total = state.attached.reduce((sum, file) => sum + (file.text || "").length, 0);
  const problems = [];
  let images = state.attached.filter((item) => item.image).length;
  let imageBytes = state.attached.reduce((sum, item) => sum + (item.image ? item.image.length : 0), 0);
  let switched = "";
  const addImage = async (name, size, getData) => {
    const vision = visionModel();
    if (!vision) {
      problems.push(name + " needs a Gemma model, and none is installed in Ollama");
      return false;
    }
    if (size > MAX_IMAGE) {
      problems.push(name + " is over 10 MB");
      return false;
    }
    if (imageBytes + size > MAX_IMAGES_TOTAL) {
      problems.push(name + " would make the pictures too big for one message");
      return false;
    }
    state.attached.push({ name, image: await getData() });
    imageBytes += size;
    images += 1;
    if (modelSelect.value !== vision) {
      modelSelect.value = vision;
      updatePictureMode();
      switched = vision;
    }
    return true;
  };
  // Add text to the message. Too long for one file or for the message:
  // cut at a line break and say so.
  const addText = (name, text, cut, limit = MAX_FILE) => {
    if (text.length > limit) {
      text = text.slice(0, limit);
      cut = true;
    }
    const room = MAX_FILES_TOTAL - total;
    if (room < 200) {
      problems.push(name + " would make the files too big");
      return;
    }
    if (text.length > room) {
      text = text.slice(0, room);
      cut = true;
    }
    if (cut) {
      const lastBreak = text.lastIndexOf("\n");
      if (lastBreak > 0) text = text.slice(0, lastBreak);
      text += "\n[cut here: this is longer than one message can hold]";
      problems.push(name + " was cut to its first " + (text.split("\n").length - 1) + " lines");
    }
    total += text.length;
    state.attached.push({ name, text });
  };
  for (const file of list) {
    if (inCodeView() && (IMAGE_TYPES.includes(file.type) || /\.(pdf|docx|xlsx)$/i.test(file.name) || file.type === "application/pdf")) {
      problems.push(file.name + " was not attached: the code editor takes text and code files only. Pictures and PDFs go in the Chat tab");
      continue;
    }
    if (IMAGE_TYPES.includes(file.type)) {
      await addImage(file.name, file.size, () => readBase64(file));
      continue;
    }
    if (file.type === "application/pdf" || /\.pdf$/i.test(file.name)) {
      if (file.size > MAX_PDF) {
        problems.push(file.name + " is over 50 MB");
        continue;
      }
      let pdf;
      try {
        pdf = await readPdf(file);
      } catch (error) {
        problems.push(file.name + " could not be read as a PDF");
        continue;
      }
      if (pdf.text.trim().length >= 40) {
        if (pdf.pages > MAX_PDF_PAGES) {
          problems.push(file.name + " has " + pdf.pages + " pages; only the first " + MAX_PDF_PAGES + " were read");
        }
        addText(file.name, pdf.text, false, MAX_FILES_TOTAL);
      } else {
        // Scanned PDF: no text inside, so Gemma reads the first pages as pictures.
        const shown = Math.min(pdf.pages, PDF_IMAGE_PAGES);
        let added = 0;
        for (let page = 1; page <= shown; page += 1) {
          const png = await pdf.renderPage(page);
          if (!(await addImage(file.name + " p" + page, png.length, async () => png))) break;
          added += 1;
        }
        if (added && pdf.pages > shown) {
          problems.push(file.name + " is scanned; only the first " + shown + " of " + pdf.pages + " pages were attached");
        }
      }
      pdf.close();
      continue;
    }
    if (/\.(docx|xlsx)$/i.test(file.name)) {
      if (file.size > MAX_OFFICE) {
        problems.push(file.name + " is over 50 MB");
        continue;
      }
      try {
        const text = /\.docx$/i.test(file.name) ? await readDocx(file) : await readXlsx(file);
        if (!text.trim()) problems.push(file.name + " has no text in it");
        else addText(file.name, text, false, MAX_FILES_TOTAL);
      } catch (error) {
        problems.push(file.name + " could not be read (" + error.message + ")");
      }
      continue;
    }
    if (/\.(doc|xls|ppt|pptx|odt|ods)$/i.test(file.name)) {
      problems.push(file.name + " is not supported; save it as .docx or .xlsx, or export it to text");
      continue;
    }
    if (!isTextKind(file.name)) {
      problems.push(file.name + " is not a supported file type");
      continue;
    }
    // Plain text, markdown, code, csv, tsv. A long one is cut at a line break, not refused.
    const head = new Uint8Array(await file.slice(0, MAX_FILE + 4).arrayBuffer());
    const text = decodeText(head, file.size > head.length);
    if (text === null) {
      problems.push(file.name + " is not a text file");
      continue;
    }
    addText(file.name, text, file.size > head.length);
  }
  renderChips();
  const notes = [];
  if (switched) notes.push("Switched to " + switched + " because you attached a picture. To get code from it, choose a coder model: " + switched + " will write the picture down for it first.");
  if (problems.length) notes.push("Heads up: " + problems.join("; ") + ".");
  status.textContent = notes.join(" ");
  saveDraft(draftKey(), true);
  if (switched) notify("Switched to " + switched + " to read your image.");
}
