// Which text and code files are accepted, reading their encoding, PDFs, and short notices.
import { MAX_FILES_TOTAL, MAX_PDF_PAGES } from "./attachments.js";

// Plain-text kinds the page accepts: notes, data, and everyday code.
// Anything else (.asm, model files, archives...) is refused, even if it
// happens to be readable text.
export const TEXT_EXTENSIONS = new Set([
  "txt", "md", "markdown", "rst", "log", "csv", "tsv",
  "json", "jsonl", "yaml", "yml", "toml", "ini", "cfg", "xml", "html", "htm", "css",
  "py", "r", "rmd", "c", "h", "cpp", "cc", "cxx", "hpp", "hh",
  "java", "kt", "go", "rs", "rb", "php", "swift", "cs", "scala", "lua", "pl",
  "js", "mjs", "ts", "jsx", "tsx", "sh", "bash", "sql", "tex",
]);
const TEXT_NAMES = new Set(["makefile", "dockerfile", "readme", "license", "cmakelists.txt"]);

export function isTextKind(name) {
  const lower = name.toLowerCase();
  if (TEXT_NAMES.has(lower)) return true;
  const dot = lower.lastIndexOf(".");
  return dot > 0 || (dot === 0 && lower.length > 1) ? TEXT_EXTENSIONS.has(lower.slice(dot + 1)) : false;
}

// Read bytes as text. Honours a byte-order mark (Excel writes UTF-16 and
// UTF-8 BOMs), tries UTF-8, then falls back to Windows-1252, which is what
// older CSV exports use. Returns null for binary data.
export function decodeText(bytes, truncated) {
  if (bytes.includes(0)) {
    if (bytes[0] === 0xff && bytes[1] === 0xfe) return new TextDecoder("utf-16le").decode(bytes.subarray(2));
    if (bytes[0] === 0xfe && bytes[1] === 0xff) return new TextDecoder("utf-16be").decode(bytes.subarray(2));
    return null;
  }
  if (bytes[0] === 0xef && bytes[1] === 0xbb && bytes[2] === 0xbf) bytes = bytes.subarray(3);
  try {
    return new TextDecoder("utf-8", { fatal: true }).decode(bytes);
  } catch (error) {
    // A multi-byte character may be split by the cut; trim up to 3 bytes and retry.
    for (let trim = 1; truncated && trim <= 3; trim += 1) {
      try {
        return new TextDecoder("utf-8", { fatal: true }).decode(bytes.subarray(0, bytes.length - trim));
      } catch (inner) { /* keep trimming */ }
    }
  }
  const text = new TextDecoder("windows-1252").decode(bytes);
  // Control characters other than tab, newline, return mean binary, not text.
  return /[\u0001-\u0008\u000e-\u001f]/.test(text) ? null : text;
}

// pdf.js is bundled in web/vendor and loaded only when a PDF is attached.
let pdfjs = null;
export async function readPdf(file) {
  if (!pdfjs) {
    pdfjs = await import("/vendor/pdf.min.mjs");
    pdfjs.GlobalWorkerOptions.workerSrc = "/vendor/pdf.worker.min.mjs";
  }
  const doc = await pdfjs.getDocument({ data: new Uint8Array(await file.arrayBuffer()) }).promise;
  let text = "";
  const limit = Math.min(doc.numPages, MAX_PDF_PAGES);
  for (let number = 1; number <= limit && text.length < MAX_FILES_TOTAL; number += 1) {
    const page = await doc.getPage(number);
    const content = await page.getTextContent();
    let line = "";
    for (const item of content.items) {
      line += item.str;
      if (item.hasEOL) line += "\n";
      else if (item.str && !line.endsWith(" ")) line += " ";
    }
    text += (limit > 1 ? "\n--- page " + number + " ---\n" : "") + line.replace(/[ \t]+\n/g, "\n").trim() + "\n";
  }
  return {
    text,
    pages: doc.numPages,
    close: () => doc.destroy(),
    async renderPage(number) {
      const page = await doc.getPage(number);
      const viewport = page.getViewport({ scale: 1.5 });
      const canvas = document.createElement("canvas");
      canvas.width = viewport.width;
      canvas.height = viewport.height;
      await page.render({ canvasContext: canvas.getContext("2d"), viewport }).promise;
      return canvas.toDataURL("image/png").split(",", 2)[1];
    },
  };
}

let noticeTimer = null;
export function notify(text) {
  const notice = document.querySelector("#notice");
  notice.textContent = text;
  notice.hidden = false;
  clearTimeout(noticeTimer);
  noticeTimer = setTimeout(() => { notice.hidden = true; }, 6000);
}

export function readBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result).split(",", 2)[1] || "");
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(file);
  });
}
