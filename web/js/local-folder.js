// A folder on this computer, read and written by the page itself, and remembered after a reload.
import { MAX_FILE } from "./attachments.js";

// A folder on this computer, picked with the browser's folder picker. The
// page does here what the helper does for the other kind of folder, with
// the same rules: everyday file kinds only, nothing outside the folder,
// 500 KB at most, the same encoding and line endings written back, and a
// refusal if the file changed since it was opened.
const LOCAL_EXTENSIONS = new Set(("txt md rst log csv tsv json yaml yml toml ini xml html css py r c h cpp hpp " +
  "java kt go rs rb php swift cs scala lua pl js ts jsx tsx sh sql tex").split(" "));
const LOCAL_NAMES = new Set(["makefile", "dockerfile", "readme", "license"]);
const LOCAL_SKIP = new Set(["node_modules", "__pycache__", "venv"]);
const LOCAL_MAX_FILES = 200;
const LOCAL_MAX_DEPTH = 4;

function localAllowed(name) {
  const lower = name.toLowerCase();
  const dot = lower.lastIndexOf(".");
  return (dot > 0 || (dot === 0 && lower.length > 1) ? LOCAL_EXTENSIONS.has(lower.slice(dot + 1)) : false) ||
    LOCAL_NAMES.has(lower);
}

async function sha1(bytes) {
  const digest = new Uint8Array(await crypto.subtle.digest("SHA-1", bytes));
  return [...digest].map((byte) => byte.toString(16).padStart(2, "0")).join("");
}

function localDecode(bytes, name) {
  const starts = (...head) => head.every((value, index) => bytes[index] === value);
  try {
    if (starts(0xef, 0xbb, 0xbf)) return { text: new TextDecoder("utf-8", { fatal: true }).decode(bytes.subarray(3)), codec: "utf-8", bom: [0xef, 0xbb, 0xbf] };
    if (starts(0xff, 0xfe)) return { text: new TextDecoder("utf-16le").decode(bytes.subarray(2)), codec: "utf-16le", bom: [0xff, 0xfe] };
    if (starts(0xfe, 0xff)) return { text: new TextDecoder("utf-16be").decode(bytes.subarray(2)), codec: "utf-16be", bom: [0xfe, 0xff] };
  } catch (error) {
    throw new Error(name + " is not text this page can read.");
  }
  if (bytes.includes(0)) throw new Error(name + " has binary bytes, so it is not opened.");
  try {
    return { text: new TextDecoder("utf-8", { fatal: true }).decode(bytes), codec: "utf-8", bom: [] };
  } catch (error) {
    return { text: new TextDecoder("windows-1252").decode(bytes), codec: "windows-1252", bom: [] };
  }
}

let cp1252Bytes = null;
function localEncode(text, codec, name) {
  if (codec === "utf-8") return new TextEncoder().encode(text);
  if (codec === "utf-16le" || codec === "utf-16be") {
    const out = new Uint8Array(text.length * 2);
    for (let i = 0; i < text.length; i += 1) {
      const unit = text.charCodeAt(i);
      const [low, high] = codec === "utf-16le" ? [i * 2, i * 2 + 1] : [i * 2 + 1, i * 2];
      out[low] = unit & 255;
      out[high] = unit >> 8;
    }
    return out;
  }
  if (!cp1252Bytes) {
    cp1252Bytes = new Map();
    const decoder = new TextDecoder("windows-1252");
    for (let value = 0; value < 256; value += 1) cp1252Bytes.set(decoder.decode(new Uint8Array([value])), value);
  }
  const out = [];
  for (const character of text) {
    const value = cp1252Bytes.get(character);
    if (value === undefined) throw new Error(name + " is saved as windows-1252, which cannot hold a character you typed.");
    out.push(value);
  }
  return Uint8Array.from(out);
}

function localFailure(error, name) {
  if (error instanceof Error && !error.name.endsWith("Error")) return error;
  if (error.name === "NotFoundError" || error.name === "TypeMismatchError") return new Error(name + " is not in the folder any more.");
  if (error.name === "NotAllowedError" || error.name === "SecurityError") return new Error("The browser no longer lets this page use that folder. Choose it again.");
  return error;
}

export function localSource(root) {
  const handleFor = async (path) => {
    const parts = path.split("/");
    if (!path || path.startsWith("/") || path.includes("\\") || parts.some((part) => !part || part === "." || part === "..")) {
      throw new Error("That path leaves the opened folder.");
    }
    let directory = root;
    for (const part of parts.slice(0, -1)) directory = await directory.getDirectoryHandle(part);
    const name = parts[parts.length - 1];
    if (!localAllowed(name)) throw new Error(name + " is not a kind of file this page opens.");
    return directory.getFileHandle(name);
  };
  const readBytes = async (handle) => {
    const file = await handle.getFile();
    if (file.size > MAX_FILE) throw new Error(file.name + " is " + Math.round(file.size / 1000) + " KB. Files over 500 KB are not opened.");
    return new Uint8Array(await file.arrayBuffer());
  };
  return {
    kind: "local",
    root,
    label: root.name + "  (on this computer)",
    async list() {
      const found = [];
      let cut = false;
      const walk = async (directory, prefix, depth) => {
        const entries = [];
        for await (const [name, entry] of directory.entries()) entries.push([name, entry]);
        entries.sort((a, b) => (a[0].toLowerCase() < b[0].toLowerCase() ? -1 : a[0].toLowerCase() > b[0].toLowerCase() ? 1 : 0));
        for (const [name, entry] of entries) {
          if (cut) return;
          if (entry.kind === "directory") {
            if (name.startsWith(".") || LOCAL_SKIP.has(name) || depth >= LOCAL_MAX_DEPTH) continue;
            await walk(entry, prefix + name + "/", depth + 1);
          } else if (localAllowed(name)) {
            if (found.length >= LOCAL_MAX_FILES) {
              cut = true;
              return;
            }
            found.push(prefix + name);
          }
        }
      };
      try {
        await walk(root, "", 0);
      } catch (error) {
        throw localFailure(error, root.name);
      }
      return { root: root.name, name: root.name, files: found, cut };
    },
    async read(path) {
      try {
        const bytes = await readBytes(await handleFor(path));
        const { text } = localDecode(bytes, path);
        return { path, text: text.replace(/\r\n/g, "\n"), version: await sha1(bytes) };
      } catch (error) {
        throw localFailure(error, path);
      }
    },
    async create(path) {
      try {
        const parts = path.split("/");
        if (!path || path.startsWith("/") || parts.some((part) => !part || part === "." || part === ".." || part.startsWith("."))) {
          throw new Error("Use a name like notes.md or src/main.py, with no leading slash or dots.");
        }
        const name = parts[parts.length - 1];
        if (!localAllowed(name)) throw new Error(name + " is not a kind of file this page opens, for example .py, .md or .txt.");
        let directory = root;
        for (const part of parts.slice(0, -1)) directory = await directory.getDirectoryHandle(part, { create: true });
        const exists = await directory.getFileHandle(name).then(() => true, (error) => {
          if (error.name === "NotFoundError") return false;
          throw error;
        });
        if (exists) throw new Error(name + " already exists in that folder.");
        await directory.getFileHandle(name, { create: true });
        return { path };
      } catch (error) {
        throw localFailure(error, path);
      }
    },
    async remove(path) {
      try {
        await handleFor(path);
        const parts = path.split("/");
        let directory = root;
        for (const part of parts.slice(0, -1)) directory = await directory.getDirectoryHandle(part);
        await directory.removeEntry(parts[parts.length - 1]);
        return { deleted: true };
      } catch (error) {
        throw localFailure(error, path);
      }
    },
    async write(path, text, version) {
      try {
        const handle = await handleFor(path);
        const bytes = await readBytes(handle);
        if ((await sha1(bytes)) !== version) {
          const error = new Error(path + " was changed on disk after you opened it.");
          error.conflict = true;
          throw error;
        }
        const { text: before, codec, bom } = localDecode(bytes, path);
        let body = text.replace(/\r\n/g, "\n");
        if (before.includes("\r\n")) body = body.replace(/\n/g, "\r\n");
        const encoded = localEncode(body, codec, path);
        const data = new Uint8Array(bom.length + encoded.length);
        data.set(bom);
        data.set(encoded, bom.length);
        if (data.length > MAX_FILE) throw new Error("The file would be over 500 KB, so it was not saved.");
        if (data.length === bytes.length && data.every((value, index) => value === bytes[index])) {
          return { version, changed: false };
        }
        const writer = await handle.createWritable();
        try {
          await writer.write(data);
          await writer.close();
        } catch (error) {
          try { await writer.abort(); } catch (ignored) { /* already closed */ }
          throw error;
        }
        return { version: await sha1(data), changed: true };
      } catch (error) {
        throw localFailure(error, path);
      }
    },
  };
}

// The browser keeps a picked folder in its own storage, so it can be
// offered again after a reload. It still asks the user's permission first.
function handleStore(mode, run) {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open("lightusagi", 1);
    request.onupgradeneeded = () => request.result.createObjectStore("handles");
    request.onerror = () => reject(request.error);
    request.onsuccess = () => {
      const db = request.result;
      const tx = db.transaction("handles", mode);
      const result = run(tx.objectStore("handles"));
      tx.oncomplete = () => { db.close(); resolve(result.result); };
      tx.onerror = () => { db.close(); reject(tx.error); };
    };
  });
}
export const rememberHandle = (handle) => handleStore("readwrite", (store) => store.put(handle, "last")).catch(() => {});
export const recallHandle = () => handleStore("readonly", (store) => store.get("last")).catch(() => undefined);
