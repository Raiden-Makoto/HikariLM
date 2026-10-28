// .docx and .xlsx are zip files of XML. The browser can inflate zip data
// itself (DecompressionStream), so this needs no library: find the entries
// in the zip's central directory and inflate only the ones we read.
const MAX_INFLATED = 100_000_000;

async function openZip(file) {
  const tail = new Uint8Array(await file.slice(Math.max(0, file.size - 70_000)).arrayBuffer());
  const tailView = new DataView(tail.buffer);
  let end = -1;
  for (let i = tail.length - 22; i >= 0; i -= 1) {
    if (tailView.getUint32(i, true) === 0x06054b50) { end = i; break; }
  }
  if (end < 0) throw new Error("not a zip file");
  const count = tailView.getUint16(end + 10, true);
  const dirSize = tailView.getUint32(end + 12, true);
  const dirStart = tailView.getUint32(end + 16, true);
  if (dirStart === 0xffffffff || count === 0xffff) throw new Error("zip is too large");
  const dir = new Uint8Array(await file.slice(dirStart, dirStart + dirSize).arrayBuffer());
  const view = new DataView(dir.buffer);
  const entries = new Map();
  let at = 0;
  for (let n = 0; n < count && at + 46 <= dir.length; n += 1) {
    if (view.getUint32(at, true) !== 0x02014b50) break;
    const nameLength = view.getUint16(at + 28, true);
    const extraLength = view.getUint16(at + 30, true);
    const commentLength = view.getUint16(at + 32, true);
    entries.set(new TextDecoder().decode(dir.subarray(at + 46, at + 46 + nameLength)), {
      method: view.getUint16(at + 10, true),
      size: view.getUint32(at + 20, true),
      offset: view.getUint32(at + 42, true),
    });
    at += 46 + nameLength + extraLength + commentLength;
  }
  return {
    has: (name) => entries.has(name),
    names: () => [...entries.keys()],
    async text(name) {
      const entry = entries.get(name);
      if (!entry) return null;
      const header = new DataView(await file.slice(entry.offset, entry.offset + 30).arrayBuffer());
      const start = entry.offset + 30 + header.getUint16(26, true) + header.getUint16(28, true);
      const raw = file.slice(start, start + entry.size);
      if (entry.method === 0) return new TextDecoder().decode(await raw.arrayBuffer());
      if (entry.method !== 8) throw new Error("unsupported zip compression");
      const stream = raw.stream().pipeThrough(new DecompressionStream("deflate-raw"));
      const reader = stream.getReader();
      const decoder = new TextDecoder();
      let out = "";
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        out += decoder.decode(value, { stream: true });
        if (out.length > MAX_INFLATED) {
          reader.cancel();
          throw new Error("file expands too much");
        }
      }
      return out + decoder.decode();
    },
  };
}

function parseXml(text) {
  const doc = new DOMParser().parseFromString(text, "application/xml");
  if (doc.querySelector("parsererror")) throw new Error("damaged XML");
  return doc;
}

const localName = (node) => node.localName;
const children = (node, name) => [...node.children].filter((child) => localName(child) === name);

export async function readDocx(file) {
  const zip = await openZip(file);
  const body = await zip.text("word/document.xml");
  if (body === null) throw new Error("no document.xml");
  const root = parseXml(body).getElementsByTagNameNS("*", "body")[0];
  const paragraph = (node) => {
    let out = "";
    const walk = (item) => {
      for (const child of item.children) {
        const name = localName(child);
        if (name === "t") out += child.textContent;
        else if (name === "tab") out += "\t";
        else if (name === "br" || name === "cr") out += "\n";
        else if (name !== "instrText") walk(child);
      }
    };
    walk(node);
    return out;
  };
  const lines = [];
  const block = (node) => {
    for (const child of node.children) {
      const name = localName(child);
      if (name === "p") lines.push(paragraph(child));
      else if (name === "tbl") {
        for (const row of children(child, "tr")) {
          lines.push(children(row, "tc").map((cell) =>
            children(cell, "p").map(paragraph).join(" ")).join("\t"));
        }
        lines.push("");
      } else if (name === "sdt" || name === "customXml") {
        for (const inner of child.children) if (localName(inner) === "sdtContent") block(inner);
      }
    }
  };
  block(root);
  return lines.join("\n").replace(/\n{3,}/g, "\n\n").trim();
}

function columnNumber(reference) {
  let n = 0;
  for (const ch of reference.replace(/[0-9]/g, "")) n = n * 26 + ch.charCodeAt(0) - 64;
  return n - 1;
}

export async function readXlsx(file) {
  const zip = await openZip(file);
  const book = await zip.text("xl/workbook.xml");
  if (book === null) throw new Error("no workbook.xml");
  const rels = {};
  const relText = await zip.text("xl/_rels/workbook.xml.rels");
  if (relText) {
    for (const rel of parseXml(relText).getElementsByTagName("Relationship")) {
      rels[rel.getAttribute("Id")] = rel.getAttribute("Target");
    }
  }
  const shared = [];
  const sharedText = await zip.text("xl/sharedStrings.xml");
  if (sharedText) {
    for (const item of parseXml(sharedText).getElementsByTagNameNS("*", "si")) {
      // Skip phonetic runs (rPh); keep ordinary and rich-text runs.
      let value = "";
      for (const t of item.getElementsByTagNameNS("*", "t")) {
        if (localName(t.parentNode) !== "rPh") value += t.textContent;
      }
      shared.push(value);
    }
  }
  const parts = [];
  const sheets = [...parseXml(book).getElementsByTagNameNS("*", "sheet")];
  for (const [index, sheet] of sheets.entries()) {
    const id = sheet.getAttributeNS("http://schemas.openxmlformats.org/officeDocument/2006/relationships", "id");
    let target = rels[id] || "worksheets/sheet" + (index + 1) + ".xml";
    target = target.startsWith("/") ? target.slice(1) : "xl/" + target;
    const xml = await zip.text(target);
    if (xml === null) continue;
    const rows = [];
    for (const row of parseXml(xml).getElementsByTagNameNS("*", "row")) {
      const cells = [];
      let next = 0;
      for (const cell of children(row, "c")) {
        const reference = cell.getAttribute("r");
        const column = reference ? columnNumber(reference) : next;
        const type = cell.getAttribute("t");
        const valueNode = children(cell, "v")[0];
        let value = "";
        if (type === "inlineStr") value = [...cell.getElementsByTagNameNS("*", "t")].map((t) => t.textContent).join("");
        else if (valueNode) {
          value = valueNode.textContent;
          if (type === "s") value = shared[Number(value)] ?? "";
          else if (type === "b") value = value === "1" ? "TRUE" : "FALSE";
        }
        while (cells.length < column) cells.push("");
        cells[column] = value.replace(/[\t\r\n]+/g, " ");
        next = column + 1;
      }
      rows.push(cells.join("\t").replace(/\t+$/, ""));
    }
    while (rows.length && !rows[rows.length - 1]) rows.pop();
    if (rows.length) parts.push((sheets.length > 1 ? "--- sheet: " + sheet.getAttribute("name") + " ---\n" : "") + rows.join("\n"));
  }
  return parts.join("\n\n");
}
