// Code blocks in replies, the way Cursor reads them. A block that changes a
// file names it after the language: ```python:src/app.py. It may leave out
// unchanged code, marked by a comment such as "// ... existing code ...".
// Nothing here touches the page, so it can be tested on its own.

// What the model is told in the code editor, so its blocks can be applied.
export function editorPrompt(folder, openPath) {
  return [
    "You are a coding assistant in the HikariLM code editor. The user's project folder is " +
      JSON.stringify(folder || "not chosen yet") + "." + (openPath ? " The open file is " + openPath + "." : ""),
    "",
    "When you change a file, or write a new one, put the code in a fenced block whose first line is the language, " +
      "a colon, and the file's path inside the folder, like this:",
    "```python:src/app.py",
    "# ... existing code ...",
    "def total(items):",
    "    return sum(item.price for item in items)",
    "# ... existing code ...",
    "```",
    "Show only the parts that change, with a few unchanged lines before and after each change so it can be found. " +
      "Where you leave out unchanged code, write one comment line saying \"... existing code ...\" in the file's own " +
      "comment style (// ... existing code ... in JavaScript, # ... existing code ... in Python). " +
      "For a new file, or one you rewrite completely, write all of it with no such comments. One block per file.",
    "",
    "Put a path on a block only when it changes a file in this project, or writes a new file the user asked for. " +
      "Anything else, such as an example to explain something, commands to run or output, goes in a plain " +
      "block with only the language (```python, ```bash) and no path.",
  ].join("\n");
}

// The fence's first line. ```python:src/app.py, ```src/app.py, ```python src/app.py
// and ```python title="src/app.py" all name a file. ```12:30:src/app.py quotes
// lines of a file (Cursor's way), which is not an edit.
export function parseInfo(info) {
  const text = (info || "").trim();
  const quote = /^(\d+):(\d+):(\S+)$/.exec(text);
  if (quote) return { lang: extension(quote[3]), path: quote[3], quote: { from: Number(quote[1]), to: Number(quote[2]) } };
  const [first = "", ...rest] = text.split(/\s+/);
  const attribute = /(?:^|\s)(?:path|file|filename|title)=["']?([^"'\s}]+)/i.exec(text);
  if (first.includes(":")) {
    const at = first.indexOf(":");
    return { lang: first.slice(0, at).toLowerCase(), path: first.slice(at + 1) };
  }
  if (attribute) return { lang: looksLikePath(first) ? extension(first) : first.toLowerCase(), path: attribute[1] };
  const named = rest.find(looksLikePath);
  if (named) return { lang: first.toLowerCase(), path: named };
  if (looksLikePath(first)) return { lang: extension(first), path: first };
  return { lang: first.toLowerCase(), path: "" };
}

const looksLikePath = (word) => /[/\\]/.test(word) || /^[\w.-]*\w\.[A-Za-z0-9]{1,8}$/.test(word) || /^(Dockerfile|Makefile)$/i.test(word);
const extension = (path) => {
  const name = path.split(/[/\\]/).pop().toLowerCase();
  return name.includes(".") ? name.split(".").pop() : name;
};

// Blocks with no path that are never code for a file: commands, output, plain text.
const NOT_FILES = new Set(["", "bash", "sh", "shell", "zsh", "fish", "console", "terminal", "cmd", "bat", "powershell",
  "ps1", "pwsh", "text", "txt", "plain", "plaintext", "output", "log", "diff", "patch", "mermaid"]);
const SAME = {
  python: ["py", "pyw", "pyi"], py: ["py", "pyw", "pyi"],
  javascript: ["js", "mjs", "cjs", "jsx"], js: ["js", "mjs", "cjs", "jsx"], jsx: ["jsx", "js"], node: ["js", "mjs", "cjs"],
  typescript: ["ts", "tsx", "mts", "cts"], ts: ["ts", "mts", "cts", "tsx"], tsx: ["tsx", "ts"],
  c: ["c", "h"], cpp: ["cpp", "cc", "cxx", "hpp", "hh", "hxx", "h"], "c++": ["cpp", "cc", "cxx", "hpp", "hh", "h"],
  csharp: ["cs"], "c#": ["cs"], cs: ["cs"], java: ["java"], kotlin: ["kt", "kts"], kt: ["kt", "kts"],
  rust: ["rs"], rs: ["rs"], go: ["go"], golang: ["go"], ruby: ["rb"], rb: ["rb"], php: ["php"], swift: ["swift"],
  r: ["r", "rmd"], lua: ["lua"], perl: ["pl", "pm"], scala: ["scala"], dart: ["dart"],
  html: ["html", "htm"], css: ["css"], scss: ["scss"], sass: ["sass"], less: ["less"],
  json: ["json", "jsonl"], jsonc: ["json", "jsonc"], yaml: ["yaml", "yml"], yml: ["yaml", "yml"], toml: ["toml"],
  xml: ["xml", "svg"], sql: ["sql"], markdown: ["md", "markdown", "rmd"], md: ["md", "markdown", "rmd"],
  tex: ["tex"], latex: ["tex"], ini: ["ini", "cfg"], dockerfile: ["dockerfile"], makefile: ["makefile", "mk"],
  vue: ["vue"], svelte: ["svelte"],
};

// A block with no path can go into the open file only if it is in the file's language.
export function fitsOpenFile(lang, path) {
  if (!path || NOT_FILES.has(lang)) return false;
  const ext = extension(path);
  return (SAME[lang] || [lang]).includes(ext);
}

// The path a block names, as a path in the open folder. Models sometimes put
// the folder's name or its full path in front, or "./".
export function resolvePath(named, files, folderName, root) {
  let path = String(named).trim().replace(/^["'`]+|["'`]+$/g, "").replace(/\\/g, "/");
  const base = (root || "").replace(/\\/g, "/").replace(/\/+$/, "");
  if (base && path.startsWith(base + "/")) path = path.slice(base.length + 1);
  path = path.replace(/^(\.\/)+/, "");
  if (folderName && path.startsWith(folderName + "/") && !files.includes(path)) path = path.slice(folderName.length + 1);
  if (!path || path.startsWith("/") || path.startsWith("~") || path.split("/").includes("..")) {
    const tail = files.filter((file) => path.endsWith("/" + file));
    return tail.length === 1 ? { path: tail[0], exists: true } : null;
  }
  if (files.includes(path)) return { path, exists: true };
  const lower = path.toLowerCase();
  const sameName = files.filter((file) => file.toLowerCase() === lower);
  if (sameName.length === 1) return { path: sameName[0], exists: true };
  const under = files.filter((file) => file.endsWith("/" + path));
  if (under.length === 1) return { path: under[0], exists: true };
  return { path, exists: false };
}

// The project files a question is about: the ones attached from the editor
// (Cmd/Ctrl+K adds "folder/path", Cmd/Ctrl+L "folder/path lines 3–9") and any it names.
export function askedFiles(question, files, folderName) {
  const asked = new Set();
  for (const file of question?.files || []) {
    let name = String(file.name || "").replace(/ lines? \d+(–\d+)?$/, "");
    if (folderName && name.startsWith(folderName + "/")) name = name.slice(folderName.length + 1);
    if (files.includes(name)) asked.add(name);
  }
  const text = String(question?.text || "");
  for (const file of files) {
    const base = file.split("/").pop();
    if (base.length > 2 && text.includes(base) && mentions(text, base)) asked.add(file);
  }
  return asked;
}

function mentions(text, name) {
  const escaped = name.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  return new RegExp("(^|[^\\w.-])" + escaped + "(?![\\w-])").test(text);
}

const WANTS_FILE = /\b(create|make|add|write|generate|scaffold|set up|new)\b[^.?!\n]*\b(files?|modules?|scripts?|components?|pages?|tests?|classes?)\b/i;

// Which file a reply's code block may be applied to, or "" for none. As in Cursor's
// editor chat, Apply belongs to a question about the project's code: one with a file
// attached or named. Then a block goes into the existing file it names, into a new
// file only if the question asks for one (or names it), and a block with no name
// into the open file only if the question was about that file and it is in its language.
export function applyTarget(block, question, place) {
  const { files = [], folderName = "", root = "", openPath = "" } = place;
  if (block.quote || (!block.path && NOT_FILES.has(block.lang))) return "";
  const asked = askedFiles(question, files, folderName);
  const text = String(question?.text || "");
  if (block.path) {
    const found = resolvePath(block.path, files, folderName, root);
    if (!found) return "";
    if (found.exists) return asked.size ? found.path : "";
    return WANTS_FILE.test(text) || mentions(text, found.path.split("/").pop()) ? found.path : "";
  }
  return openPath && asked.has(openPath) && fitsOpenFile(block.lang, openPath) ? openPath : "";
}

// "// ... existing code ...", "# ...", "<!-- ... -->", "/* rest of the code */".
const MARKER = new RegExp(
  "^\\s*(?:\\/\\/+|#+|--|;+|%+|\\/\\*+|<!--|\\{\\/\\*)\\s*" +
  "(?:\\.{3}|…|.*\\b(?:existing code|rest of (?:the )?(?:code|file|function|class|component|module)|unchanged (?:code|lines))\\b)",
  "i");
export const isMarker = (line) => MARKER.test(line);

const norm = (line) => line.trim().replace(/\s+/g, " ");
const meaningful = (line) => /[A-Za-z0-9]/.test(line);
const weight = (line) => (meaningful(line) ? 3 : line.trim() ? 0.5 : 0.25);
const NEW_LINE = 0.5;
const DROPPED_LINE = 0.1;
const MAX_CELLS = 6_000_000;

// Where one piece of the edit goes in the file: the stretch of lines it
// replaces, found from the unchanged lines it shares with the file. Its first
// and last lines must be ones the file has, and no other place may fit as well.
function place(piece, lines, from) {
  const rows = piece.length;
  const cols = lines.length - from;
  if (rows * (cols + 1) > MAX_CELLS) return null;
  const width = cols + 1;
  const keys = piece.map(norm);
  const fileKeys = lines.map(norm);
  let previous = new Float64Array(width);
  const moves = new Uint8Array((rows + 1) * width);
  for (let i = 1; i <= rows; i++) {
    const current = new Float64Array(width);
    current[0] = previous[0] - NEW_LINE;
    moves[i * width] = 1;
    for (let k = 1; k <= cols; k++) {
      let best = previous[k] - NEW_LINE;
      let move = 1;
      const dropped = current[k - 1] - DROPPED_LINE;
      if (dropped > best) { best = dropped; move = 2; }
      if (keys[i - 1] === fileKeys[from + k - 1]) {
        const kept = previous[k - 1] + weight(piece[i - 1]);
        if (kept >= best) { best = kept; move = 0; }
      }
      current[k] = best;
      moves[i * width + k] = move;
    }
    previous = current;
  }
  let end = 0;
  for (let k = 1; k <= cols; k++) if (previous[k] > previous[end]) end = k;
  const score = previous[end];
  const pairs = [];
  let i = rows;
  let k = end;
  while (i > 0) {
    const move = moves[i * width + k];
    if (move === 0) {
      pairs.push([i - 1, from + k - 1]);
      i--;
      k--;
    } else if (move === 1) i--;
    else k--;
  }
  pairs.reverse();
  if (!pairs.some(([at]) => meaningful(piece[at]))) return null;
  const firstLine = piece.findIndex((line) => line.trim());
  const lastLine = piece.length - 1 - [...piece].reverse().findIndex((line) => line.trim());
  if (pairs[0][0] !== firstLine || pairs[pairs.length - 1][0] !== lastLine) return null;
  const start = pairs[0][1];
  const stop = pairs[pairs.length - 1][1] + 1;
  for (let other = 1; other <= cols; other++) {
    const at = from + other;
    if ((at <= start || at > stop) && previous[other] >= score - 1e-9) return null;
  }
  return { start, stop, pairs };
}

// A piece shown with less indentation than the file has (a method shown
// outside its class) gets the file's indentation back.
function reindent(piece, pairs, lines) {
  let extra = null;
  for (const [at, line] of pairs) {
    if (!piece[at].trim()) continue;
    const mine = /^\s*/.exec(piece[at])[0];
    const theirs = /^\s*/.exec(lines[line])[0];
    if (!theirs.endsWith(mine)) return piece;
    const gap = theirs.slice(0, theirs.length - mine.length);
    if (extra === null) extra = gap;
    else if (extra !== gap) return piece;
  }
  return extra ? piece.map((line) => (line.trim() ? extra + line : line)) : piece;
}

const trimBlank = (piece) => {
  let a = 0;
  let b = piece.length;
  while (a < b && !piece[a].trim()) a++;
  while (b > a && !piece[b - 1].trim()) b--;
  return piece.slice(a, b);
};

// The file with the edit in it, or null when the edit cannot be placed for
// certain. Then a model has to do it.
export function mergeEdit(original, code) {
  const crlf = original.includes("\r\n");
  const text = original.replace(/\r\n/g, "\n");
  const edit = code.replace(/\r\n/g, "\n").replace(/\n+$/, "");
  const finish = (lines, endsWithBreak) => {
    const joined = lines.join("\n") + (endsWithBreak ? "\n" : "");
    return crlf ? joined.replace(/\n/g, "\r\n") : joined;
  };
  const editLines = edit.split("\n");
  const marked = editLines.some(isMarker);
  if (!text.trim()) return marked ? null : finish(editLines, true);
  const endsWithBreak = text.endsWith("\n");
  const lines = (endsWithBreak ? text.slice(0, -1) : text).split("\n");

  if (!marked) {
    const part = trimBlank(editLines);
    const spot = part.length ? place(part, lines, 0) : null;
    if (spot) {
      return finish([...lines.slice(0, spot.start), ...reindent(part, spot.pairs, lines), ...lines.slice(spot.stop)],
        endsWithBreak);
    }
    // A rewrite of the whole file: it starts and ends where the file does and
    // is at least half as long, or does one of those and is nearly as long.
    const mine = editLines.filter(meaningful).map(norm);
    const theirs = lines.filter(meaningful).map(norm);
    const first = mine[0] === theirs[0];
    const last = mine.at(-1) === theirs.at(-1);
    if ((first && last && mine.length >= 0.5 * theirs.length) || ((first || last) && mine.length >= 0.8 * theirs.length)) {
      return finish(editLines, endsWithBreak);
    }
    return null;
  }

  const pieces = [];
  let piece = [];
  for (const line of editLines) {
    if (isMarker(line)) {
      pieces.push(piece);
      piece = [];
    } else {
      piece.push(line);
    }
  }
  pieces.push(piece);
  const out = [];
  let cursor = 0;
  for (const part of pieces.map(trimBlank).filter((part) => part.length)) {
    const spot = place(part, lines, cursor);
    if (!spot) return null;
    out.push(...lines.slice(cursor, spot.start), ...reindent(part, spot.pairs, lines));
    cursor = spot.stop;
  }
  if (!out.length && cursor === 0) return null;
  out.push(...lines.slice(cursor));
  return finish(out, endsWithBreak);
}

// What a model wrote back when asked for the whole file: the code in its
// longest fenced block, or all of it when it used no fence. Null if cut off.
export function fileFromReply(reply) {
  const text = reply.replace(/\r\n/g, "\n");
  const fence = /^(`{3,}|~{3,})[^\n]*\n([\s\S]*?)\n?\1[ \t]*$/gm;
  let best = null;
  let match;
  while ((match = fence.exec(text))) if (best === null || match[2].length > best.length) best = match[2];
  if (best !== null) return best;
  if (/^\s*(`{3,}|~{3,})/.test(text)) return null;
  return text.trim() ? text.replace(/^\n+/, "") : null;
}

// Asking a model to apply an edit, like Cursor's apply model: the whole file
// back, with the edit in place and everything else as it was.
export function applyPrompt(path, original, code) {
  const fence = (body) => {
    let mark = "```";
    while (body.includes(mark)) mark += "`";
    return mark;
  };
  const a = fence(original);
  const b = fence(code);
  return [
    {
      role: "system",
      content: "You merge an edit into a file. You are given the file's full current text and an edit to it. " +
        "The edit may leave out unchanged code and mark each gap with a comment such as \"// ... existing code ...\". " +
        "Write the complete new file: every line of the current file, with the edit applied where it belongs. " +
        "Keep everything the edit does not change exactly as it is, including comments, blank lines and indentation. " +
        "Never write \"... existing code ...\" comments yourself, and never shorten the file. " +
        "Reply with only the complete file in one fenced code block, and nothing else.",
    },
    {
      role: "user",
      content: "Current file " + path + ":\n" + a + "\n" + original + "\n" + a + "\n\nEdit to apply:\n" + b + "\n" + code +
        "\n" + b + "\n\nWrite the complete new " + path + ".",
    },
  ];
}
