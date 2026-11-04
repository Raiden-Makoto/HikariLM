// What reading an answer aloud says: its words, from marked's tokens, as short lines.
// Code blocks and big tables are named, not read; small tables are read row by row; math
// is said in words ("x squared plus 1 equals 0"), or named when it is too much to say;
// links by their text, pictures by their alt text, emoji and markup not at all. Thinking is
// never in the answer's text. No imports, so the tests run it in node with the real marked.

const LANGUAGES = {
  py: "Python", python: "Python", js: "JavaScript", javascript: "JavaScript", ts: "TypeScript",
  typescript: "TypeScript", jsx: "JavaScript", tsx: "TypeScript", sh: "shell", bash: "shell", zsh: "shell",
  shell: "shell", console: "shell", html: "HTML", css: "CSS", json: "JSON", yaml: "YAML", yml: "YAML",
  sql: "SQL", c: "C", cpp: "C plus plus", "c++": "C plus plus", cs: "C sharp", csharp: "C sharp", java: "Java",
  go: "Go", rust: "Rust", rs: "Rust", swift: "Swift", kotlin: "Kotlin", rb: "Ruby", ruby: "Ruby", php: "PHP",
  r: "R", tex: "LaTeX", latex: "LaTeX", md: "Markdown", markdown: "Markdown", xml: "XML", toml: "TOML",
  diff: "diff", text: "text", txt: "text", mermaid: "diagram",
};
const ENTITIES = { amp: "&", lt: "<", gt: ">", quot: '"', apos: "'", nbsp: " ", "#39": "'" };
// Symbols a voice would skip or misread.
const SYMBOLS = [
  [/(\d)\s*°\s*([CF])\b/g, "$1 degrees $2"], [/°/g, " degrees"], [/(\d)\s*%/g, "$1 percent"],
  [/[→⟶]/g, " to "], [/[⇒⟹]/g, " implies "], [/⇔/g, " if and only if "], [/←/g, " from "],
  [/×/g, " times "], [/÷/g, " divided by "], [/≈/g, " approximately "], [/≠/g, " is not equal to "],
  [/≤/g, " is at most "], [/≥/g, " is at least "], [/±/g, " plus or minus "], [/√/g, " the square root of "],
  [/∞/g, " infinity "], [/π/g, " pi "], [/∑/g, " the sum of "], [/∫/g, " the integral of "], [/∆|Δ/g, " delta "],
  [/\s[—–]\s|[—]/g, ", "], [/…/g, "..."], [/&/g, " and "], [/\s\*\s/g, " times "],
  [/\s?(->|=>)\s?/g, " to "], [/\s(<=|=<)\s/g, " is at most "], [/\s>=\s/g, " is at least "],
  [/\s<\s/g, " is less than "], [/\s>\s/g, " is greater than "], [/\s==?\s/g, " equals "],
  [/(\w)\s\+\s(\w)/g, "$1 plus $2"], [/(\w)\^2\b/g, "$1 squared"], [/(\w)\^(\w+)/g, "$1 to the power of $2"],
];
const EMOJI = /[\p{Extended_Pictographic}\u{1F1E6}-\u{1F1FF}\u{FE0F}\u{200D}\u{20E3}]/gu;

export function decodeEntities(text) {
  return String(text).replace(/&(#?\w+);/g, (whole, name) => {
    if (ENTITIES[name] !== undefined) return ENTITIES[name];
    const code = /^#x([0-9a-f]+)$/i.exec(name) ? parseInt(name.slice(2), 16) : /^#(\d+)$/.exec(name) ? Number(name.slice(1)) : NaN;
    return Number.isFinite(code) ? String.fromCodePoint(code) : whole;
  });
}

export function cleanSpeech(text) {
  let out = decodeEntities(text).replace(EMOJI, "");
  for (const [pattern, words] of SYMBOLS) out = out.replace(pattern, words);
  // Markup a model left in plain text, and footnote marks like [1].
  out = out.replace(/\[\d+\]/g, "").replace(/[*_`#~|<>{}\\]+/g, " ");
  return out.replace(/\s+([,.;:!?])/g, "$1").replace(/\s+/g, " ").trim();
}

const domain = (href) => {
  try {
    return new URL(href).hostname.replace(/^www\./, "");
  } catch (error) {
    return "";
  }
};

function inline(tokens) {
  let out = "";
  for (const token of tokens || []) {
    switch (token.type) {
      case "text":
      case "escape":
        out += token.tokens ? inline(token.tokens) : token.text;
        break;
      case "strong": case "em": case "del":
        out += inline(token.tokens);
        break;
      case "codespan": {
        const code = decodeEntities(token.text);
        out += code.length <= 40 ? " " + code.replace(/[_/]+/g, " ") + " " : " some code ";
        break;
      }
      case "link": {
        const text = inline(token.tokens).trim();
        const bare = token.autolink || text === token.href || /^(https?:|www\.)/i.test(text);
        out += bare ? (domain(token.href) ? " a link to " + domain(token.href) + " " : " a link ") : text;
        break;
      }
      case "image":
        out += token.text ? " a picture: " + token.text + ". " : " a picture ";
        break;
      case "inlineMath":
        out += " " + mathWords(token.text, token.display) + " ";
        break;
      case "br":
        out += ". ";
        break;
      case "html":
        out += /^<br\s*\/?>$/i.test(token.text.trim()) ? ". " : "";
        break;
      case "checkbox":
        break;
      default:
        out += token.tokens ? inline(token.tokens) : token.text || "";
    }
  }
  return out;
}

// Sentences, so the first one is heard while the rest are made. A long one is cut at a comma.
export function sentences(text) {
  const clean = cleanSpeech(text);
  if (!clean) return [];
  const parts = clean.split(/(?<=[.!?])\s+(?=["'(]?[A-Z0-9])/);
  const out = [];
  for (const part of parts) {
    if (part.length <= 300) {
      out.push(part);
      continue;
    }
    let rest = part;
    while (rest.length > 300) {
      const cut = Math.max(rest.lastIndexOf(", ", 299), rest.lastIndexOf("; ", 299));
      const space = rest.lastIndexOf(" ", 300);
      const at = cut > 80 ? cut + 1 : space > 80 ? space : 300;
      out.push(rest.slice(0, at).trim());
      rest = rest.slice(at).trim();
    }
    if (rest) out.push(rest);
  }
  return out.filter((line) => /[\p{L}\p{N}]/u.test(line));
}

const end = (text) => (/[.!?:]$/.test(text.trim()) ? text.trim() : text.trim() + ".");
const plural = (n, word) => n + " " + word + (n === 1 ? "" : "s");

function table(token) {
  const header = token.header.map((cell) => cleanSpeech(inline(cell.tokens)));
  const rows = token.rows.map((row) => row.map((cell) => cleanSpeech(inline(cell.tokens))));
  if (rows.length <= 6 && header.length <= 4) {
    const said = [end("A table with " + plural(header.length, "column") + ": " + header.filter(Boolean).join(", "))];
    for (const row of rows) {
      said.push(end(row.map((cell, i) => (header[i] && cell ? header[i] + ", " + cell : cell)).filter(Boolean).join("; ")));
    }
    return said;
  }
  return [end("There is a table here with " + plural(rows.length, "row") + " and " + plural(header.length, "column") +
    (header.some(Boolean) ? ": " + header.filter(Boolean).join(", ") : ""))];
}

function blocks(tokens, out) {
  for (const token of tokens || []) {
    switch (token.type) {
      case "heading":
        out.push(...sentences(end(inline(token.tokens))));
        break;
      case "paragraph":
      case "text":
        out.push(...sentences(token.tokens ? inline(token.tokens) : token.text));
        break;
      case "list":
        for (const item of token.items) {
          const said = [];
          blocks(item.tokens, said);
          if (said.length) said[said.length - 1] = end(said[said.length - 1]);
          out.push(...said);
        }
        break;
      case "blockquote":
        blocks(token.tokens, out);
        break;
      case "code": {
        const lang = String(token.lang || "").trim().split(/[\s:{]/)[0].toLowerCase().replace(/^\./, "");
        const name = LANGUAGES[lang] || (/^[a-z][\w+#-]{0,15}$/.test(lang) ? lang : "");
        const lines = String(token.text || "").split("\n").length;
        out.push(name === "diagram" ? "There is a diagram here." :
          "There is " + (name ? (/^[aeiou]/i.test(name) ? "an " : "a ") + name + " code block" : "a code block") +
          " here, " + plural(lines, "line") + " long.");
        break;
      }
      case "table":
        out.push(...table(token));
        break;
      case "blockMath":
        out.push(end(mathWords(token.text, true)));
        break;
      case "html": {
        const text = token.text.replace(/<[^>]*>/g, " ");
        out.push(...sentences(text));
        break;
      }
      default:
        break;
    }
  }
  return out;
}

// An answer's words without its thinking: <think>…</think>, and in chats saved before the
// helper split stray </think> tags out, everything up to the last </think>.
export function answerText(text) {
  const words = String(text || "").replace(/<think>[\s\S]*?<\/think>/g, "");
  const end = words.lastIndexOf("</think>");
  return (end < 0 ? words : words.slice(end + 8)).replace(/<\/?think>/g, "");
}

// The lines to say for an answer. `plots` is how many plots the Python tool made, which
// are shown above its words.
export function spokenLines(tokens, { plots = 0 } = {}) {
  const out = [];
  if (plots) out.push(plots === 1 ? "There is a plot above." : "There are " + plots + " plots above.");
  blocks(tokens, out);
  return out.map((line) => line.trim()).filter(Boolean);
}

// --- math in words ---------------------------------------------------------------

const GREEK = ("alpha beta gamma delta epsilon varepsilon zeta eta theta vartheta iota kappa lambda mu nu xi " +
  "omicron pi varpi rho varrho sigma varsigma tau upsilon phi varphi chi psi omega").split(" ");
const WORDS = {
  "+": "plus", "-": "minus", "=": "equals", "<": "is less than", ">": "is greater than", "/": "over", "*": "times",
  ",": ",", ";": ",", ":": ",", "!": "factorial", "'": "prime",
  neq: "is not equal to", ne: "is not equal to", le: "is less than or equal to", leq: "is less than or equal to",
  ge: "is greater than or equal to", geq: "is greater than or equal to", ll: "is much less than", gg: "is much greater than",
  approx: "is approximately", sim: "is about", simeq: "is approximately", cong: "is congruent to", equiv: "is equivalent to",
  propto: "is proportional to", pm: "plus or minus", mp: "minus or plus", times: "times", cdot: "times", ast: "times",
  div: "divided by", to: "goes to", rightarrow: "goes to", longrightarrow: "goes to", mapsto: "maps to",
  Rightarrow: "implies", implies: "implies", Longrightarrow: "implies", iff: "if and only if", Leftrightarrow: "if and only if",
  leftarrow: "from", in: "is in", notin: "is not in", ni: "contains", subset: "is a subset of", subseteq: "is a subset of",
  supset: "is a superset of", cup: "union", cap: "intersection", setminus: "minus", emptyset: "the empty set",
  varnothing: "the empty set", infty: "infinity", partial: "partial", nabla: "nabla", forall: "for all",
  exists: "there exists", neg: "not", land: "and", lor: "or", wedge: "and", vee: "or", circ: "composed with",
  angle: "angle", perp: "is perpendicular to", parallel: "is parallel to", degree: "degrees", prime: "prime",
  cdots: "and so on", ldots: "and so on", dots: "and so on", vdots: "and so on", ddots: "and so on",
  therefore: "therefore", because: "because", mid: "such that", vert: "", Vert: "", lvert: "", rvert: "",
  hbar: "h bar", ell: "l", Re: "the real part of", Im: "the imaginary part of", star: "star", dagger: "dagger",
  oplus: "direct sum", otimes: "tensor", bullet: "times", colon: ",", gets: "gets",
};
const FUNCTIONS = {
  sin: "sine", cos: "cosine", tan: "tangent", cot: "cotangent", sec: "secant", csc: "cosecant",
  arcsin: "arc sine", arccos: "arc cosine", arctan: "arc tangent", sinh: "hyperbolic sine", cosh: "hyperbolic cosine",
  tanh: "hyperbolic tangent", "log": "log", ln: "the natural log", lg: "log", exp: "e to the", det: "the determinant",
  dim: "the dimension", ker: "the kernel", deg: "the degree", gcd: "the greatest common divisor",
  max: "the maximum", min: "the minimum", sup: "the supremum", inf: "the infimum", arg: "the argument",
  Pr: "the probability", lim: "the limit",
};
const BIG = { sum: "the sum", prod: "the product", int: "the integral", iint: "the double integral",
  iiint: "the triple integral", oint: "the contour integral", bigcup: "the union", bigcap: "the intersection" };
const ACCENTS = { hat: "hat", bar: "bar", overline: "bar", tilde: "tilde", widetilde: "tilde", dot: "dot", ddot: "double dot",
  vec: "vector", overrightarrow: "vector", underline: "" };
// Words written out: \text{if } x > 0 keeps its spaces.
const TEXT = new Set(["text", "textrm", "textbf", "textit", "textsf", "texttt", "mbox", "emph", "operatorname"]);
// Styles of letters: \mathbf{v} is v.
const STYLE = new Set(["mathrm", "mathbf", "mathit", "mathsf", "mathtt", "mathcal", "mathbb", "mathfrak", "mathscr",
  "boldsymbol", "bm", "boxed", "pmb"]);
const SKIP = new Set(["left", "right", "big", "Big", "bigg", "Bigg", "bigl", "bigr", "Bigl", "Bigr", "biggl", "biggr",
  "bigm", "Bigm", ",", ";", ":", "!", " ", "quad", "qquad", "displaystyle", "textstyle", "scriptstyle", "limits",
  "nolimits", "middle", "nonumber", "notag", "label", "tag", "phantom", "hfill", "{", "}"]);
const SIGNS = { "%": "percent", "$": "dollars", "#": "number", "&": "and", "_": "underscore", "|": "", "\\": "," };
// Words after which a minus is a sign, not taking away.
const OPERATORS = new Set(("equals than to plus minus over times by of in and or implies approximately about " +
  "negative choose from as if otherwise").split(" "));
const ORDINALS = { 2: "squared", 3: "cubed" };
const ROOTS = { 2: "the square root of", 3: "the cube root of" };

class TooMuch extends Error {}

function lex(tex) {
  const out = [];
  let i = 0;
  while (i < tex.length) {
    const c = tex[i];
    if (c === "\\") {
      const name = /^\\([a-zA-Z]+|.)/.exec(tex.slice(i));
      if (!name) throw new TooMuch();
      i += name[0].length;
      if (TEXT.has(name[1])) {
        const text = /^\s*\{([^{}]*)\}/.exec(tex.slice(i));
        if (!text) throw new TooMuch();
        out.push({ text: text[1].trim() });
        i += text[0].length;
      } else {
        out.push({ cmd: name[1] });
      }
    } else if (/\s/.test(c)) {
      i++;
    } else if (/\d/.test(c)) {
      const number = /^\d+(?:\.\d+)?/.exec(tex.slice(i))[0];
      out.push({ ch: number });
      i += number.length;
    } else {
      out.push({ ch: c });
      i++;
    }
  }
  return out;
}

// A small reader for the LaTeX models write: atoms, groups, ^ and _, and the commands above.
// Anything it does not know throws TooMuch, and the math is named instead of misread.
function readItems(items) {
  let at = 0;
  const peek = () => items[at];
  const closing = (ch) => {
    if (items[at]?.ch !== ch) throw new TooMuch();
    at++;
  };
  const group = () => {
    if (!items[at]) throw new TooMuch();
    if (items[at].ch === "{") {
      at++;
      const words = sequence("}");
      closing("}");
      return words;
    }
    return atom();
  };
  const optional = () => {
    if (peek()?.ch !== "[") return null;
    at++;
    const words = sequence("]");
    closing("]");
    return words;
  };
  const scripts = () => {
    let sub = null;
    let sup = null;
    while (peek() && (peek().ch === "_" || peek().ch === "^")) {
      const kind = items[at++].ch;
      if (kind === "_") sub = group();
      else sup = group();
    }
    return { sub, sup };
  };
  const power = (base, sup) => {
    const s = sup.trim();
    const of = base || "all";
    if (ORDINALS[s]) return of + " " + ORDINALS[s];
    if (s === "prime" || s === "prime prime") return of + " " + s;
    if (s === "composed with" || s === "degrees") return of + " degrees";
    if (s === "negative 1") return of + " inverse";
    if (s === "T" || s === "top") return of + " transpose";
    return of + " to the power of " + s + ",";
  };
  const withScripts = (base) => {
    const { sub, sup } = scripts();
    let words = base;
    if (sub !== null) words += /^[\d\s]+$/.test(sub) ? " " + sub : " sub " + sub;
    if (sup !== null) words = power(words, sup);
    return words;
  };
  const environment = () => {
    const kind = group().replace(/\s/g, "");
    let depth = 1;
    const start = at;
    while (at < items.length) {
      if (items[at].cmd === "begin") depth++;
      if (items[at].cmd === "end" && --depth === 0) break;
      at++;
    }
    if (at >= items.length) throw new TooMuch();
    const inside = items.slice(start, at);
    at++;
    group();
    if (/matrix|array/.test(kind)) return "a matrix";
    return readItems(inside);
  };
  const atom = () => {
    const item = items[at++];
    if (!item) throw new TooMuch();
    if (item.text !== undefined) return item.text;
    if (item.ch !== undefined) {
      if (item.ch === "{") {
        const words = sequence("}");
        closing("}");
        return words;
      }
      if ("()[]|&~".includes(item.ch)) return "";
      if (item.ch === ".") return "";
      if (WORDS[item.ch] !== undefined) return WORDS[item.ch];
      if (/^[\d.]+$|^\p{L}$/u.test(item.ch)) return item.ch;
      throw new TooMuch();
    }
    const name = item.cmd;
    if (SKIP.has(name)) return "";
    if (SIGNS[name] !== undefined) return SIGNS[name];
    if (name === "frac" || name === "dfrac" || name === "tfrac" || name === "cfrac") {
      const top = group();
      const bottom = group();
      const simple = (w) => !/\s/.test(w.trim());
      return simple(top) && simple(bottom) ? top + " over " + bottom : "the fraction " + top + ", over " + bottom + ",";
    }
    if (name === "binom") return group() + " choose " + group();
    if (name === "sqrt") {
      const index = optional();
      const inside = group();
      const root = index === null ? ROOTS[2] : ROOTS[index.trim()] || "the " + index + "th root of";
      return root + " " + inside + ",";
    }
    if (STYLE.has(name)) return group();
    if (name in ACCENTS) {
      const inside = group();
      return ACCENTS[name] ? inside + " " + ACCENTS[name] : inside;
    }
    if (name === "begin") return environment();
    if (GREEK.includes(name)) return name.replace(/^var/, "");
    if (GREEK.includes(name.toLowerCase())) return "capital " + name.toLowerCase();
    if (name in BIG) {
      const { sub, sup } = scripts();
      let words = BIG[name];
      if (sub !== null && sup !== null) words += " from " + sub + " to " + sup;
      else if (sub !== null) words += " over " + sub;
      return words + " of";
    }
    if (name === "lim" || name === "limsup" || name === "liminf") {
      const { sub } = scripts();
      return "the limit" + (sub !== null ? " as " + sub : "") + " of";
    }
    if (name in FUNCTIONS) {
      const { sub, sup } = scripts();
      let words = FUNCTIONS[name];
      if (sub !== null) words += " base " + sub;
      if (sup !== null) words = power(words, sup).replace(/,$/, "");
      return name === "exp" ? words : words + " of";
    }
    if (WORDS[name] !== undefined) return WORDS[name];
    throw new TooMuch();
  };
  const sequence = (until) => {
    const words = [];
    while (at < items.length && !(until && items[at].ch === until)) {
      const item = items[at];
      const last = words.filter((w) => w !== "").pop() || "";
      // A minus with nothing before it to take from is a sign: -3 is "negative 3".
      if (item.ch === "-" && (!last || last === "," || OPERATORS.has(last))) {
        at++;
        words.push("negative");
        continue;
      }
      // f(x) is "f of x"; 2(x+1) and n(n+1) are times.
      if (item.ch === "(" && /(^|\s)[fgh]( prime)*$/.test(words.filter((w) => w !== "").join(" "))) words.push("of");
      else if (item.ch === "(" && /^(\p{L}|[\d.]+)$/u.test(last)) words.push("times");
      words.push(withScripts(atom()));
    }
    if (until && at >= items.length) throw new TooMuch();
    return words.filter((w) => w !== "").join(" ");
  };
  return sequence(null);
}

// Math in words, or "an equation" (block) / "a formula" when it is too much to say well.
export function mathWords(tex, display = false) {
  const fallback = display ? "There is an equation here" : "a formula";
  const source = String(tex || "").trim();
  if (!source) return "";
  if (source.length > 160) return fallback;
  try {
    const words = readItems(lex(source)).replace(/\s+,/g, ",").replace(/,(\s*,)+/g, ",").replace(/^,\s*/, "")
      .replace(/,\s*$/, "").replace(/\s+/g, " ").trim();
    if (!words || words.split(" ").length > 40) return fallback;
    return words;
  } catch (error) {
    if (error instanceof TooMuch) return fallback;
    throw error;
  }
}
