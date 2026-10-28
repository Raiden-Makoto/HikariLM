// Turning replies into HTML: Markdown, a few safe tags, math, and the folded thinking.
import { scrollLog } from "./core.js";

// Replies are Markdown, sometimes with LaTeX. Raw HTML from a model is shown
// as text, apart from a few formatting tags, links go only to web and mail addresses, and pictures on other
// sites are never fetched. KaTeX writes the math as MathML, which the
// browser lays out itself, so it needs no fonts; it loads only once a reply
// has math in it.
let markdown = null;
let katexLoad = null;
const escapeHtml = (text) =>
  String(text).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const safeHref = (href) => (/^(https?:|mailto:|#)/i.test(href || "") ? href : "");
// Models often write <br> in table cells, or a small HTML table. These tags are
// drawn, with their attributes dropped; any other tag shows as text.
const SAFE_TAGS = new Set(("br hr b strong i em u s del ins sub sup mark small code kbd pre p ul ol li " +
  "blockquote table thead tbody tfoot tr th td h1 h2 h3 h4 h5 h6 details summary").split(" "));
function safeHtml(text) {
  return String(text).split(/(<\/?[a-z][a-z0-9]*(?:\s[^<>]*)?\/?>)/i).map((part) => {
    const tag = /^<(\/?)([a-z][a-z0-9]*)(?:\s[^<>]*)?\/?>$/i.exec(part);
    if (!tag) return escapeHtml(part);
    const name = tag[2].toLowerCase();
    if (name === "think") return "";
    return SAFE_TAGS.has(name) ? "<" + tag[1] + name + ">" : escapeHtml(part);
  }).join("");
}

function mathHtml(tex, display) {
  if (window.katex) {
    return window.katex.renderToString(tex, {
      displayMode: display, output: "mathml", throwOnError: false, strict: "ignore", maxSize: 20, maxExpand: 500,
    });
  }
  loadKatex();
  return '<span class="math-pending" data-display="' + display + '">' + escapeHtml(tex) + "</span>";
}

function loadKatex() {
  katexLoad ??= new Promise((resolve, reject) => {
    const script = document.createElement("script");
    script.src = "/vendor/katex.min.js";
    script.onload = resolve;
    script.onerror = reject;
    document.head.append(script);
  }).then(() => {
    for (const pending of document.querySelectorAll(".math-pending")) {
      pending.outerHTML = mathHtml(pending.textContent, pending.dataset.display === "true");
    }
  }, () => {
    // Without KaTeX the math stays as the model wrote it.
  });
}

const blockMath = {
  name: "blockMath",
  level: "block",
  start: (src) => {
    const found = /(^|\n) {0,3}(\$\$|\\\[)/.exec(src);
    return found ? found.index + found[1].length : undefined;
  },
  tokenizer(src) {
    const found = /^ {0,3}(?:\$\$([\s\S]+?)\$\$|\\\[([\s\S]+?)\\\])[ \t]*(?:\n+|$)/.exec(src);
    if (found) return { type: "blockMath", raw: found[0], text: (found[1] ?? found[2]).trim() };
  },
  renderer: (token) => '<div class="math-block">' + mathHtml(token.text, true) + "</div>",
};
const inlineMath = {
  name: "inlineMath",
  level: "inline",
  start: (src) => {
    const index = src.search(/\$|\\[([]/);
    return index < 0 ? undefined : index;
  },
  tokenizer(src) {
    let found = /^\$\$([\s\S]+?)\$\$/.exec(src) || /^\\\[([\s\S]+?)\\\]/.exec(src);
    if (found) return { type: "inlineMath", raw: found[0], text: found[1].trim(), display: true };
    // $5 and $10 is money, not math: no space just inside the dollars, no digit after.
    found = /^\\\(([\s\S]+?)\\\)/.exec(src) || /^\$(?![\s$])((?:\\.|[^\\$\n])+?)(?<![\s\\])\$(?!\d)/.exec(src);
    if (found) return { type: "inlineMath", raw: found[0], text: found[1], display: false };
  },
  renderer: (token) => mathHtml(token.text, token.display),
};

// A failed import is remembered by the browser for that address, so a retry asks for a new one.
const loadMarked = (attempt = 0) =>
  import("/vendor/marked.esm.js" + (attempt ? "?try=" + attempt : "")).catch((error) => {
    if (attempt >= 2) throw error;
    return new Promise((resolve) => setTimeout(resolve, 1500)).then(() => loadMarked(attempt + 1));
  });

export function renderMarkdown(element, text) {
  if (!markdown) {
    element.textContent = text;
    element.classList.add("md-pending");
    return;
  }
  element.classList.remove("md-pending");
  element.classList.add("md");
  try {
    element.innerHTML = markdown.parse(text);
  } catch (error) {
    element.textContent = text;
  }
}

// While a reply streams, draw it at most every 80 ms rather than on every token.
export function streamMarkdown(element, text) {
  element.streamText = text;
  element.streamTimer ??= setTimeout(() => {
    element.streamTimer = null;
    renderMarkdown(element, element.streamText);
    scrollLog();
  }, 80);
}

export function finishMarkdown(element, text) {
  clearTimeout(element.streamTimer);
  element.streamTimer = null;
  renderMarkdown(element, text);
}

// The folded thinking line: "Thinking for 4s" while it thinks, then "Thought
// for 12s". The text is only put in the page when the line is opened.
export function showThought(view, text, { seconds = null, live = false, cut = false } = {}) {
  view.full = text;
  view.box.hidden = !text;
  const shown = seconds === null ? "" : " for " + Math.max(1, Math.round(seconds)) + "s";
  view.label.textContent = (live ? "Thinking" + (shown || "…") : "Thought" + shown) +
    (cut ? " · stopped at the thinking limit" : "");
  view.box.classList.toggle("live", live);
  if (view.box.open) {
    const atEnd = view.text.scrollHeight - view.text.scrollTop - view.text.clientHeight < 24;
    view.text.textContent = text;
    if (live && atEnd) view.text.scrollTop = view.text.scrollHeight;
  }
}

// Chats saved before the helper split stray </think> tags out still have the
// reasoning in the answer: everything up to the last </think> is reasoning.
export function splitStrayThinking(text, thought) {
  const end = text.lastIndexOf("</think>");
  if (end < 0) return { text, thought };
  const before = text.slice(0, end).replace(/<\/?think>/g, "").trim();
  return { text: text.slice(end + 8).trim(), thought: [thought, before].filter(Boolean).join("\n\n") };
}

// Listeners and first steps, run by main.js once every part is loaded.
export function start() {
  loadMarked().then(({ Marked }) => {
    markdown = new Marked({ gfm: true });
    markdown.use({
      extensions: [blockMath, inlineMath],
      renderer: {
        html: ({ text }) => safeHtml(text),
        link({ href, title, tokens }) {
          const label = this.parser.parseInline(tokens);
          const safe = safeHref(href);
          if (!safe) return label;
          return '<a href="' + escapeHtml(safe) + '"' + (title ? ' title="' + escapeHtml(title) + '"' : "") +
            ' target="_blank" rel="noopener noreferrer">' + label + "</a>";
        },
        image({ href, text }) {
          const safe = safeHref(href);
          const label = escapeHtml(text || "picture");
          return safe ? '<a href="' + escapeHtml(safe) + '" target="_blank" rel="noopener noreferrer">' + label + "</a>" : label;
        },
      },
    });
    for (const waiting of document.querySelectorAll(".md-pending")) renderMarkdown(waiting, waiting.textContent);
  }).catch(() => {
    // Without marked, replies stay plain text.
  });
}
