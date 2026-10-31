// The code editor, built on CodeMirror 6 and bundled into web/vendor/codemirror.js.
// The page imports createEditor() from that one file. Run build.sh after changing this.

import { Compartment, EditorState, Prec } from "@codemirror/state";
import {
  EditorView, crosshairCursor, drawSelection, dropCursor, highlightActiveLine, highlightActiveLineGutter,
  highlightSpecialChars, keymap, lineNumbers, rectangularSelection,
} from "@codemirror/view";
import {
  defaultKeymap, history, historyKeymap, indentLess, indentMore,
} from "@codemirror/commands";
import {
  HighlightStyle, StreamLanguage, bracketMatching, foldGutter, foldKeymap, indentOnInput, indentUnit,
  syntaxHighlighting,
} from "@codemirror/language";
import {
  acceptCompletion, autocompletion, closeBrackets, closeBracketsKeymap, completeAnyWord, completionKeymap,
  completionStatus,
} from "@codemirror/autocomplete";
import { highlightSelectionMatches, searchKeymap } from "@codemirror/search";
import { getChunks, getOriginalDoc, unifiedMergeView } from "@codemirror/merge";
import { tags as t } from "@lezer/highlight";

import { cpp } from "@codemirror/lang-cpp";
import { css } from "@codemirror/lang-css";
import { go } from "@codemirror/lang-go";
import { html } from "@codemirror/lang-html";
import { java } from "@codemirror/lang-java";
import { javascript } from "@codemirror/lang-javascript";
import { json } from "@codemirror/lang-json";
import { markdown } from "@codemirror/lang-markdown";
import { php } from "@codemirror/lang-php";
import { python } from "@codemirror/lang-python";
import { rust } from "@codemirror/lang-rust";
import { sql } from "@codemirror/lang-sql";
import { xml } from "@codemirror/lang-xml";
import { yaml } from "@codemirror/lang-yaml";

import { dockerFile } from "@codemirror/legacy-modes/mode/dockerfile";
import { csharp, kotlin, scala } from "@codemirror/legacy-modes/mode/clike";
import { lua } from "@codemirror/legacy-modes/mode/lua";
import { perl } from "@codemirror/legacy-modes/mode/perl";
import { properties } from "@codemirror/legacy-modes/mode/properties";
import { r } from "@codemirror/legacy-modes/mode/r";
import { ruby } from "@codemirror/legacy-modes/mode/ruby";
import { shell } from "@codemirror/legacy-modes/mode/shell";
import { stex } from "@codemirror/legacy-modes/mode/stex";
import { swift } from "@codemirror/legacy-modes/mode/swift";
import { toml } from "@codemirror/legacy-modes/mode/toml";

const old = (mode) => () => StreamLanguage.define(mode);

// Language by file extension (and a few whole names). Anything else is plain text.
const BY_EXTENSION = {
  py: python,
  js: () => javascript(), mjs: () => javascript(), jsx: () => javascript({ jsx: true }),
  ts: () => javascript({ typescript: true }), tsx: () => javascript({ jsx: true, typescript: true }),
  c: cpp, h: cpp, cpp, cc: cpp, cxx: cpp, hpp: cpp, hh: cpp,
  java, rs: rust, go, php,
  html, htm: html, css, json, jsonl: json, xml, sql, yaml, yml: yaml,
  md: markdown, markdown, rmd: markdown,
  r: old(r), lua: old(lua), rb: old(ruby), swift: old(swift), kt: old(kotlin), scala: old(scala), cs: old(csharp),
  pl: old(perl), sh: old(shell), bash: old(shell), toml: old(toml), tex: old(stex),
  ini: old(properties), cfg: old(properties),
};
const BY_NAME = { dockerfile: old(dockerFile) };

function languageFor(path) {
  const name = path.split("/").pop().toLowerCase();
  const make = BY_NAME[name] || BY_EXTENSION[name.includes(".") ? name.split(".").pop() : ""];
  return make ? [make()] : [];
}

// The colours come from CSS variables on the page, so the light and dark
// switch recolours the editor with no rebuild of anything.
const theme = EditorView.theme({
  "&": {
    height: "100%",
    color: "var(--ink)",
    backgroundColor: "var(--panel)",
    fontSize: "0.86rem",
    border: "1px solid var(--line)",
    borderRadius: "12px",
  },
  "&.cm-focused": { outline: "none", borderColor: "var(--you)" },
  ".cm-scroller": {
    fontFamily: 'ui-monospace, "Cascadia Mono", Consolas, Menlo, monospace',
    lineHeight: "1.5",
    borderRadius: "12px",
  },
  ".cm-content": { caretColor: "var(--ink)", padding: "0.6rem 0" },
  ".cm-cursor, .cm-dropCursor": { borderLeftColor: "var(--ink)" },
  ".cm-gutters": { backgroundColor: "transparent", color: "var(--muted)", border: "none" },
  ".cm-activeLine": { backgroundColor: "color-mix(in srgb, var(--hover) 55%, transparent)" },
  ".cm-activeLineGutter": { backgroundColor: "transparent", color: "var(--ink)" },
  ".cm-selectionBackground, &.cm-focused .cm-selectionBackground, ::selection": {
    backgroundColor: "color-mix(in srgb, var(--you) 30%, transparent) !important",
  },
  ".cm-selectionMatch": { backgroundColor: "color-mix(in srgb, var(--you) 18%, transparent)" },
  "&.cm-focused .cm-matchingBracket": {
    backgroundColor: "color-mix(in srgb, var(--you) 25%, transparent)",
    outline: "1px solid var(--you)",
  },
  "&.cm-focused .cm-nonmatchingBracket": { backgroundColor: "color-mix(in srgb, var(--danger) 30%, transparent)" },
  ".cm-foldPlaceholder": { backgroundColor: "var(--hover)", border: "none", color: "var(--muted)" },
  ".cm-panels": { backgroundColor: "var(--field)", color: "var(--ink)", borderColor: "var(--line)" },
  ".cm-panels.cm-panels-top": { borderBottom: "1px solid var(--line)" },
  ".cm-panels.cm-panels-bottom": { borderTop: "1px solid var(--line)" },
  ".cm-panel.cm-search": { padding: "0.4rem 0.6rem" },
  ".cm-panel.cm-search input, .cm-panel.cm-search button, .cm-panel.cm-search label": { fontSize: "0.8rem" },
  ".cm-textfield": {
    backgroundColor: "var(--bg)", color: "var(--ink)", border: "1px solid var(--line)", borderRadius: "6px",
  },
  ".cm-button": {
    backgroundImage: "none", backgroundColor: "var(--hover)", color: "var(--ink)",
    border: "1px solid var(--line)", borderRadius: "6px",
  },
  ".cm-searchMatch": { backgroundColor: "color-mix(in srgb, var(--you) 25%, transparent)" },
  ".cm-searchMatch.cm-searchMatch-selected": { backgroundColor: "color-mix(in srgb, var(--you) 45%, transparent)" },
  ".cm-tooltip": { backgroundColor: "var(--field)", color: "var(--ink)", border: "1px solid var(--line)" },
  ".cm-tooltip.cm-tooltip-autocomplete": { borderRadius: "8px", overflow: "hidden" },
  ".cm-tooltip.cm-tooltip-autocomplete > ul": { fontFamily: 'ui-monospace, "Cascadia Mono", Consolas, Menlo, monospace' },
  ".cm-tooltip.cm-tooltip-autocomplete > ul > li": { padding: "0.1rem 0.6rem" },
  ".cm-tooltip.cm-tooltip-autocomplete > ul > li[aria-selected]": {
    backgroundColor: "color-mix(in srgb, var(--you) 35%, transparent)", color: "var(--ink)",
  },
  ".cm-completionMatchedText": { textDecoration: "none", fontWeight: "bold" },
  ".cm-completionDetail": { color: "var(--muted)", fontStyle: "normal" },
  // Changes waiting to be accepted: new lines green, removed lines red above them.
  "&.cm-merge-b .cm-changedLine": { backgroundColor: "color-mix(in srgb, #22c55e 16%, transparent)" },
  "&.cm-merge-b .cm-changedText": { background: "color-mix(in srgb, #22c55e 32%, transparent)" },
  ".cm-deletedChunk": {
    position: "relative",
    backgroundColor: "color-mix(in srgb, #ef4444 14%, transparent)",
    paddingLeft: "6px",
  },
  ".cm-deletedChunk .cm-deletedText": { background: "color-mix(in srgb, #ef4444 32%, transparent)" },
  ".cm-deletedChunk .cm-deletedLine del": { textDecoration: "none", color: "var(--muted)" },
  ".cm-changedLineGutter": { background: "#22c55e" },
  ".cm-deletedLineGutter": { background: "#ef4444" },
  ".cm-deletedChunk .cm-chunkButtons": { position: "absolute", insetInlineEnd: "8px", top: "2px", zIndex: "2" },
  ".cm-deletedChunk .cm-chunkButtons button": {
    font: "inherit", fontFamily: "system-ui, sans-serif", fontSize: "0.74rem", lineHeight: "1.2",
    padding: "0.15rem 0.5rem", margin: "0 0 0 4px", borderRadius: "5px", cursor: "pointer",
    border: "1px solid var(--line)", color: "var(--ink)", background: "var(--panel)",
  },
  ".cm-deletedChunk .cm-chunkButtons button[name=accept]": { background: "#16a34a", borderColor: "#16a34a", color: "#fff" },
});

const colours = HighlightStyle.define([
  { tag: [t.keyword, t.modifier, t.operatorKeyword, t.controlKeyword, t.definitionKeyword, t.moduleKeyword], color: "var(--syn-keyword)" },
  { tag: [t.comment, t.lineComment, t.blockComment, t.docComment], color: "var(--syn-comment)", fontStyle: "italic" },
  { tag: [t.string, t.special(t.string), t.regexp, t.character], color: "var(--syn-string)" },
  { tag: [t.number, t.bool, t.null, t.atom, t.integer, t.float], color: "var(--syn-number)" },
  { tag: [t.function(t.variableName), t.function(t.propertyName), t.macroName], color: "var(--syn-function)" },
  { tag: [t.typeName, t.className, t.namespace, t.tagName], color: "var(--syn-type)" },
  { tag: [t.definition(t.variableName), t.definition(t.propertyName)], color: "var(--syn-definition)" },
  { tag: [t.propertyName, t.attributeName], color: "var(--syn-property)" },
  { tag: [t.operator, t.punctuation, t.separator, t.bracket], color: "var(--syn-punctuation)" },
  { tag: [t.meta, t.processingInstruction, t.annotation, t.labelName], color: "var(--syn-meta)" },
  { tag: [t.heading, t.strong], color: "var(--syn-keyword)", fontWeight: "bold" },
  { tag: t.emphasis, fontStyle: "italic" },
  { tag: [t.link, t.url], color: "var(--syn-function)", textDecoration: "underline" },
  { tag: t.invalid, color: "var(--danger)" },
]);

// Suggestions as you type: the language's own (keywords, names in the file)
// plus any word already in the file. Tab takes the highlighted one. Enter
// still starts a new line, so typing never gets eaten by a suggestion.
// A word with no type merges with the language's suggestion of the same name.
const anyWord = (context) => {
  const found = completeAnyWord(context);
  return found && { ...found, options: found.options.map(({ type, ...option }) => option) };
};
const suggestions = [
  autocompletion({ defaultKeymap: false, icons: false }),
  EditorState.languageData.of(() => [{ autocomplete: anyWord }]),
  Prec.high(keymap.of(completionKeymap.filter((binding) => binding.key !== "Enter"))),
];
const suggesting = (view) => completionStatus(view.state) === "active";

// One change from `from` text to `to` text, leaving out what they share at
// both ends, so the cursor and scroll stay where they were.
function smallestChange(from, to) {
  let start = 0;
  while (start < from.length && start < to.length && from[start] === to[start]) start++;
  let endFrom = from.length;
  let endTo = to.length;
  while (endFrom > start && endTo > start && from[endFrom - 1] === to[endTo - 1]) {
    endFrom--;
    endTo--;
  }
  return { from: start, to: endFrom, insert: to.slice(start, endTo) };
}

export function createEditor(parent, { onChange, onSelection, onSave, onReview }) {
  const editable = new Compartment();
  const reviewing = new Compartment();
  let escapedFromTab = false;
  // Set while code from a reply is shown as changes to accept or reject.
  let inReview = false;
  let view;

  const changesLeft = (state) => getChunks(state)?.chunks.length ?? 0;
  const endReview = (how) => {
    if (!inReview) return;
    inReview = false;
    view.dispatch({ effects: reviewing.reconfigure([]) });
    onReview({ left: 0, ended: how });
  };
  const acceptAll = () => endReview("accepted");
  const rejectAll = () => {
    if (!inReview) return;
    const original = getOriginalDoc(view.state).toString();
    view.dispatch({ changes: smallestChange(view.state.doc.toString(), original), userEvent: "input.reject" });
    endReview("rejected");
  };

  const keys = Prec.highest(keymap.of([
    { key: "Mod-s", preventDefault: true, run: () => { onSave(); return true; } },
    { key: "Mod-Enter", preventDefault: true, run: () => (inReview ? (acceptAll(), true) : false) },
    { key: "Mod-Backspace", preventDefault: true, run: () => (inReview ? (rejectAll(), true) : false) },
    // Esc, then Tab, moves focus on instead of indenting, so the keyboard is not trapped.
    // Esc that only closes the suggestions does not count.
    { key: "Escape", run: (view) => { if (!suggesting(view)) escapedFromTab = true; return false; } },
    {
      key: "Tab",
      run: (view) => (suggesting(view) ? acceptCompletion(view)
        : escapedFromTab ? (escapedFromTab = false, false) : indentMore(view)),
    },
    { key: "Shift-Tab", run: (view) => (escapedFromTab ? (escapedFromTab = false, false) : indentLess(view)) },
  ]));

  const stateFor = (doc, path, disabled) => EditorState.create({
    doc,
    extensions: [
      keys,
      lineNumbers(), highlightActiveLineGutter(), highlightSpecialChars(), history(), foldGutter(),
      drawSelection(), dropCursor(), EditorState.allowMultipleSelections.of(true),
      indentOnInput(), bracketMatching(), closeBrackets(), rectangularSelection(), crosshairCursor(),
      highlightActiveLine(), highlightSelectionMatches(), suggestions,
      indentUnit.of("    "), EditorState.tabSize.of(4),
      keymap.of([...closeBracketsKeymap, ...defaultKeymap, ...searchKeymap, ...historyKeymap, ...foldKeymap]),
      theme, syntaxHighlighting(colours),
      languageFor(path),
      editable.of([EditorView.editable.of(!disabled), EditorState.readOnly.of(disabled)]),
      reviewing.of([]),
      EditorView.domEventHandlers({ keyup: (event) => { if (event.key !== "Escape") escapedFromTab = false; } }),
      EditorView.updateListener.of((update) => {
        if (update.docChanged) onChange();
        if (update.docChanged || update.selectionSet) onSelection();
        if (inReview && (update.docChanged || update.transactions.some((tr) => tr.effects.length))) {
          // Every change accepted or rejected one by one, or undone: the review is over.
          const left = changesLeft(update.state);
          if (left) onReview({ left, ended: "" });
          else queueMicrotask(() => endReview("accepted"));
        }
      }),
    ],
  });

  view = new EditorView({ state: stateFor("", "", true), parent });
  const leaveReview = () => {
    if (!inReview) return;
    inReview = false;
    onReview({ left: 0, ended: "closed" });
  };

  return {
    // Show a file. A new state means a fresh undo history for it.
    open(text, path) {
      leaveReview();
      view.setState(stateFor(text, path, false));
      view.scrollDOM.scrollTop = 0;
      view.scrollDOM.scrollLeft = 0;
    },
    // Nothing open: empty and read-only.
    close() {
      leaveReview();
      view.setState(stateFor("", "", true));
    },
    // Show new text as changes to the text now in the editor, each with Accept
    // and Reject. Applying again before they are all decided adds to the same
    // review, still compared with the text from before the first one.
    review(text) {
      if (!inReview) {
        view.dispatch({
          effects: reviewing.reconfigure(unifiedMergeView({
            original: view.state.doc,
            mergeControls: (type, action) => {
              const button = document.createElement("button");
              button.type = "button";
              button.name = type;
              button.textContent = type === "accept" ? "Accept" : "Reject";
              button.title = type === "accept" ? "Keep this change" : "Put back what was here";
              button.onmousedown = action;
              return button;
            },
            highlightChanges: false,
            syntaxHighlightDeletions: true,
          })),
        });
      }
      const changes = smallestChange(view.state.doc.toString(), text);
      view.dispatch({ changes, selection: { anchor: changes.from }, userEvent: "input.apply" });
      inReview = true;
      const left = changesLeft(view.state);
      if (!left) {
        endReview("accepted");
        return 0;
      }
      const first = getChunks(view.state).chunks[0];
      view.dispatch({ effects: EditorView.scrollIntoView(first.fromB, { y: "center" }) });
      onReview({ left, ended: "" });
      return left;
    },
    changesLeft: () => (inReview ? changesLeft(view.state) : 0),
    acceptAll,
    rejectAll,
    getValue: () => view.state.doc.toString(),
    getSelection() {
      const { from, to } = view.state.selection.main;
      return { from, to };
    },
    // Put the cursor at a point on the screen, as a click there would.
    cursorAt(x, y) {
      const pos = view.posAtCoords({ x, y });
      if (pos === null) return;
      view.dispatch({ selection: { anchor: pos } });
      view.focus();
    },
    hasFocus: () => view.hasFocus,
    focus: () => view.focus(),
  };
}
