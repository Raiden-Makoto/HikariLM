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

export function createEditor(parent, { onChange, onSelection, onSave }) {
  const editable = new Compartment();
  let escapedFromTab = false;

  const keys = Prec.highest(keymap.of([
    { key: "Mod-s", preventDefault: true, run: () => { onSave(); return true; } },
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
      EditorView.domEventHandlers({ keyup: (event) => { if (event.key !== "Escape") escapedFromTab = false; } }),
      EditorView.updateListener.of((update) => {
        if (update.docChanged) onChange();
        if (update.docChanged || update.selectionSet) onSelection();
      }),
    ],
  });

  const view = new EditorView({ state: stateFor("", "", true), parent });

  return {
    // Show a file. A new state means a fresh undo history for it.
    open(text, path) {
      view.setState(stateFor(text, path, false));
      view.scrollDOM.scrollTop = 0;
      view.scrollDOM.scrollLeft = 0;
    },
    // Nothing open: empty and read-only.
    close() {
      view.setState(stateFor("", "", true));
    },
    getValue: () => view.state.doc.toString(),
    getSelection() {
      const { from, to } = view.state.selection.main;
      return { from, to };
    },
    // Replace a range as one edit, so Ctrl+Z puts it back. Leaves the new text selected.
    replace(from, to, text) {
      view.dispatch({
        changes: { from, to, insert: text },
        selection: { anchor: from, head: from + text.length },
        scrollIntoView: true,
        userEvent: "input.paste",
      });
    },
    hasFocus: () => view.hasFocus,
    focus: () => view.focus(),
  };
}
