// What reading an answer aloud says, with the page's own math rules and the real marked:
// node tests/spoken.test.mjs
import { readFileSync } from "fs";

const { Marked } = await import(new URL("../web/vendor/marked.esm.js", import.meta.url));
const { answerText, spokenLines, mathWords, sentences, decodeEntities } = await import(new URL("../web/js/spoken.js", import.meta.url));
const src = readFileSync(new URL("../web/js/markdown.js", import.meta.url), "utf8");
const pick = (name) => {
  const at = src.indexOf("{", src.indexOf("const " + name + " = {"));
  let depth = 0;
  for (let j = at; j < src.length; j++) {
    if (src[j] === "{") depth++;
    if (src[j] === "}" && --depth === 0) return src.slice(at, j + 1);
  }
};
const mathHtml = () => "";
const md = new Marked({ gfm: true });
md.use({ extensions: [eval("(" + pick("blockMath") + ")"), eval("(" + pick("inlineMath") + ")")] });
const say = (text, extra) => spokenLines(md.lexer(text), extra);

let passed = 0;
const failed = [];
const same = (what, got, want) => {
  if (JSON.stringify(got) === JSON.stringify(want)) passed++;
  else failed.push(what + "\n    got  " + JSON.stringify(got) + "\n    want " + JSON.stringify(want));
};
const has = (what, got, ...parts) => {
  const text = JSON.stringify(got);
  if (parts.every((part) => text.includes(part))) passed++;
  else failed.push(what + "\n    got " + text + "\n    missing " + parts.filter((part) => !text.includes(part)).join(" | "));
};
const lacks = (what, got, ...parts) => {
  const text = JSON.stringify(got);
  if (!parts.some((part) => text.includes(part))) passed++;
  else failed.push(what + "\n    got " + text + "\n    should not have " + parts.filter((part) => text.includes(part)).join(" | "));
};

// Math, in words.
for (const [tex, words] of [
  ["x^2 + y^2 = r^2", "x squared plus y squared equals r squared"],
  ["\\frac{1}{2}", "1 over 2"],
  ["\\frac{a+b}{c}", "the fraction a plus b, over c"],
  ["\\sqrt{2}", "the square root of 2"],
  ["\\sqrt[3]{x}", "the cube root of x"],
  ["\\sum_{i=1}^{n} i", "the sum from i equals 1 to n of i"],
  ["\\int_0^1 x\\,dx", "the integral from 0 to 1 of x d x"],
  ["\\lim_{x \\to 0} f(x)", "the limit as x goes to 0 of f of x"],
  ["f(x) = 3x - 2", "f of x equals 3 x minus 2"],
  ["-3", "negative 3"],
  ["x = -y", "x equals negative y"],
  ["x_1, x_2", "x 1, x 2"],
  ["x_{i}", "x sub i"],
  ["\\alpha \\leq \\beta", "alpha is less than or equal to beta"],
  ["\\sin^2 x + \\cos^2 x = 1", "sine squared of x plus cosine squared of x equals 1"],
  ["A^{-1}", "A inverse"],
  ["A^T", "A transpose"],
  ["e^{i\\pi} + 1 = 0", "e to the power of i pi, plus 1 equals 0"],
  ["(x+1)^2", "x plus 1 all squared"],
  ["n(n+1)", "n times n plus 1"],
  ["f'(x)", "f prime of x"],
  ["\\text{if } x > 0", "if x is greater than 0"],
  ["\\begin{pmatrix} 1 & 2 \\\\ 3 & 4 \\end{pmatrix}", "a matrix"],
  ["|x| = \\begin{cases} x & \\text{if } x \\ge 0 \\\\ -x & \\text{otherwise} \\end{cases}",
    "x equals x if x is greater than or equal to 0, negative x otherwise"],
  ["\\binom{n}{k}", "n choose k"],
  ["90^\\circ", "90 degrees"],
  ["50\\%", "50 percent"],
  ["\\hat{y}", "y hat"],
  ["\\log_2 8 = 3", "log base 2 of 8 equals 3"],
  ["\\Delta x", "capital delta x"],
  ["n!", "n factorial"],
  ["x \\in \\mathbb{R}", "x is in R"],
]) same("math " + tex, mathWords(tex), words);
same("unknown command, inline", mathWords("\\weird{x}"), "a formula");
same("unknown command, block", mathWords("\\weird{x}", true), "There is an equation here");
same("too long", mathWords("x + ".repeat(60) + "1", true), "There is an equation here");
same("unclosed brace", mathWords("\\frac{1}{2"), "a formula");
same("empty", mathWords(""), "");

// Qwen3's answer from issue 11, with a plot from the Python tool.
const answer = `The graph of $ y = x^2 $ from $ x = 0 $ to $ x = 4 $ is a parabola that opens upward. Here's what it looks like:

**Shape:** A smooth curve symmetric about the y-axis. 📈
- At $ x = 0 $, $ y = 0 $ (vertex at the origin).
- At $ x = 4 $, $ y = 16 $.

\`\`\`python
import matplotlib.pyplot as plt
plt.plot(x, x**2)
\`\`\`

$$\\int_0^4 x^2\\,dx = \\frac{64}{3}$$

See [the docs](https://matplotlib.org) or https://www.example.com/a/b. Done &amp; dusted! 🎉`;
const lines = say(answer, { plots: 1 });
same("the plot is named first", lines[0], "There is a plot above.");
has("inline math in words", lines, "The graph of y equals x squared from x equals 0 to x equals 4 is a parabola that opens upward.");
has("list items are sentences", lines, "At x equals 0, y equals 0 (vertex at the origin).", "At x equals 4, y equals 16.");
has("code is named, not read", lines, "There is a Python code block here, 2 lines long.");
lacks("no code read out", lines, "import", "matplotlib.pyplot", "plt.plot");
has("block math in words", lines, "the integral from 0 to 4 of x squared d x equals 64 over 3.");
has("links by their text, bare ones by their site", lines, "See the docs or a link to example.com.");
lacks("no emoji, markup or entities", lines, "📈", "🎉", "**", "$", "&amp;", "https://");
has("entities said", lines, "Done and dusted!");
same("two plots", say("Here.", { plots: 2 })[0], "There are 2 plots above.");

// Tables: small ones row by row, big ones named.
same("small table", say("| Name | Age |\n|---|---|\n| Ann | 30 |\n| Bo | 4 |"),
  ["A table with 2 columns: Name, Age.", "Name, Ann; Age, 30.", "Name, Bo; Age, 4."]);
const big = "| a | b | c | d | e |\n|---|---|---|---|---|\n" + "| 1 | 2 | 3 | 4 | 5 |\n".repeat(8);
same("big table", say(big), ["There is a table here with 8 rows and 5 columns: a, b, c, d, e."]);

// Pictures, links, code, quotes, html, rules.
same("picture with alt text", say("![a plot of sales](x.png)"), ["a picture: a plot of sales."]);
same("money stays money", say("It costs $5 and $10."), ["It costs $5 and $10."]);
same("lists", say("1. First\n2. Second"), ["First.", "Second."]);
same("quote", say("> quoted\n\nWhy?"), ["quoted", "Why?"]);
same("long inline code", say("Use `a_very_long_function_name_that_goes_on_and_on()` now."), ["Use some code now."]);
same("short inline code", say("Run `pip install numpy` first."), ["Run pip install numpy first."]);
same("symbols", say("x < 3 and y > 2, 30°C, 50% done -> ok"),
  ["x is less than 3 and y is greater than 2, 30 degrees C, 50 percent done to ok"]);
same("html is stripped", say("<details><summary>More</summary>hidden text</details>"), ["More hidden text"]);
same("a rule says nothing", say("---"), []);
same("headings end a sentence", say("# Title\nText"), ["Title.", "Text"]);
same("code with no language", say("```\nls\n```"), ["There is a code block here, 1 line long."]);
same("a diagram", say("```mermaid\ngraph TD\n```"), ["There is a diagram here."]);
same("only emoji", say("👍"), []);
same("nothing", say(""), []);
same("thinking never read", say(answerText("<think>secret</think>Answer.")), ["Answer."]);
same("stray thinking never read", say(answerText("reasoning here</think>\n\nAnswer.")), ["Answer."]);
same("task list", say("- [x] done thing\n- [ ] open thing"), ["done thing.", "open thing."]);
same("footnote marks", say("True [1]."), ["True."]);

// Sentences: split for streaming; a long one at a comma.
same("sentences", sentences("One. Two! Three? e.g. this"), ["One.", "Two!", "Three? e.g. this"]);
const long = sentences(("word ".repeat(30) + ", ").repeat(4));
has("a long sentence is cut", long.length > 1 ? "cut" : "whole", "cut");
same("each piece is short", long.every((line) => line.length <= 300), true);
same("entities", decodeEntities("&lt;a&gt; &#39;b&#39; &#x41;"), "<a> 'b' A");

for (const line of failed) console.log("FAIL " + line);
console.log(passed + " of " + (passed + failed.length) + " passed");
process.exit(failed.length ? 1 : 0);
