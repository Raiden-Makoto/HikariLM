// Inline math in replies, with the page's own math rules and the real marked: node tests/math.test.mjs
import { readFileSync } from "fs";

const { Marked } = await import(new URL("../web/vendor/marked.esm.js", import.meta.url));
const src = readFileSync(new URL("../web/js/markdown.js", import.meta.url), "utf8");
const pick = (name) => {
  const at = src.indexOf("{", src.indexOf("const " + name + " = {"));
  let depth = 0;
  for (let j = at; j < src.length; j++) {
    if (src[j] === "{") depth++;
    if (src[j] === "}" && --depth === 0) return src.slice(at, j + 1);
  }
};
const mathHtml = (tex, display) => "[" + (display ? "D:" : "I:") + tex + "]";
const blockMath = eval("(" + pick("blockMath") + ")");
const inlineMath = eval("(" + pick("inlineMath") + ")");
const md = new Marked({ gfm: true });
md.use({ extensions: [blockMath, inlineMath] });

const math = [
  // Qwen3 (issue 11)
  ["The graph of $ y = x^2 $ from $ x = 0 $ to $ x = 4 $ is", "[I:y = x^2]", "[I:x = 0]", "[I:x = 4]"],
  ["At $ x = 0 $, $ y = 0 $ (vertex).", "[I:x = 0]", "[I:y = 0]"],
  ["steeper as $ x $ increases", "[I:x]"],
  ["- At $ x = 3 $, $ y = 9 $.", "[I:x = 3]", "[I:y = 9]"],
  ["$ 2x + 3 $", "[I:2x + 3]"],
  ["$ \\frac{1}{2} $", "[I:\\frac{1}{2}]"],
  ["Let $x$ be 2.", "[I:x]"],
  ["**Answer:** $x = 3$", "[I:x = 3]"],
  ["Let \\(x\\) be", "[I:x]"],
  ["$x$ and $$y$$", "[I:x]", "[D:y]"],
  ["| a | $x$ |\n|---|---|\n| 1 | $ y $ |", "[I:x]", "[I:y]"],
];
const money = ["costs $5 and $10", "pay $ 5 to get $ 10 back", "from $ 5 to $ 10", "it is $ 20 dollars or $ 30"];

let passed = 0;
const failed = [];
for (const [text, ...want] of math) {
  const out = md.parse(text);
  if (want.every((w) => out.includes(w))) passed++;
  else failed.push(JSON.stringify(text) + " => " + out.trim());
}
for (const text of money) {
  const out = md.parse(text);
  if (!out.includes("[I:")) passed++;
  else failed.push("money taken as math: " + JSON.stringify(text) + " => " + out.trim());
}
for (const line of failed) console.log("FAIL " + line);
console.log(passed + " of " + (math.length + money.length) + " passed");
process.exit(failed.length ? 1 : 0);
