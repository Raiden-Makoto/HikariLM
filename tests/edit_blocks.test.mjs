// Run by test_web.py with node: node tests/edit_blocks.test.mjs
import assert from "node:assert/strict";
import { fileURLToPath } from "node:url";
import path from "node:path";

const here = path.dirname(fileURLToPath(import.meta.url));
const { applyTarget, askedFiles, fileFromReply, fitsOpenFile, isMarker, mergeEdit, parseInfo, resolvePath } =
  await import(path.join(here, "..", "web", "js", "edit-blocks.js"));

const cases = [];
const test = (name, run) => cases.push([name, run]);
const lines = (...parts) => parts.join("\n") + "\n";

const shop = lines(
  "import math",
  "",
  "",
  "def total(items):",
  "    result = 0",
  "    for item in items:",
  "        result += item.price",
  "    return result",
  "",
  "",
  "def tax(amount):",
  "    return amount * 0.13",
  "",
  "",
  "def label(item):",
  "    return item.name.title()",
);

test("an edit between markers changes only that function", () => {
  const edit = lines(
    "# ... existing code ...",
    "def total(items):",
    "    return sum(item.price for item in items)",
    "",
    "",
    "def tax(amount):",
    "# ... existing code ...",
  );
  assert.equal(mergeEdit(shop, edit), shop.replace(
    "    result = 0\n    for item in items:\n        result += item.price\n    return result\n",
    "    return sum(item.price for item in items)\n"));
});

test("a function shown on its own replaces that function", () => {
  const edit = "def tax(amount):\n    rate = 0.13\n    return amount * 0.13\n";
  assert.equal(mergeEdit(shop, edit), shop.replace("def tax(amount):\n", "def tax(amount):\n    rate = 0.13\n"));
});

test("a function whose last line changed cannot be placed for certain", () => {
  assert.equal(mergeEdit(shop, "def tax(amount):\n    return amount * 0.15\n"), null);
});

test("two pieces go in order, with the code between them kept", () => {
  const edit = lines(
    "import math",
    "import os",
    "",
    "",
    "def total(items):",
    "# ... existing code ...",
    "def label(item):",
    "    return item.name.upper()",
  );
  const merged = mergeEdit(shop, edit);
  assert.equal(merged, null, "the last line is new, so this needs a model");
  const edit2 = lines(
    "import math",
    "import os",
    "",
    "",
    "def total(items):",
    "# ... existing code ...",
    "def label(item):",
    "    print(item)",
    "    return item.name.title()",
  );
  assert.equal(mergeEdit(shop, edit2), shop
    .replace("import math\n", "import math\nimport os\n")
    .replace("def label(item):\n", "def label(item):\n    print(item)\n"));
});

test("a whole new version of the file replaces it", () => {
  const edit = shop.replace("0.13", "0.15").replace("title()", "upper()");
  assert.equal(mergeEdit(shop, edit), edit);
});

test("lines left out between kept lines are deleted", () => {
  const edit = "def total(items):\n    result = 0\n    return result\n";
  assert.equal(mergeEdit(shop, edit), shop.replace("    for item in items:\n        result += item.price\n", ""));
});

test("a piece that only shares a closing brace is not placed", () => {
  const js = "function a() {\n  one();\n}\n\nfunction b() {\n  two();\n}\n";
  assert.equal(mergeEdit(js, "// ... existing code ...\n}\n// ... existing code ...\n"), null);
});

test("the same lines in two places are not guessed between", () => {
  const js = "if (x) {\n  run();\n}\nif (x) {\n  run();\n}\n";
  assert.equal(mergeEdit(js, "if (x) {\n  run();\n  log();\n}\n"), null);
});

test("unrelated code is not placed", () => {
  assert.equal(mergeEdit(shop, "def greet(name):\n    print('hi', name)\n"), null);
});

test("a method shown without its class indentation gets it back", () => {
  const py = "class Cart:\n    def add(self, item):\n        self.items.append(item)\n\n    def clear(self):\n        self.items = []\n";
  const edit = "def add(self, item):\n    print(item)\n    self.items.append(item)\n";
  assert.equal(mergeEdit(py, edit), py.replace("        self.items.append(item)\n",
    "        print(item)\n        self.items.append(item)\n"));
});

test("an empty file takes the code as it is", () => {
  assert.equal(mergeEdit("", "print('hi')"), "print('hi')\n");
  assert.equal(mergeEdit("\n", "# ... existing code ...\nx = 1\n"), null);
});

test("Windows line breaks and a missing last line break are kept", () => {
  const crlf = "a = 1\r\nb = 2\r\nc = 3";
  assert.equal(mergeEdit(crlf, "a = 1\nb = 20\nc = 3\n"), "a = 1\r\nb = 20\r\nc = 3");
});

test("markers in each comment style, and lines that only look like them", () => {
  for (const line of ["// ... existing code ...", "# ...", "  /* ... */", "<!-- ... existing code -->", "-- ...",
    "{/* ... */}", "// rest of the file unchanged", "# existing code here", "% ..."]) {
    assert.ok(isMarker(line), line);
  }
  for (const line of ["...", "    ...", "# Other settings", "// TODO ...", "x = '...'", "...args", "# see the existing docs"]) {
    assert.ok(!isMarker(line), line);
  }
});

test("the fence's first line names the file in each usual way", () => {
  assert.deepEqual(parseInfo("python:src/app.py"), { lang: "python", path: "src/app.py" });
  assert.deepEqual(parseInfo("src/app.py"), { lang: "py", path: "src/app.py" });
  assert.deepEqual(parseInfo("python src/app.py"), { lang: "python", path: "src/app.py" });
  assert.deepEqual(parseInfo('js title="web/main.js"'), { lang: "js", path: "web/main.js" });
  assert.deepEqual(parseInfo("app.py"), { lang: "py", path: "app.py" });
  assert.deepEqual(parseInfo("Dockerfile"), { lang: "dockerfile", path: "Dockerfile" });
  assert.deepEqual(parseInfo("python"), { lang: "python", path: "" });
  assert.deepEqual(parseInfo("bash"), { lang: "bash", path: "" });
  assert.deepEqual(parseInfo(""), { lang: "", path: "" });
  assert.deepEqual(parseInfo("12:30:src/app.py"), { lang: "py", path: "src/app.py", quote: { from: 12, to: 30 } });
});

test("a block with no path fits only a file in its language", () => {
  assert.ok(fitsOpenFile("python", "src/app.py"));
  assert.ok(fitsOpenFile("ts", "web/a.tsx"));
  assert.ok(fitsOpenFile("r", "analysis.R"));
  assert.ok(!fitsOpenFile("python", "web/app.js"));
  assert.ok(!fitsOpenFile("bash", "run.sh"));
  assert.ok(!fitsOpenFile("", "app.py"));
  assert.ok(!fitsOpenFile("text", "notes.txt"));
  assert.ok(!fitsOpenFile("python", ""));
});

test("paths are found in the folder however the model wrote them", () => {
  const files = ["src/app.py", "src/util/money.py", "README.md"];
  assert.deepEqual(resolvePath("src/app.py", files, "shop", "/home/me/shop"), { path: "src/app.py", exists: true });
  assert.deepEqual(resolvePath("./src/app.py", files, "shop", ""), { path: "src/app.py", exists: true });
  assert.deepEqual(resolvePath("shop/src/app.py", files, "shop", ""), { path: "src/app.py", exists: true });
  assert.deepEqual(resolvePath("/home/me/shop/src/app.py", files, "shop", "/home/me/shop"), { path: "src/app.py", exists: true });
  assert.deepEqual(resolvePath("money.py", files, "shop", ""), { path: "src/util/money.py", exists: true });
  assert.deepEqual(resolvePath("readme.md", files, "shop", ""), { path: "README.md", exists: true });
  assert.deepEqual(resolvePath("src/new.py", files, "shop", ""), { path: "src/new.py", exists: false });
  assert.equal(resolvePath("../outside.py", files, "shop", ""), null);
  assert.equal(resolvePath("/etc/passwd", files, "shop", "/home/me/shop"), null);
});

test("the whole file is taken from a model's reply", () => {
  assert.equal(fileFromReply("Here it is:\n```python\nx = 1\n```\n"), "x = 1");
  assert.equal(fileFromReply("```python\nx = 1\ny = 2"), null);
  assert.equal(fileFromReply("x = 1\n"), "x = 1\n");
  assert.equal(fileFromReply("````md\n```py\nx\n```\n````"), "```py\nx\n```");
});

// The same kind of edit in each common language, with its own comment style.
// null: a piece starts or ends with a new line next to "... existing code ...",
// so it could replace the line there or go beside it. A model decides those.
const LANGUAGES = [
  ["JavaScript", "function total(items) {\n  let sum = 0;\n  for (const item of items) sum += item.price;\n  return sum;\n}\n\nfunction tax(amount) {\n  return amount * 0.13;\n}\n\nfunction label(item) {\n  return item.name;\n}\n",
    "// ... existing code ...\nfunction tax(amount) {\n  return amount * 0.15;\n}\n// ... existing code ...\n",
    (f) => f.replace("0.13", "0.15")],
  ["TypeScript", "export interface Item {\n  name: string;\n  price: number;\n}\n\nexport function tax(amount: number): number {\n  return amount * 0.13;\n}\n",
    "// ... existing code ...\nexport function tax(amount: number): number {\n  const rate = 0.13;\n  return amount * rate;\n}\n",
    (f) => f.replace("  return amount * 0.13;\n", "  const rate = 0.13;\n  return amount * rate;\n")],
  ["Go, with tabs", "package shop\n\nfunc Total(items []Item) float64 {\n\tsum := 0.0\n\tfor _, item := range items {\n\t\tsum += item.Price\n\t}\n\treturn sum\n}\n\nfunc Tax(amount float64) float64 {\n\treturn amount * 0.13\n}\n",
    "// ... existing code ...\nfunc Tax(amount float64) float64 {\n\treturn amount * 0.15\n}\n",
    (f) => f.replace("0.13", "0.15")],
  ["Rust", "pub fn total(items: &[Item]) -> f64 {\n    items.iter().map(|i| i.price).sum()\n}\n\npub fn tax(amount: f64) -> f64 {\n    amount * 0.13\n}\n",
    "pub fn tax(amount: f64) -> f64 {\n    let rate = 0.13;\n    amount * rate\n}\n",
    (f) => f.replace("    amount * 0.13\n", "    let rate = 0.13;\n    amount * rate\n")],
  ["C", "#include <stdio.h>\n\nint add(int a, int b) {\n    return a + b;\n}\n\nint main(void) {\n    printf(\"%d\\n\", add(1, 2));\n    return 0;\n}\n",
    "#include <stdio.h>\n#include <stdlib.h>\n\n// ... existing code ...\n",
    null],
  ["Java", "public class Shop {\n    public double tax(double amount) {\n        return amount * 0.13;\n    }\n\n    public String label(Item item) {\n        return item.name;\n    }\n}\n",
    "public double tax(double amount) {\n    double rate = 0.13;\n    return amount * rate;\n}\n",
    (f) => f.replace("        return amount * 0.13;\n", "        double rate = 0.13;\n        return amount * rate;\n")],
  ["HTML", "<!doctype html>\n<html>\n  <head>\n    <title>Shop</title>\n  </head>\n  <body>\n    <h1>Shop</h1>\n    <p>Welcome</p>\n  </body>\n</html>\n",
    "<!-- ... existing code ... -->\n  <body>\n    <h1>Shop</h1>\n    <p>Welcome back</p>\n  </body>\n<!-- ... existing code ... -->\n",
    (f) => f.replace("Welcome", "Welcome back")],
  ["CSS", "body {\n  margin: 0;\n  color: #111;\n}\n\n.button {\n  padding: 4px;\n  color: blue;\n}\n",
    "/* ... existing code ... */\n.button {\n  padding: 8px;\n  color: blue;\n}\n",
    (f) => f.replace("padding: 4px", "padding: 8px")],
  ["SQL", "CREATE TABLE items (\n  id INTEGER PRIMARY KEY,\n  name TEXT NOT NULL\n);\n\nCREATE TABLE orders (\n  id INTEGER PRIMARY KEY,\n  item_id INTEGER\n);\n",
    "-- ... existing code ...\nCREATE TABLE orders (\n  id INTEGER PRIMARY KEY,\n  item_id INTEGER,\n  quantity INTEGER NOT NULL\n);\n",
    (f) => f.replace("  item_id INTEGER\n", "  item_id INTEGER,\n  quantity INTEGER NOT NULL\n")],
  ["R", "library(dplyr)\n\nfit_model <- function(data) {\n  glm(y ~ x, data = data, family = binomial)\n}\n\nsummarise_fit <- function(fit) {\n  summary(fit)\n}\n",
    "# ... existing code ...\nfit_model <- function(data) {\n  glm(y ~ x + z, data = data, family = binomial)\n}\n# ... existing code ...\n",
    (f) => f.replace("y ~ x,", "y ~ x + z,")],
  ["Ruby", "class Shop\n  def tax(amount)\n    amount * 0.13\n  end\n\n  def label(item)\n    item.name\n  end\nend\n",
    "  def tax(amount)\n    amount * 0.15\n  end\n",
    (f) => f.replace("0.13", "0.15")],
  ["YAML", "name: shop\nversion: 1\nservices:\n  web:\n    image: nginx\n    port: 80\n  db:\n    image: postgres\n",
    "# ... existing code ...\n  web:\n    image: nginx\n    port: 8080\n  db:\n# ... existing code ...\n",
    (f) => f.replace("port: 80\n", "port: 8080\n")],
  ["JSON", "{\n  \"name\": \"shop\",\n  \"version\": \"1.0.0\",\n  \"scripts\": {\n    \"start\": \"node app.js\"\n  }\n}\n",
    "{\n  \"name\": \"shop\",\n  \"version\": \"1.1.0\",\n  // ... existing code ...\n}\n",
    null],
  ["Markdown", "# Shop\n\nA small shop.\n\n## Run it\n\nType `npm start`.\n\n## License\n\nMIT\n",
    "## Run it\n\nType `npm install`, then `npm start`.\n\n## License\n",
    (f) => f.replace("Type `npm start`.", "Type `npm install`, then `npm start`.")],
  ["Shell script", "#!/bin/sh\nset -e\n\necho building\nmake\n\necho done\n",
    "#!/bin/sh\nset -e\n\necho building\nmake all\n\necho done\n",
    (f) => f.replace("make\n", "make all\n")],
  ["LaTeX", "\\documentclass{article}\n\\begin{document}\nHello.\n\\end{document}\n",
    "% ... existing code ...\n\\begin{document}\nHello, world.\n\\end{document}\n",
    (f) => f.replace("Hello.", "Hello, world.")],
  ["JSX", "export function App() {\n  return (\n    <div>\n      <h1>Shop</h1>\n      <p>Hi</p>\n    </div>\n  );\n}\n",
    "    <div>\n      {/* ... existing code ... */}\n      <p>Hello</p>\n    </div>\n",
    null],
];
for (const [name, file, edit, expect] of LANGUAGES) {
  test(name + ": the edit lands in the right place", () => {
    const merged = mergeEdit(file, edit);
    if (expect === null) assert.equal(merged, null, "needs a model");
    else assert.equal(merged, expect(file));
  });
}

const project = { files: ["shop.py", "src/app.py", "web/app.js", "README.md"], folderName: "shop", root: "", openPath: "shop.py" };
const general = { text: "how do decorators work in python?" };
const aboutOpen = { text: "why does this fail?", files: [{ name: "shop/shop.py" }] };

test("a general question gets no Apply, whatever the blocks are named", () => {
  assert.equal(applyTarget({ lang: "python", path: "" }, general, project), "");
  assert.equal(applyTarget({ lang: "python", path: "shop.py" }, general, project), "");
  assert.equal(applyTarget({ lang: "python", path: "example.py" }, general, project), "");
  assert.equal(applyTarget({ lang: "python", path: "src/app.py" }, undefined, project), "");
});

test("a question about the open file: its language goes there, named files go to themselves", () => {
  assert.equal(applyTarget({ lang: "python", path: "" }, aboutOpen, project), "shop.py");
  assert.equal(applyTarget({ lang: "javascript", path: "" }, aboutOpen, project), "");
  assert.equal(applyTarget({ lang: "bash", path: "" }, aboutOpen, project), "");
  assert.equal(applyTarget({ lang: "python", path: "src/app.py" }, aboutOpen, project), "src/app.py");
  assert.equal(applyTarget({ lang: "python", path: "example.py" }, aboutOpen, project), "", "a new file nobody asked for");
  assert.equal(applyTarget({ lang: "python", path: "shop.py", quote: { from: 1, to: 3 } }, aboutOpen, project), "");
});

test("lines from Cmd/Ctrl+L and a file named in the text count", () => {
  const lines = { text: "tidy this", files: [{ name: "shop/shop.py lines 3–9" }] };
  assert.deepEqual([...askedFiles(lines, project.files, "shop")], ["shop.py"]);
  const named = { text: "the total in src/app.py is off by one" };
  assert.deepEqual([...askedFiles(named, project.files, "shop")], ["src/app.py"]);
  assert.equal(applyTarget({ lang: "python", path: "src/app.py" }, named, project), "src/app.py");
  assert.equal(applyTarget({ lang: "python", path: "" }, named, project), "", "the open file was not asked about");
  assert.deepEqual([...askedFiles({ text: "myshop.py and shop.pyc" }, project.files, "shop")], []);
});

test("a new file only when the question asks for one or names it", () => {
  assert.equal(applyTarget({ lang: "python", path: "tests/test_shop.py" }, { text: "write tests for the shop" }, project),
    "tests/test_shop.py");
  assert.equal(applyTarget({ lang: "python", path: "utils.py" }, { text: "put the helpers in utils.py" }, project),
    "utils.py");
  assert.equal(applyTarget({ lang: "python", path: "utils.py" }, { text: "what does a utility do?" }, project), "");
  assert.equal(applyTarget({ lang: "", path: "notes/todo.md" }, { text: "create a todo file" }, project), "notes/todo.md");
});

let failed = 0;
for (const [name, run] of cases) {
  try {
    run();
  } catch (error) {
    failed++;
    console.log("FAIL " + name + "\n" + error.message);
  }
}
console.log(cases.length - failed + " of " + cases.length + " passed");
process.exit(failed ? 1 : 0);
