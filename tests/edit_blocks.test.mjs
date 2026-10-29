// Run by test_web.py with node: node tests/edit_blocks.test.mjs
import assert from "node:assert/strict";
import { fileURLToPath } from "node:url";
import path from "node:path";

const here = path.dirname(fileURLToPath(import.meta.url));
const { fileFromReply, fitsOpenFile, isMarker, mergeEdit, parseInfo, resolvePath } =
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
