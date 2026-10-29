"""The page is split into web/js modules and web/css files with no build step.
These checks stand in for a JavaScript linter, so a name missing from an
import is found here rather than when someone presses the button that needs it."""

import re
import shutil
import subprocess
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

from helper import server

WEB = server.WEB_ROOT
JS = WEB / "js"
DECLARED = re.compile(r"^(?:export )?(?:const|let|var|async function|function|class) +([A-Za-z_$][\w$]*)", re.M)
EXPORTED = re.compile(r"^export (?:const|let|var|async function|function|class) +([A-Za-z_$][\w$]*)", re.M)
IMPORT = re.compile(r'^import \{([^}]*)\} from "\./([\w-]+)\.js";\n', re.M)
# A name not after a property dot; a spread (...name) is a real use.
USE = r"(?<![\w$])(?:(?<=\.\.\.)|(?<!\.)){}(?![\w$])"
BEFORE_REGEX = {"return", "typeof", "in", "of", "new", "delete", "void", "throw", "case", "else", "await", "yield"}


def code_only(src: str) -> str:
    """The source with comments, strings and regular expressions blanked out,
    keeping the ${...} parts of template strings."""
    out, i, n, last = [], 0, len(src), ""

    def regex_can_start():
        return not last or last in BEFORE_REGEX or last[-1] in "(,=:[!&|?{};+-*%<>~^"

    def code(i, closing=False):
        nonlocal last
        depth = 0
        while i < n:
            c = src[i]
            if closing and c == "}" and depth == 0:
                return i + 1
            depth += c == "{"
            depth -= c == "}"
            if src.startswith("//", i):
                j = src.find("\n", i)
                i = n if j < 0 else j
                continue
            if src.startswith("/*", i):
                end = src.index("*/", i) + 2
                out.append("\n" * src.count("\n", i, end))
                i = end
                continue
            if c in "'\"":
                j = i + 1
                while src[j] != c:
                    j += 2 if src[j] == "\\" else 1
                i, last = j + 1, "string"
                out.append(" 0 ")
                continue
            if c == "`":
                i += 1
                while src[i] != "`":
                    if src[i] == "\\":
                        i += 2
                    elif src.startswith("${", i):
                        out.append(" ")
                        i = code(i + 2, closing=True)
                        out.append(" ")
                    else:
                        i += 1
                i, last = i + 1, "string"
                out.append(" 0 ")
                continue
            if c == "/" and regex_can_start():
                j, in_class = i + 1, False
                while src[j] != "/" or in_class:
                    if src[j] == "\\":
                        j += 1
                    elif src[j] == "[":
                        in_class = True
                    elif src[j] == "]":
                        in_class = False
                    j += 1
                j += 1
                while j < n and src[j].isalpha():
                    j += 1
                i, last = j, "regex"
                out.append(" 0 ")
                continue
            word = re.match(r"[A-Za-z_$][\w$]*|\d[\w.]*", src[i:i + 100])
            if word:
                out.append(word.group())
                i, last = i + len(word.group()), word.group()
                continue
            out.append(c)
            if not c.isspace():
                last = c
            i += 1
        return i

    code(0)
    return "".join(out)


def modules():
    return {path.stem: path.read_text() for path in sorted(JS.glob("*.js"))}


def without_imports(text: str) -> str:
    """The code after the import lines, with the lines still numbered as in the file."""
    return code_only(IMPORT.sub(lambda found: "\n" * found.group().count("\n"), text))


class PageFilesTest(unittest.TestCase):
    def test_everything_the_page_loads_is_there(self):
        page = (WEB / "index.html").read_text()
        wanted = re.findall(r'(?:src|href)="(/(?:js|css)/[^"]+)"', page)
        self.assertIn("/js/main.js", wanted)
        for address in wanted:
            self.assertTrue(server.PAGE_PART.fullmatch(address), address)
            self.assertTrue((WEB / address.lstrip("/")).is_file(), address)
        self.assertNotIn("<style>", page)
        self.assertEqual(page.count("<script"), 2, "the theme line and main.js only")

    def test_the_helper_serves_the_parts_and_nothing_beside_them(self):
        http = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
        threading.Thread(target=http.serve_forever, daemon=True).start()
        self.addCleanup(http.server_close)
        self.addCleanup(http.shutdown)
        base = f"http://127.0.0.1:{http.server_address[1]}"
        with urllib.request.urlopen(base + "/js/main.js") as response:
            self.assertEqual(response.headers["Content-Type"], "text/javascript; charset=utf-8")
        with urllib.request.urlopen(base + "/css/base.css") as response:
            self.assertEqual(response.headers["Content-Type"], "text/css; charset=utf-8")
        for address in ("/js/missing.js", "/js/../../helper/server.py", "/js/Main.JS", "/css/x.js.map"):
            with self.subTest(address), self.assertRaises(urllib.error.HTTPError) as caught:
                urllib.request.urlopen(base + address)
            self.assertEqual(caught.exception.code, 404)


class ModuleImportsTest(unittest.TestCase):
    def setUp(self):
        self.files = modules()
        self.exports = {}
        for name, text in self.files.items():
            for exported in EXPORTED.findall(text):
                self.exports.setdefault(exported, []).append(name)

    def test_the_scanner_keeps_code_and_drops_text(self):
        cleaned = code_only('a("b c", /d[/]e/g, `f ${g + "h"} i`) / j; // k\n[...l]')
        for kept in ("a", "g", "j", "l"):
            self.assertRegex(cleaned, USE.format(kept))
        for dropped in ("b", "c", "d", "e", "f", "h", "i", "k"):
            self.assertNotRegex(cleaned, USE.format(dropped))

    def test_imports_name_real_exports_and_are_used(self):
        for name, text in self.files.items():
            body = without_imports(text)
            for names, source in IMPORT.findall(text):
                self.assertIn(source, self.files, f"{name}.js imports from a missing {source}.js")
                for part in filter(None, (p.strip() for p in names.split(","))):
                    real, _, local = part.partition(" as ")
                    with self.subTest(module=name, name=real):
                        self.assertIn(source, self.exports.get(real.strip(), []), f"{source}.js does not export it")
                        self.assertRegex(body, USE.format(re.escape((local or real).strip())), "imported, not used")

    def test_names_from_other_parts_are_imported(self):
        for name, text in self.files.items():
            imported = {part.strip().split(" as ")[-1].strip()
                        for names, _ in IMPORT.findall(text) for part in names.split(",") if part.strip()}
            own = set(DECLARED.findall(text))
            body = without_imports(text)
            for exported, owners in self.exports.items():
                if exported in imported or exported in own or exported == "start":
                    continue
                found = re.search(USE.format(re.escape(exported)), body)
                if found:
                    line = body.count("\n", 0, found.start()) + 1
                    self.fail(f"{name}.js line {line} uses {exported} without importing it from {owners[0]}.js")

    def test_each_name_is_exported_once(self):
        twice = {n: o for n, o in self.exports.items() if len(o) > 1 and n != "start"}
        self.assertEqual(twice, {})

    def test_shared_variables_are_set_only_on_state(self):
        # Another module's let cannot be set from outside; shared ones live on state.
        lets = {}
        for name, text in self.files.items():
            for declared in re.findall(r"^export let ([A-Za-z_$][\w$]*)", text, re.M):
                lets[declared] = name
        for name, text in self.files.items():
            body = without_imports(text)
            for variable, owner in lets.items():
                if owner == name:
                    continue
                with self.subTest(module=name, variable=variable):
                    self.assertNotRegex(body, USE.format(re.escape(variable)) + r"\s*(=(?![=>])|[-+*/%|&?]=|\+\+|--)")


class EditBlocksTest(unittest.TestCase):
    def test_apply_reads_and_merges_code_blocks_like_cursor(self):
        node = shutil.which("node")
        if not node:
            self.skipTest("node is not installed")
        run = subprocess.run([node, str(Path(__file__).with_name("edit_blocks.test.mjs"))],
                             capture_output=True, text=True, timeout=60)
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)


if __name__ == "__main__":
    unittest.main()
