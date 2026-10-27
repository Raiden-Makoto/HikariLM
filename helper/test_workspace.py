import os
import tempfile
import unittest
from pathlib import Path

from helper import workspace


class WorkspaceTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name) / "proj"
        self.root.mkdir()
        (self.root / "a.py").write_text("print(1)\n")
        (self.root / "sub").mkdir()
        (self.root / "sub" / "b.c").write_text("int x;\n")

    def test_listing_skips_unwanted(self):
        (self.root / "boot.asm").write_text("x")
        (self.root / "m.ckpt").write_bytes(b"x")
        (self.root / ".git").mkdir()
        (self.root / ".git" / "c.py").write_text("x")
        (self.root / "node_modules").mkdir()
        (self.root / "node_modules" / "d.js").write_text("x")
        (self.root / "Makefile").write_text("all:\n")
        found = workspace.listing(str(self.root))
        self.assertEqual(found["files"], ["a.py", "Makefile", "sub/b.c"])
        self.assertFalse(found["cut"])

    def test_listing_caps(self):
        for i in range(workspace.MAX_FILES + 5):
            (self.root / f"f{i:03}.txt").write_text("x")
        found = workspace.listing(str(self.root))
        self.assertEqual(len(found["files"]), workspace.MAX_FILES)
        self.assertTrue(found["cut"])

    def test_bad_roots(self):
        for bad in ("", None, str(self.root / "nope"), str(self.root / "a.py")):
            with self.assertRaises(workspace.WorkspaceError):
                workspace.listing(bad)

    def test_read_and_write_roundtrip(self):
        got = workspace.read(str(self.root), "a.py")
        self.assertEqual(got["text"], "print(1)\n")
        saved = workspace.write(str(self.root), "a.py", "print(2)\n", got["version"])
        self.assertTrue(saved["changed"])
        self.assertEqual((self.root / "a.py").read_text(), "print(2)\n")
        self.assertEqual(workspace.read(str(self.root), "a.py")["version"], saved["version"])
        leftovers = [p.name for p in self.root.iterdir() if p.name.endswith(".tmp")]
        self.assertEqual(leftovers, [])

    def test_unchanged_save_does_nothing(self):
        got = workspace.read(str(self.root), "a.py")
        self.assertFalse(workspace.write(str(self.root), "a.py", "print(1)\n", got["version"])["changed"])

    def test_conflict_when_changed_on_disk(self):
        got = workspace.read(str(self.root), "a.py")
        (self.root / "a.py").write_text("other\n")
        with self.assertRaises(workspace.Conflict):
            workspace.write(str(self.root), "a.py", "mine\n", got["version"])
        self.assertEqual((self.root / "a.py").read_text(), "other\n")

    def test_refuses_escape(self):
        (Path(self._tmp.name) / "secret.txt").write_text("s")
        for rel in ("../secret.txt", "/etc/hosts", "sub/../../secret.txt", "..\\secret.txt"):
            with self.assertRaises(workspace.WorkspaceError, msg=rel):
                workspace.read(str(self.root), rel)

    def test_refuses_symlink_out(self):
        outside = Path(self._tmp.name) / "secret.txt"
        outside.write_text("s")
        os.symlink(outside, self.root / "link.txt")
        with self.assertRaises(workspace.WorkspaceError):
            workspace.read(str(self.root), "link.txt")
        self.assertNotIn("link.txt", workspace.listing(str(self.root))["files"])

    def test_never_creates_or_opens_other_kinds(self):
        with self.assertRaises(workspace.WorkspaceError):
            workspace.write(str(self.root), "new.py", "x", "v")
        (self.root / "boot.asm").write_text("x")
        with self.assertRaises(workspace.WorkspaceError):
            workspace.read(str(self.root), "boot.asm")
        self.assertFalse((self.root / "new.py").exists())

    def test_refuses_binary_and_large(self):
        (self.root / "bin.txt").write_bytes(b"ab\x00cd")
        with self.assertRaises(workspace.WorkspaceError):
            workspace.read(str(self.root), "bin.txt")
        (self.root / "big.txt").write_bytes(b"x" * (workspace.MAX_FILE + 1))
        with self.assertRaises(workspace.WorkspaceError):
            workspace.read(str(self.root), "big.txt")

    def test_crlf_and_encodings_survive(self):
        (self.root / "w.txt").write_bytes(b"one\r\ntwo\r\n")
        got = workspace.read(str(self.root), "w.txt")
        self.assertEqual(got["text"], "one\ntwo\n")
        workspace.write(str(self.root), "w.txt", "one\nTWO\n", got["version"])
        self.assertEqual((self.root / "w.txt").read_bytes(), b"one\r\nTWO\r\n")

        (self.root / "bom.txt").write_bytes(b"\xef\xbb\xbfhi\n")
        got = workspace.read(str(self.root), "bom.txt")
        workspace.write(str(self.root), "bom.txt", "hi!\n", got["version"])
        self.assertEqual((self.root / "bom.txt").read_bytes(), b"\xef\xbb\xbfhi!\n")

        (self.root / "u16.txt").write_bytes("\ufeffhé\n".encode("utf-16-le"))
        got = workspace.read(str(self.root), "u16.txt")
        self.assertEqual(got["text"], "hé\n")
        workspace.write(str(self.root), "u16.txt", "hé!\n", got["version"])
        self.assertEqual((self.root / "u16.txt").read_bytes(), b"\xff\xfe" + "hé!\n".encode("utf-16-le"))

        (self.root / "win.txt").write_bytes("caf\xe9\n".encode("cp1252"))
        got = workspace.read(str(self.root), "win.txt")
        self.assertEqual(got["text"], "café\n")
        workspace.write(str(self.root), "win.txt", "café!\n", got["version"])
        self.assertEqual((self.root / "win.txt").read_bytes(), "café!\n".encode("cp1252"))
        with self.assertRaises(workspace.WorkspaceError):
            workspace.write(str(self.root), "win.txt", "日本\n", workspace.read(str(self.root), "win.txt")["version"])

    def test_permissions_kept(self):
        os.chmod(self.root / "a.py", 0o755)
        got = workspace.read(str(self.root), "a.py")
        workspace.write(str(self.root), "a.py", "x\n", got["version"])
        self.assertEqual((self.root / "a.py").stat().st_mode & 0o777, 0o755)


if __name__ == "__main__":
    unittest.main()
