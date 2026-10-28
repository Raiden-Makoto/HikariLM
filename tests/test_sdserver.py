"""The helper's own sd-server: started for a picture, reused, and stopped. Runs a fake
sd-server under this Python, so it works on any computer."""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from helper import sdserver

FAKE = str(Path(__file__).with_name("fake_sdserver.py"))


def model(name, *extra):
    return {"name": name, "args": [FAKE, "--model", name, *extra]}


class ManagedTest(unittest.TestCase):
    def setUp(self):
        self.home = tempfile.TemporaryDirectory()
        patch = mock.patch.dict(os.environ, {"LIGHTUSAGI_HOME": self.home.name})
        patch.start()
        self.addCleanup(patch.stop)
        self.managed = sdserver.Managed("5m")
        self.addCleanup(self.managed.stop)
        self.addCleanup(self.home.cleanup)

    def test_start_draw_reuse_and_stop(self):
        slow = model("one", "--steps-delay", "0.2")
        base = self.managed.ensure(sys.executable, slow)
        self.assertEqual(sdserver.current_model(base), "one")
        steps = []
        picture = sdserver.txt2img(base, {"prompt": "a fox", "width": 64, "height": 64, "seed": 1, "steps": 8},
                                   lambda step, total: steps.append((step, total)), lambda: False)
        self.assertTrue(picture)
        self.assertTrue(any(total == 8 for _, total in steps))
        self.assertEqual(self.managed.ensure(sys.executable, slow), base)
        process = self.managed._process
        other = self.managed.ensure(sys.executable, model("two"))
        self.assertNotEqual(other, base)
        self.assertIsNotNone(process.poll(), "the first model's server must be gone before the next starts")
        self.managed.stop()
        self.assertFalse(self.managed.running())

    def test_a_model_that_fails_to_load_says_why(self):
        with self.assertRaises(sdserver.PictureError) as caught:
            self.managed.ensure(sys.executable, model("bad", "--fail"))
        self.assertIn("failed to load model", str(caught.exception))
        self.assertFalse(self.managed.running())

    def test_keepalive_stops_it(self):
        self.managed.keep_alive = 0.2
        self.managed.ensure(sys.executable, model("one"))
        self.managed.touch()
        self.managed._timer.join(5)
        self.assertFalse(self.managed.running())


if __name__ == "__main__":
    unittest.main()
