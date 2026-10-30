import threading
import unittest
from datetime import datetime, timezone

from helper import keepalive
from helper.keepalive import GRACE, Keeper


class Clock:
    def __init__(self, now=1_000_000.0):
        self.now = now

    def __call__(self):
        return self.now


class SecondsTest(unittest.TestCase):
    def test_durations(self):
        self.assertEqual(keepalive.seconds("5m"), 300)
        self.assertEqual(keepalive.seconds("1h30m"), 5400)
        self.assertEqual(keepalive.seconds("90s"), 90)
        self.assertEqual(keepalive.seconds("300"), 300)
        self.assertEqual(keepalive.seconds("1m30s"), 90)
        self.assertEqual(keepalive.seconds("500ms"), 0.5)

    def test_never_and_nonsense(self):
        self.assertEqual(keepalive.seconds("-1"), float("inf"))
        self.assertEqual(keepalive.seconds("-1m"), float("inf"))
        self.assertEqual(keepalive.seconds(""), 300)
        self.assertEqual(keepalive.seconds("soon"), 300)


class ExpiryTest(unittest.TestCase):
    def test_nanoseconds_and_offsets(self):
        want = datetime(2026, 10, 27, 12, 0, 0, 123456, tzinfo=timezone.utc).timestamp()
        self.assertAlmostEqual(keepalive.parse_expiry("2026-10-27T12:00:00.123456789Z"), want, places=5)
        self.assertAlmostEqual(keepalive.parse_expiry("2026-10-27T14:00:00.123456789+02:00"), want, places=5)
        self.assertAlmostEqual(keepalive.parse_expiry("2026-10-27T12:00:00.1234Z"), want - 0.000056, places=5)
        self.assertEqual(keepalive.parse_expiry("2026-10-27T12:00:00Z"),
                         datetime(2026, 10, 27, 12, tzinfo=timezone.utc).timestamp())

    def test_nothing_useful(self):
        self.assertIsNone(keepalive.parse_expiry(None))
        self.assertIsNone(keepalive.parse_expiry(""))
        self.assertIsNone(keepalive.parse_expiry("tomorrow"))
        self.assertIsNone(keepalive.parse_expiry("0001-01-01T00:00:00Z"))
        self.assertIsNone(keepalive.parse_expiry("2026-10-27T12:00:00"))

    def test_tags(self):
        self.assertEqual(keepalive.key("gemma3"), "gemma3:latest")
        self.assertEqual(keepalive.key("qwen3:8b"), "qwen3:8b")
        self.assertEqual(keepalive.key("host:5000/me/model"), "host:5000/me/model:latest")


class KeeperTest(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.keeper = Keeper(300, clock=self.clock)

    def use(self, model):
        with self.keeper.using(model):
            pass

    def test_ours_after_keepalive(self):
        self.use("gemma3")
        loaded = [{"name": "gemma3:latest", "expires": None}]
        self.clock.now += 300 + GRACE - 1
        self.assertEqual(self.keeper.overdue(loaded), [])
        self.clock.now += 2
        self.assertEqual(self.keeper.overdue(loaded), ["gemma3:latest"])

    def test_ours_without_expiry(self):
        model = "falcon3:10b"
        self.assertIsNone(self.keeper.leaves_at(model))
        with self.keeper.using(model, "1m"):
            self.assertIsNone(self.keeper.leaves_at(model))
        self.assertEqual(self.keeper.leaves_at(model), self.clock.now + 60)
        self.clock.now += 60 + GRACE + 1
        self.assertEqual(self.keeper.overdue([{"name": model, "expires": None}]), [model])

    def test_ollama_missed_its_own_expiry(self):
        loaded = [{"name": "other:7b", "expires": self.clock.now - GRACE - 1}]
        self.assertEqual(self.keeper.overdue(loaded), ["other:7b"])

    def test_not_ours_and_not_past(self):
        loaded = [{"name": "other:7b", "expires": self.clock.now + 60}, {"name": "bare:1b", "expires": None}]
        self.assertEqual(self.keeper.overdue(loaded), [])

    def test_busy_left_alone(self):
        loaded = [{"name": "qwen3:8b", "expires": self.clock.now - 1000}]
        with self.keeper.using("qwen3:8b"):
            self.assertEqual(self.keeper.overdue(loaded), [])
        self.assertEqual(self.keeper.overdue(loaded), ["qwen3:8b"])

    def test_pinned_left_alone(self):
        self.use("qwen3:8b")
        self.clock.now += 10_000
        loaded = [{"name": "qwen3:8b", "expires": self.clock.now + 10 ** 9}]
        self.assertEqual(self.keeper.overdue(loaded), [])

    def test_check_unloads_and_forgets(self):
        self.use("gemma3")
        self.clock.now += 1000
        unloaded = []
        loaded = [{"name": "gemma3:latest", "expires": None}]
        self.assertEqual(self.keeper.check(lambda: loaded, unloaded.append), ["gemma3:latest"])
        self.assertEqual(unloaded, ["gemma3:latest"])
        self.assertEqual(self.keeper.check(lambda: loaded, unloaded.append), [])

    def test_never_unload_setting(self):
        keeper = Keeper(float("inf"), clock=self.clock)
        with keeper.using("gemma3"):
            pass
        self.clock.now += 10 ** 6
        self.assertEqual(keeper.overdue([{"name": "gemma3:latest", "expires": None}]), [])

    def test_watch_survives_errors_and_stops(self):
        calls = []
        stopping = threading.Event()

        def running():
            calls.append(1)
            if len(calls) == 1:
                raise OSError("Ollama is not running")
            if len(calls) >= 3:
                stopping.set()
            return [{"name": "old:1b", "expires": self.clock.now - 1000}]

        unloaded = []
        thread = self.keeper.watch(running, unloaded.append, stopping, every=0.01, on_unload=unloaded.append)
        thread.join(2)
        self.assertFalse(thread.is_alive())
        self.assertIn("old:1b", unloaded)


if __name__ == "__main__":
    unittest.main()
