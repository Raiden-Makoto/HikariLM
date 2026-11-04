import threading
import unittest

from helper import keepalive
from helper.keepalive import GRACE, Keeper

R1 = "mlx-community/DeepSeek-R1-0528-Qwen3-8B-4bit"
GEMMA = "mlx-community/gemma-3-12b-it-4bit"


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


class KeeperTest(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.keeper = Keeper(300, clock=self.clock)

    def use(self, model):
        with self.keeper.using(model):
            pass

    def test_after_keepalive(self):
        self.use(GEMMA)
        loaded = [{"name": GEMMA}]
        self.clock.now += 300 + GRACE - 1
        self.assertEqual(self.keeper.overdue(loaded), [])
        self.clock.now += 2
        self.assertEqual(self.keeper.overdue(loaded), [GEMMA])

    def test_leaves_at_follows_the_keepalive_sent(self):
        self.assertIsNone(self.keeper.leaves_at(R1))
        with self.keeper.using(R1, "1m"):
            self.assertIsNone(self.keeper.leaves_at(R1))
        self.assertEqual(self.keeper.leaves_at(R1), self.clock.now + 60)
        self.clock.now += 60 + GRACE + 1
        self.assertEqual(self.keeper.overdue([{"name": R1}]), [R1])

    def test_a_model_never_used_is_left_alone(self):
        self.clock.now += 10 ** 6
        self.assertEqual(self.keeper.overdue([{"name": R1}]), [])

    def test_busy_left_alone(self):
        self.use(R1)
        self.clock.now += 1000
        loaded = [{"name": R1}]
        with self.keeper.using(R1):
            self.assertEqual(self.keeper.overdue(loaded), [])
        self.clock.now += 1000
        self.assertEqual(self.keeper.overdue(loaded), [R1])

    def test_check_unloads_and_forgets(self):
        self.use(GEMMA)
        self.clock.now += 1000
        unloaded = []
        loaded = [{"name": GEMMA}]
        self.assertEqual(self.keeper.check(lambda: loaded, unloaded.append), [GEMMA])
        self.assertEqual(unloaded, [GEMMA])
        self.assertEqual(self.keeper.check(lambda: loaded, unloaded.append), [])

    def test_never_unload_setting(self):
        keeper = Keeper(float("inf"), clock=self.clock)
        with keeper.using(GEMMA):
            pass
        self.clock.now += 10 ** 6
        self.assertEqual(keeper.overdue([{"name": GEMMA}]), [])

    def test_watch_survives_errors_and_stops(self):
        self.use("old")
        self.clock.now += 1000
        calls = []
        stopping = threading.Event()

        def running():
            calls.append(1)
            if len(calls) == 1:
                raise OSError("the worker is busy")
            if len(calls) >= 3:
                stopping.set()
            return [{"name": "old"}]

        unloaded = []
        thread = self.keeper.watch(running, unloaded.append, stopping, every=0.01, on_unload=unloaded.append)
        thread.join(2)
        self.assertFalse(thread.is_alive())
        self.assertIn("old", unloaded)


if __name__ == "__main__":
    unittest.main()
