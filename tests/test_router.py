import unittest

from helper.router import ModelRouter


class ModelRouterTest(unittest.TestCase):
    def test_follow_up_keeps_the_loaded_model(self) -> None:
        unloaded: list[str] = []
        router = ModelRouter(unloaded.append)
        router.begin("deepseekr1:8b")
        router.commit("deepseekr1:8b")
        router.begin("deepseekr1:8b")
        router.commit("deepseekr1:8b")
        self.assertEqual(unloaded, [])
        self.assertEqual(router.large, "deepseekr1:8b")

    def test_switch_unloads_the_previous_large_model(self) -> None:
        unloaded: list[str] = []
        router = ModelRouter(unloaded.append)
        router.begin("deepseekr1:8b")
        router.commit("deepseekr1:8b")
        router.begin("qwencoder2.5-7b")
        self.assertEqual(unloaded, ["deepseekr1:8b"])
        self.assertIsNone(router.large)
        router.commit("qwencoder2.5-7b")
        self.assertEqual(router.large, "qwencoder2.5-7b")

    def test_forget_after_the_keepalive_unloaded_it(self) -> None:
        unloaded: list[str] = []
        router = ModelRouter(unloaded.append)
        router.begin("gemma3")
        router.commit("gemma3")
        router.forget("gemma3")
        router.begin("qwencoder2.5-7b")
        self.assertEqual(unloaded, [])

    def test_failed_load_does_not_count_as_loaded(self) -> None:
        router = ModelRouter(lambda model: None)
        router.begin("deepseekr1:8b")
        self.assertIsNone(router.large)


if __name__ == "__main__":
    unittest.main()
