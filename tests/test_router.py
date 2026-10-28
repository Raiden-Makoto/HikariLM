import unittest

from helper.router import ModelRouter, chat_models


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

    def test_nomic_does_not_evict_the_large_model(self) -> None:
        unloaded: list[str] = []
        router = ModelRouter(unloaded.append)
        router.begin("gemma3")
        router.commit("gemma3")
        router.begin("nomic-embed-text:latest")
        router.commit("nomic-embed-text:latest")
        self.assertEqual(unloaded, [])
        self.assertEqual(router.large, "gemma3")

    def test_next_large_model_still_replaces_the_previous_one(self) -> None:
        unloaded: list[str] = []
        router = ModelRouter(unloaded.append)
        router.begin("gemma3")
        router.commit("gemma3")
        router.begin("nomic-embed-text")
        router.begin("qwencoder2.5-7b")
        router.commit("qwencoder2.5-7b")
        self.assertEqual(unloaded, ["gemma3"])
        self.assertEqual(router.large, "qwencoder2.5-7b")

    def test_chat_models_leave_out_nomic(self) -> None:
        names = ["gemma3:latest", "nomic-embed-text:latest", "deepseekr1:8b", "nomic-embed-text"]
        self.assertEqual(chat_models(names), ["gemma3:latest", "deepseekr1:8b"])

    def test_failed_load_does_not_count_as_loaded(self) -> None:
        router = ModelRouter(lambda model: None)
        router.begin("deepseekr1:8b")
        self.assertIsNone(router.large)


if __name__ == "__main__":
    unittest.main()
