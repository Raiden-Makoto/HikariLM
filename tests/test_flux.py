import json
import os
import struct
import sys
import tempfile
import threading
import time
import unittest
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest import mock

from helper import engine, flux, imagegen, server
from helper.engine import ModelError

HERE = Path(__file__).resolve().parent
KLEIN = "mlx-community/FLUX.2-Klein-4B-4bit"


def weight_names(double: int, single: int) -> list[str]:
    names = ["x_embedder.weight", "context_embedder.weight", "proj_out.weight"]
    names += [f"transformer_blocks.{i}.attn.to_q.weight" for i in range(double)]
    names += [f"single_transformer_blocks.{i}.attn.to_qkv_mlp_proj.weight" for i in range(single)]
    return names


def safetensors_header(names: list[str]) -> bytes:
    header = json.dumps({"__metadata__": {"format": "pt"},
                         **{name: {"dtype": "BF16", "shape": [1], "data_offsets": [0, 2]} for name in names}}).encode()
    return struct.pack("<Q", len(header)) + header + b"\0\0"


def make_model(folder: Path, double: int = 5, single: int = 20, bits: int | None = 4, index: dict | None = None,
               diffusers: bool = False) -> Path:
    """A FLUX.2 folder laid out the way mflux-save writes one (or the way Black Forest Labs
    ships theirs, with diffusers=True), with no real weights."""
    for part in flux.PARTS:
        (folder / part).mkdir(parents=True, exist_ok=True)
        (folder / part / "0.safetensors").write_bytes(safetensors_header(["w"]))
    (folder / "tokenizer").mkdir(exist_ok=True)
    (folder / "tokenizer" / "tokenizer.json").write_text("{}")
    names = weight_names(double, single)
    if diffusers:
        (folder / "transformer" / "0.safetensors").unlink()
        (folder / "transformer" / "diffusion_pytorch_model.safetensors").write_bytes(safetensors_header(names))
    else:
        metadata = {"quantization_level": str(bits)} if bits else {}
        (folder / "transformer" / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": metadata, "weight_map": {name: "0.safetensors" for name in names}}))
    if index is not None:
        (folder / "model_index.json").write_text(json.dumps(index))
    return folder


class FluxCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        root = Path(self._tmp.name)
        self.hub = root / "hub"
        self.home = root / "home"
        env = {"HF_HUB_CACHE": str(self.hub), "HIKARILM_HOME": str(self.home),
               "HIKARILM_MLX_PYTHON": sys.executable, "HIKARILM_MLX_WORKER": str(HERE / "fake_mlx_worker.py"),
               "HIKARILM_PICTURE_PYTHON": sys.executable, "HIKARILM_PICTURE_WORKER": str(HERE / "fake_flux_worker.py")}
        patcher = mock.patch.dict(os.environ, env)
        patcher.start()
        self.addCleanup(patcher.stop)
        flux.forget_scan()
        self.addCleanup(flux.forget_scan)
        self.addCleanup(engine.unload)

    def download(self, repo: str, **shape) -> Path:
        folder = self.hub / ("models--" + repo.replace("/", "--"))
        (folder / "refs").mkdir(parents=True, exist_ok=True)
        (folder / "refs" / "main").write_text("abc123")
        made = make_model(folder / "snapshots" / "abc123", **shape)
        flux.forget_scan()
        return made


class NamesTest(unittest.TestCase):
    def test_every_flux2_klein_is_known_by_its_mflux_name(self):
        for name, wanted in [
            ("black-forest-labs/FLUX.2-klein-4B", ("flux2-klein-4b", None)),
            ("black-forest-labs/FLUX.2-klein-9B", ("flux2-klein-9b", None)),
            ("black-forest-labs/FLUX.2-klein-9b-kv", ("flux2-klein-9b-kv", None)),
            ("black-forest-labs/FLUX.2-klein-base-4B", ("flux2-klein-base-4b", None)),
            ("black-forest-labs/FLUX.2-klein-base-9B", ("flux2-klein-base-9b", None)),
            ("mlx-community/FLUX.2-Klein-4B-4bit", ("flux2-klein-4b", 4)),
            ("mlx-community/flux2-klein-9b-8bit", ("flux2-klein-9b", 8)),
            ("mlx-community/FLUX.2-klein-base-4B-bf16", ("flux2-klein-base-4b", None)),
            ("flux2_klein_4b_3bit", ("flux2-klein-4b", 3)),
            ("mflux-community/flux-2-klein-base-4b-mflux-q4", ("flux2-klein-base-4b", 4)),
            ("mflux-community/flux-2-klein-base-9b-mflux-bf16", ("flux2-klein-base-9b", None)),
            ("mflux-community/flux2-klein-9b-kv-mflux-q8", ("flux2-klein-9b-kv", 8)),
            ("mflux-community/flux2-klein-4b-mflux-q3", ("flux2-klein-4b", 3)),
        ]:
            self.assertEqual(flux.variant_of(name), wanted, name)

    def test_anything_else_has_no_name(self):
        for name in ("mlx-community/flux2-klein-4b-uncensored", "someone/FLUX.2-klein-4B-lora",
                     "mlx-community/flux-3-image-klein-bf16", "black-forest-labs/FLUX.2-dev",
                     "mlx-community/flux2-klein-base-4b-kv", "black-forest-labs/FLUX.1-schnell",
                     "mflux-community/flux-3-image-klein-mflux-q4", "mflux-community/flux-2-klein-4b-mflux-lora"):
            self.assertIsNone(flux.variant_of(name), name)

    def test_every_variant_is_one_mflux_runs(self):
        self.assertEqual(set(flux.VARIANTS), {"flux2-klein-4b", "flux2-klein-9b", "flux2-klein-9b-kv",
                                              "flux2-klein-base-4b", "flux2-klein-base-9b"})

    def test_blocks_are_counted_apart(self):
        self.assertEqual(flux.blocks(weight_names(8, 24)), (8, 24))


class FindTest(FluxCase):
    def test_mflux_saved_copy_is_listed_with_its_bits(self):
        self.download(KLEIN)
        found, notes = flux.scan()
        model = found[KLEIN]
        self.assertEqual((model["variant"], model["label"], model["steps"], model["guidance"]),
                         ("flux2-klein-4b", "FLUX.2 klein", 4, 1.0))
        self.assertEqual(notes, [])

    def test_black_forest_labs_own_files_are_listed(self):
        self.download("black-forest-labs/FLUX.2-klein-base-9B", double=8, single=24, diffusers=True,
                      index={"_class_name": "Flux2KleinPipeline", "is_distilled": False})
        model = flux.find("black-forest-labs/FLUX.2-klein-base-9B")
        self.assertEqual((model["variant"], model["label"], model["steps"], model["guidance"]),
                         ("flux2-klein-base-9b", "FLUX.2 klein base", 50, 4.0))

    def test_mflux_community_copies_are_listed(self):
        self.download("mflux-community/flux-2-klein-base-4b-mflux-q4")
        found, notes = flux.scan()
        model = found["mflux-community/flux-2-klein-base-4b-mflux-q4"]
        self.assertEqual((model["variant"], model["label"], model["steps"], model["guidance"]),
                         ("flux2-klein-base-4b", "FLUX.2 klein base", 50, 4.0))
        self.assertEqual(notes, [])

    def test_fine_tunes_and_other_makers_are_left_out_with_a_note(self):
        self.download("someone/FLUX.2-klein-4B-4bit")
        self.download("mlx-community/flux2-klein-4b-uncensored-4bit")
        found, notes = flux.scan()
        self.assertEqual(found, {})
        self.assertEqual(len(notes), 2)
        self.assertTrue(all("not fine-tunes or LoRA adapters" in note for note in notes))

    def test_a_lora_is_not_a_whole_model(self):
        folder = self.hub / "models--mlx-community--flux2-klein-4b-4bit"
        (folder / "refs").mkdir(parents=True)
        (folder / "refs" / "main").write_text("abc")
        (folder / "snapshots" / "abc").mkdir(parents=True)
        (folder / "snapshots" / "abc" / "adapter.safetensors").write_bytes(safetensors_header(["lora_A"]))
        found, notes = flux.scan()
        self.assertEqual(found, {})
        self.assertIn("not a whole model", notes[0])
        self.assertNotIn("mlx-community", notes[0])

    def test_weights_that_do_not_match_the_name_are_left_out(self):
        self.download("mlx-community/flux2-klein-9b-4bit", double=5, single=20)
        found, notes = flux.scan()
        self.assertEqual(found, {})
        self.assertIn("its weights are not the model its name says", notes[0])

    def test_model_index_must_agree_on_distilled(self):
        self.download("black-forest-labs/FLUX.2-klein-4B", diffusers=True, index={"is_distilled": False})
        self.assertIsNone(flux.find("black-forest-labs/FLUX.2-klein-4B"))

    def test_a_download_still_running_is_left_out(self):
        folder = self.download(KLEIN)
        (folder.parent.parent / "blobs").mkdir()
        (folder.parent.parent / "blobs" / "x.incomplete").write_bytes(b"")
        flux.forget_scan()
        self.assertEqual(flux.scan(), ({}, []))

    def test_own_folder_counts(self):
        make_model(self.home / "models" / "flux2-klein-9b-kv", double=8, single=24, bits=None)
        self.assertEqual(flux.find("flux2-klein-9b-kv")["variant"], "flux2-klein-9b-kv")

    def test_two_copies_of_one_model_keep_their_names(self):
        self.download(KLEIN)
        self.download("mlx-community/flux2-klein-4b-4bit")
        items, _ = imagegen.picture_models()
        self.assertEqual(sorted(item["label"] for item in items), ["FLUX.2-Klein-4B-4bit", "flux2-klein-4b-4bit"])

    def test_the_4b_and_the_9b_say_their_size(self):
        self.download(KLEIN)
        self.download(imagegen.LARGER, double=8, single=24)
        self.download("black-forest-labs/FLUX.2-klein-base-9B", double=8, single=24, bits=None)
        items, _ = imagegen.picture_models()
        self.assertEqual(sorted(item["label"] for item in items),
                         ["FLUX.2 klein 4B", "FLUX.2 klein 9B", "FLUX.2 klein base"])

    def test_picture_models_are_not_chat_models(self):
        self.download(KLEIN)
        self.assertTrue(imagegen.is_picture_model(KLEIN))
        self.assertNotIn(KLEIN, engine.names())


class DrawTest(FluxCase):
    def request(self, prompt="a fox", steps=None):
        return imagegen.clean_request({"model": KLEIN, "prompt": prompt, "seed": 7, "steps": steps})

    def test_draws_with_the_model_defaults_and_reports_each_step(self):
        self.download(KLEIN)
        steps = []
        image = engine.make_picture(flux.find(KLEIN), self.request(), lambda *step: steps.append(step), lambda: False)
        self.assertTrue(image.startswith(imagegen.PNG))
        self.assertTrue(image.endswith(b" flux2-klein-4b 4 1.0"))
        self.assertEqual(steps, [(0, 0), (1, 4), (2, 4), (3, 4), (4, 4)])
        self.assertEqual([item["name"] for item in engine.running()], [KLEIN])
        self.assertEqual(list(self.home.glob(".drawing.*")), [])

    def test_base_models_get_their_steps_and_guidance(self):
        self.download("mlx-community/FLUX.2-klein-base-4B-bf16", bits=None)
        image = engine.make_picture(flux.find("mlx-community/FLUX.2-klein-base-4B-bf16"), self.request(steps=3),
                                    lambda *_: None, lambda: False)
        self.assertTrue(image.endswith(b" flux2-klein-base-4b 3 4.0"))

    def test_stopping_ends_the_worker(self):
        self.download(KLEIN)
        started = time.monotonic()
        image = engine.make_picture(flux.find(KLEIN), self.request("slow"), lambda *_: None,
                                    lambda: time.monotonic() - started > 0.5)
        self.assertIsNone(image)
        self.assertEqual(engine.running(), [])

    def test_an_error_keeps_the_model_and_a_crash_ends_it(self):
        self.download(KLEIN)
        with self.assertRaises(ModelError) as caught:
            engine.make_picture(flux.find(KLEIN), self.request("fail"), lambda *_: None, lambda: False)
        self.assertIn("could not be made", str(caught.exception))
        self.assertEqual(len(engine.running()), 1)
        with self.assertRaises(ModelError) as caught:
            engine.make_picture(flux.find(KLEIN), self.request("crash"), lambda *_: None, lambda: False)
        self.assertIn("picture-worker.log", str(caught.exception))
        self.assertEqual(engine.running(), [])

    def test_mflux_missing_says_how_to_get_it(self):
        self.download(KLEIN)
        with mock.patch.dict(os.environ, {"HIKARILM_PICTURE_PYTHON": "/nowhere/python"}):
            with self.assertRaises(ModelError) as caught:
                engine.make_picture(flux.find(KLEIN), self.request(), lambda *_: None, lambda: False)
        self.assertIn("installs mflux", str(caught.exception))


class GenerateTest(FluxCase):
    def setUp(self):
        super().setUp()
        self.http = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
        threading.Thread(target=self.http.serve_forever, daemon=True).start()
        self.addCleanup(self.http.server_close)
        self.addCleanup(self.http.shutdown)
        self.base = f"http://127.0.0.1:{self.http.server_address[1]}"

    def post(self, payload: dict):
        request = urllib.request.Request(self.base + "/api/generate", json.dumps(payload).encode(),
                                         {"Content-Type": "application/json", "Origin": self.base})
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return [json.loads(line) for line in response.read().decode().splitlines() if line.strip()]
        except urllib.error.HTTPError as error:
            return json.loads(error.read())

    def test_a_picture_is_drawn_and_saved(self):
        self.download(KLEIN)
        lines = self.post({"model": KLEIN, "prompt": "a fox", "seed": 5})
        self.assertEqual(lines[0]["hikarilm"], "loading")
        self.assertIn({"progress": 4, "total": 4}, lines)
        record = lines[-1]["image"]
        self.assertEqual((record["model"], record["seed"]), (KLEIN, 5))
        self.assertTrue(imagegen.read(record["id"]).startswith(imagegen.PNG))
        self.assertEqual(server.router.large, KLEIN)

    def test_a_model_that_is_not_downloaded_says_how_to_get_one(self):
        body = self.post({"model": "mlx-community/flux2-klein-9b-4bit", "prompt": "a fox"})
        self.assertIn("hf download " + imagegen.SUGGESTED, body["error"])
        self.assertIn(imagegen.LARGER + ", the 9B", body["error"])

    def test_the_settings_note_lists_what_was_found(self):
        self.download(KLEIN)
        self.download("someone/flux2-klein-4b-4bit")
        with urllib.request.urlopen(self.base + "/api/pictures", timeout=10) as response:
            body = json.loads(response.read())
        self.assertEqual(body["models"], ["FLUX.2 klein"])
        self.assertEqual(body["ids"], [KLEIN])
        self.assertTrue(body["ready"])
        self.assertEqual(len(body["notes"]), 1)

    def test_the_setup_page_offers_the_4b_and_the_9b(self):
        with urllib.request.urlopen(self.base + "/api/pictures", timeout=10) as response:
            choices = json.loads(response.read())["choices"]
        self.assertEqual([choice["repo"] for choice in choices], [imagegen.SUGGESTED, imagegen.LARGER])
        self.assertIn("less than 18 GB", choices[0]["note"])
        for choice in choices:
            self.assertEqual(choice["command"], "~/.hikarilm/venv/bin/hf download " + choice["repo"])
            variant = flux.variant_of(choice["repo"])
            self.assertIsNotNone(variant, choice["repo"])
            self.assertEqual(variant[0], "flux2-klein-" + choice["name"].split()[-1].lower())


if __name__ == "__main__":
    unittest.main()
