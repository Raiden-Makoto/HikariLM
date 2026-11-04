import json
import struct
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from unittest import mock

from helper import engine, flux, imagegen, lora, server
from tests.test_flux import KLEIN, FluxCase

BASE = "mlx-community/flux-2-klein-base-4b-mflux-q4"
PIXEL = "adirik/pixel-art-lora-flux.2-klein-4B"


def lora_header(double: int = 5, single: int = 20, width: int = 3072, meta: dict | None = None,
                extra: list[str] | None = None) -> dict:
    """A LoRA's safetensors header the way ai-toolkit saves one."""
    header = {"__metadata__": meta or {"format": "pt"}}
    for i in range(double):
        header[f"diffusion_model.double_blocks.{i}.img_attn.qkv.lora_A.weight"] = {"dtype": "BF16", "shape": [16, width]}
        header[f"diffusion_model.double_blocks.{i}.img_attn.qkv.lora_B.weight"] = {"dtype": "BF16", "shape": [3 * width, 16]}
    for i in range(single):
        header[f"diffusion_model.single_blocks.{i}.linear1.lora_A.weight"] = {"dtype": "BF16", "shape": [16, width]}
    for key in extra or []:
        header[key] = {"dtype": "BF16", "shape": [16]}
    return header


def file_bytes(header: dict) -> bytes:
    raw = json.dumps(header).encode()
    return struct.pack("<Q", len(raw)) + raw + b"\0\0"


class FakeHub:
    """Answers lora._curl the way Hugging Face does, for one repo."""

    def __init__(self, repo: str, files: dict[str, dict], card: dict | None = None, code: int = 200):
        self.repo, self.files, self.card, self.code = repo, files, card or {}, code

    def __call__(self, url: str, headers: dict | None = None):
        if url.startswith(lora.API):
            if self.code != 200 or not url.startswith(lora.API + self.repo + "?"):
                return self.code if self.code != 200 else 404, b"{}"
            siblings = [{"rfilename": name, "size": len(file_bytes(header))} for name, header in self.files.items()]
            siblings.append({"rfilename": "README.md", "size": 10})
            return 200, json.dumps({"siblings": siblings, "cardData": self.card}).encode()
        name = url.split("/resolve/main/", 1)[1]
        data = file_bytes(self.files[name])
        start, end = (int(part) for part in headers["Range"].removeprefix("bytes=").split("-"))
        return 206, data[start:end + 1]


class LookUpTest(unittest.TestCase):
    def test_a_repo_can_be_written_three_ways(self):
        self.assertEqual(lora.parse("adirik/pixel-art"), ("adirik/pixel-art", ""))
        self.assertEqual(lora.parse(" adirik/pixel-art:sub/a.safetensors "), ("adirik/pixel-art", "sub/a.safetensors"))
        self.assertEqual(lora.parse("https://huggingface.co/adirik/pixel-art/blob/main/a.safetensors"),
                         ("adirik/pixel-art", "a.safetensors"))
        self.assertEqual(lora.parse("https://huggingface.co/adirik/pixel-art/"), ("adirik/pixel-art", ""))
        for bad in ["", "pixel-art", "a/b/c", "../x/y", "a/b:notes.txt", "a/b:../x.safetensors"]:
            with self.assertRaises(lora.LoraError, msg=bad):
                lora.parse(bad)

    def test_what_the_card_says_it_was_made_for(self):
        self.assertEqual(lora.made_for(["black-forest-labs/FLUX.2-klein-base-4B"]), ("", ""))
        refusal, note = lora.made_for(["black-forest-labs/FLUX.2-klein-4B"])
        self.assertEqual(refusal, "")
        self.assertIn("quick FLUX.2 klein 4B", note)
        self.assertIn("9B", lora.made_for(["black-forest-labs/FLUX.2-klein-base-9B"])[0])
        self.assertIn("9B", lora.made_for(["black-forest-labs/FLUX.2-klein-9B"])[0])
        self.assertIn("FLUX.1", lora.made_for(["black-forest-labs/FLUX.1-dev"])[0])
        self.assertIn("not FLUX.2 klein", lora.made_for(["stabilityai/sdxl"])[0])
        self.assertEqual(lora.made_for([])[0], "")
        self.assertIn("does not say", lora.made_for([])[1])

    def test_the_file_shows_what_it_was_made_for(self):
        self.assertEqual(lora.check_header(lora_header()), "")
        self.assertEqual(lora.check_header(lora_header(double=2, single=0)), "")
        diffusers = {"transformer.single_transformer_blocks.19.attn.to_qkv_mlp_proj.lora_A.weight":
                     {"dtype": "BF16", "shape": [16, 3072]}}
        self.assertEqual(lora.check_header(diffusers), "")
        self.assertIn("DoRA", lora.check_header(lora_header(extra=["diffusion_model.double_blocks.0.img_attn.qkv.lora_magnitude_vector"])))
        self.assertIn("not a LoRA", lora.check_header({"model.weight": {"dtype": "BF16", "shape": [3]}}))
        self.assertIn("9B", lora.check_header(lora_header(width=4096)))
        self.assertIn("9B", lora.check_header(lora_header(meta={"model": "flux2-klein-base-9b"})))
        self.assertIn("larger model", lora.check_header(lora_header(double=19, single=38)))
        self.assertIn("larger model", lora.check_header(lora_header(double=5, single=24)))
        self.assertIn("no layers", lora.check_header({"unet.lora_A.weight": {"dtype": "BF16", "shape": [16, 8]}}))

    def test_the_trigger_word_comes_from_the_card_or_the_trainer(self):
        self.assertEqual(lora.trigger_of({"instance_prompt": " PXART4 "}, {}), "PXART4")
        self.assertEqual(lora.trigger_of({"instance_prompt": ["zq_usagi"]}, {}), "zq_usagi")
        meta = {"ss_tag_frequency": json.dumps({"1_bld5g style": {"bld5g": 30}})}
        self.assertEqual(lora.trigger_of({}, meta), "bld5g style")
        self.assertEqual(lora.trigger_of({}, {"ss_tag_frequency": "not json"}), "")

    def test_the_lora_is_picked_over_its_checkpoints(self):
        files = [{"name": name, "size": 1} for name in
                 ["style_000002750.safetensors", "checkpoint-500/a.safetensors", "style_comfy.safetensors", "style.safetensors"]]
        self.assertEqual(lora._pick(files, ""), "style.safetensors")
        self.assertEqual(lora._pick(files, "style_000002750.safetensors"), "style_000002750.safetensors")
        with self.assertRaises(lora.LoraError):
            lora._pick(files, "other.safetensors")

    def test_a_base_4b_lora_is_looked_up(self):
        hub = FakeHub(PIXEL, {"pixel.safetensors": lora_header(), "pixel_000001000.safetensors": lora_header()},
                      {"base_model": "black-forest-labs/FLUX.2-klein-base-4B", "instance_prompt": "PXART4"})
        with mock.patch.object(lora, "_curl", hub):
            found = lora.look_up("https://huggingface.co/" + PIXEL)
        self.assertNotIn("error", found)
        self.assertEqual((found["file"], found["trigger"], found["note"]), ("pixel.safetensors", "PXART4", ""))
        self.assertEqual(found["id"], PIXEL + ":pixel.safetensors")
        self.assertEqual(len(found["files"]), 2)
        self.assertEqual(found["command"], f"~/.hikarilm/venv/bin/hf download {PIXEL} pixel.safetensors")

    def test_the_trigger_word_can_come_from_the_file(self):
        meta = {"ss_tag_frequency": json.dumps({"10_bld5g style": {}})}
        hub = FakeHub("a/b", {"b.safetensors": lora_header(meta=meta)}, {"base_model": "black-forest-labs/FLUX.2-klein-base-4B"})
        with mock.patch.object(lora, "_curl", hub):
            self.assertEqual(lora.look_up("a/b")["trigger"], "bld5g style")

    def test_a_9b_lora_is_refused_with_why(self):
        hub = FakeHub("a/b", {"b.safetensors": lora_header(width=4096, double=8, single=24)}, {"base_model": []})
        with mock.patch.object(lora, "_curl", hub):
            self.assertIn("9B", lora.look_up("a/b")["error"])
        hub = FakeHub("a/b", {"b.safetensors": lora_header()}, {"base_model": "black-forest-labs/FLUX.2-klein-base-9B"})
        with mock.patch.object(lora, "_curl", hub):
            self.assertIn("9B", lora.look_up("a/b")["error"])

    def test_a_missing_repo_or_one_without_a_lora(self):
        with mock.patch.object(lora, "_curl", FakeHub("a/b", {}, code=404)):
            with self.assertRaises(lora.LoraError) as caught:
                lora.look_up("a/b")
        self.assertIn("no public repo", str(caught.exception))
        with mock.patch.object(lora, "_curl", FakeHub("a/b", {})):
            with self.assertRaises(lora.LoraError) as caught:
                lora.look_up("a/b")
        self.assertIn("no .safetensors", str(caught.exception))

    def test_the_trigger_word_goes_in_front_once(self):
        self.assertEqual(lora.with_trigger("a fox", "PXART4"), "PXART4, a fox")
        self.assertEqual(lora.with_trigger("a pxart4 fox", "PXART4"), "a pxart4 fox")
        self.assertEqual(lora.with_trigger("a fox", ""), "a fox")


class LoraCase(FluxCase):
    def found(self, repo=PIXEL, file="pixel.safetensors", trigger="PXART4", error=""):
        out = {"repo": repo, "file": file, "id": f"{repo}:{file}", "trigger": trigger, "note": "", "size": 100}
        return {**out, "error": error} if error else out

    def download_lora(self, repo=PIXEL, file="pixel.safetensors", header=None):
        folder = self.hub / ("models--" + repo.replace("/", "--"))
        (folder / "refs").mkdir(parents=True, exist_ok=True)
        (folder / "refs" / "main").write_text("abc123")
        path = folder / "snapshots" / "abc123" / file
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(file_bytes(header or lora_header()))
        return folder


class SavedTest(LoraCase):
    def test_added_kept_changed_and_removed(self):
        lora.add(self.found())
        lora.add(self.found())
        self.assertEqual([item["id"] for item in lora.saved()], [PIXEL + ":pixel.safetensors"])
        self.assertEqual(lora.set_trigger(PIXEL + ":pixel.safetensors", "  pixel   art ")["trigger"], "pixel art")
        self.assertEqual(lora.saved()[0]["trigger"], "pixel art")
        with self.assertRaises(lora.LoraError):
            lora.set_trigger("a/b:c.safetensors", "x")
        self.assertTrue(lora.remove(PIXEL + ":pixel.safetensors"))
        self.assertFalse(lora.remove(PIXEL + ":pixel.safetensors"))
        self.assertEqual(lora.saved(), [])

    def test_one_that_cannot_be_drawn_is_never_kept(self):
        with self.assertRaises(lora.LoraError):
            lora.add(self.found(error="It is a DoRA, which mflux does not load."))
        self.assertEqual(lora.saved(), [])

    def test_a_broken_list_is_read_as_empty(self):
        self.home.mkdir(parents=True)
        (self.home / "loras.json").write_text("{not json")
        self.assertEqual(lora.saved(), [])
        (self.home / "loras.json").write_text(json.dumps({"loras": [{"repo": "../x", "file": "a.safetensors"}, {"file": 3}]}))
        self.assertEqual(lora.saved(), [])

    def test_ready_once_downloaded_in_full(self):
        lora.add(self.found())
        self.assertEqual([(item["ready"], item["error"]) for item in lora.listing()], [(False, "")])
        self.assertIsNone(lora.find(PIXEL + ":pixel.safetensors"))
        folder = self.download_lora()
        (folder / "blobs").mkdir()
        (folder / "blobs" / "x.incomplete").write_bytes(b"")
        self.assertFalse(lora.listing()[0]["ready"])
        (folder / "blobs" / "x.incomplete").unlink()
        self.assertTrue(lora.listing()[0]["ready"])
        self.assertTrue(lora.find(PIXEL + ":pixel.safetensors")["path"].endswith("pixel.safetensors"))

    def test_a_downloaded_file_that_cannot_be_drawn_says_why(self):
        lora.add(self.found())
        self.download_lora(header=lora_header(width=4096))
        item = lora.listing()[0]
        self.assertFalse(item["ready"])
        self.assertIn("9B", item["error"])


class DrawTest(LoraCase):
    def setUp(self):
        super().setUp()
        self.download(BASE)
        lora.add(self.found())
        self.download_lora()

    def draw(self, model=BASE, prompt="a fox", **extra):
        request = imagegen.clean_request({"model": model, "prompt": prompt, "seed": 7, **extra})
        imagegen.attach_lora(request, flux.find(model))
        return request, engine.make_picture(flux.find(model), request, lambda *_: None, lambda: False)

    def test_drawn_with_the_lora_and_its_trigger_word(self):
        request, image = self.draw(lora=PIXEL + ":pixel.safetensors", lora_strength=0.8)
        self.assertIn(b"prompt=PXART4, a fox| lora=pixel.safetensors@0.8 ", image)
        self.assertEqual(request["lora"]["label"], "pixel-art-lora-flux.2-klein-4B")
        record = imagegen.save(image, request)
        self.assertEqual(record["lora"], {"id": PIXEL + ":pixel.safetensors", "label": "pixel-art-lora-flux.2-klein-4B",
                                          "strength": 0.8, "trigger": "PXART4"})
        self.assertEqual(record["prompt"], "PXART4, a fox")

    def test_another_lora_or_none_loads_the_model_again(self):
        _, first = self.draw(lora=PIXEL + ":pixel.safetensors")
        self.assertIn(b"lora=pixel.safetensors@1 ", first)
        _, plain = self.draw()
        self.assertIn(b"prompt=a fox| refs=", plain)
        _, stronger = self.draw(lora=PIXEL + ":pixel.safetensors", lora_strength=1.5)
        self.assertIn(b"lora=pixel.safetensors@1.5 ", stronger)
        self.assertEqual(len(engine.running()), 1)

    def test_only_on_the_base_4b(self):
        self.download(KLEIN)
        request = imagegen.clean_request({"model": KLEIN, "prompt": "a fox", "lora": PIXEL + ":pixel.safetensors"})
        with self.assertRaises(ValueError) as caught:
            imagegen.attach_lora(request, flux.find(KLEIN))
        self.assertIn("base 4B", str(caught.exception))

    def test_not_downloaded_or_not_in_the_list(self):
        for lora_id in ["a/b:c.safetensors"]:
            request = imagegen.clean_request({"model": BASE, "prompt": "a fox", "lora": lora_id})
            with self.assertRaises(ValueError) as caught:
                imagegen.attach_lora(request, flux.find(BASE))
            self.assertIn("not downloaded", str(caught.exception))

    def test_the_strength_is_checked(self):
        for strength in [0, -1, 2.5, "strong", True]:
            with self.assertRaises(ValueError, msg=strength):
                imagegen.clean_request({"model": BASE, "prompt": "a fox", "lora": "a/b:c.safetensors", "lora_strength": strength})


class ApiTest(LoraCase):
    def setUp(self):
        super().setUp()
        self.http = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
        import threading
        threading.Thread(target=self.http.serve_forever, daemon=True).start()
        self.addCleanup(self.http.server_close)
        self.addCleanup(self.http.shutdown)
        self.base = f"http://127.0.0.1:{self.http.server_address[1]}"

    def call(self, path: str, payload: dict | None = None):
        data = json.dumps(payload).encode() if payload is not None else None
        request = urllib.request.Request(self.base + path, data, {"Content-Type": "application/json", "Origin": self.base})
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                text = response.read().decode()
        except urllib.error.HTTPError as error:
            return error.code, json.loads(error.read())
        lines = [json.loads(line) for line in text.splitlines() if line.strip()]
        return 200, lines[0] if len(lines) == 1 else lines

    def test_looked_up_then_added_then_changed_and_removed(self):
        hub = FakeHub(PIXEL, {"pixel.safetensors": lora_header()},
                      {"base_model": "black-forest-labs/FLUX.2-klein-base-4B", "instance_prompt": "PXART4"})
        with mock.patch.object(lora, "_curl", hub):
            code, body = self.call("/api/loras", {"repo": PIXEL})
            self.assertEqual((code, body["found"]["trigger"], body["loras"]), (200, "PXART4", []))
            code, body = self.call("/api/loras", {"repo": body["found"]["id"], "add": True})
        self.assertEqual([item["id"] for item in body["loras"]], [PIXEL + ":pixel.safetensors"])
        self.assertFalse(body["loras"][0]["ready"])
        self.download_lora()
        code, body = self.call("/api/loras")
        self.assertEqual((body["for"], body["loras"][0]["ready"]), (lora.VARIANT, True))
        code, body = self.call("/api/loras/trigger", {"id": PIXEL + ":pixel.safetensors", "trigger": "pixel"})
        self.assertEqual(body["lora"]["trigger"], "pixel")
        code, body = self.call("/api/loras/remove", {"id": PIXEL + ":pixel.safetensors"})
        self.assertTrue(body["removed"])

    def test_one_that_cannot_be_drawn_is_not_added(self):
        hub = FakeHub("a/b", {"b.safetensors": lora_header(width=4096)})
        with mock.patch.object(lora, "_curl", hub):
            code, body = self.call("/api/loras", {"repo": "a/b", "add": True})
        self.assertIn("9B", body["found"]["error"])
        self.assertEqual(body["loras"], [])

    def test_a_bad_name_is_answered_400(self):
        code, body = self.call("/api/loras", {"repo": "nope"})
        self.assertEqual(code, 400)
        self.assertIn("user/name", body["error"])

    def test_a_picture_is_drawn_with_the_lora(self):
        self.download(BASE)
        lora.add(self.found())
        self.download_lora()
        code, lines = self.call("/api/generate", {"model": BASE, "prompt": "a fox", "lora": PIXEL + ":pixel.safetensors"})
        self.assertEqual(lines[0]["lora"], "pixel-art-lora-flux.2-klein-4B")
        record = lines[-1]["image"]
        self.assertEqual(record["lora"]["strength"], 1.0)
        self.assertIn(b"lora=pixel.safetensors@1 ", imagegen.read(record["id"]))
        self.assertIn("variant", imagegen.picture_models()[0][0])

    def test_a_picture_with_a_lora_on_another_model_is_refused(self):
        self.download(KLEIN)
        lora.add(self.found())
        self.download_lora()
        code, body = self.call("/api/generate", {"model": KLEIN, "prompt": "a fox", "lora": PIXEL + ":pixel.safetensors"})
        self.assertEqual(code, 400)
        self.assertIn("base 4B", body["error"])


if __name__ == "__main__":
    unittest.main()
