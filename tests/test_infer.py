"""Unit tests for infer.py.

These tests mock the SGLang HTTP endpoint and never load the model, so they
run on CPU-only machines (and in CI) without a GPU or the sglang package.
"""

import contextlib
import io
import json
import os
import sys
import tempfile
import types
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import infer


def make_dummy_image(directory, name="page_0001.png"):
    """encode_image only base64-encodes bytes, so any file content works."""
    path = os.path.join(directory, name)
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + b"\x00" * 16)
    return path


def make_args(**overrides):
    """Build an argparse-like namespace with the defaults infer.py expects."""
    defaults = dict(
        image_dir="",
        pdf="",
        output_dir="",
        concurrency=2,
        gpu="0",
        model_dir="baidu/Unlimited-OCR",
        image_mode="gundam",
        server_log="./log/sglang_server.log",
        overwrite=False,
        keep_temp=False,
    )
    defaults.update(overrides)
    return types.SimpleNamespace(**defaults)


class FakeResponse:
    """Minimal stand-in for a streaming requests.Response."""

    def __init__(self, deltas, status_code=200, raise_mid_stream=False):
        self.status_code = status_code
        self._deltas = deltas
        self._raise_mid_stream = raise_mid_stream

    def iter_lines(self, *args, **kwargs):
        for d in self._deltas:
            chunk = {"choices": [{"delta": {"content": d}}]}
            yield f"data: {json.dumps(chunk)}".encode("utf-8")
            if self._raise_mid_stream:
                raise ConnectionError("connection dropped mid-stream")
        yield b"data: [DONE]"

    def raise_for_status(self):
        if self.status_code >= 400:
            raise infer.requests.HTTPError(f"status {self.status_code}")


class IsCompletedTest(unittest.TestCase):
    def test_missing_empty_and_nonempty(self):
        with tempfile.TemporaryDirectory() as d:
            missing = os.path.join(d, "missing.md")
            self.assertFalse(infer.is_completed(missing))

            empty = os.path.join(d, "empty.md")
            open(empty, "w").close()
            self.assertFalse(infer.is_completed(empty))

            full = os.path.join(d, "full.md")
            with open(full, "w", encoding="utf-8") as f:
                f.write("# parsed")
            self.assertTrue(infer.is_completed(full))


class CollectStreamTest(unittest.TestCase):
    def test_atomic_write_renames_part_into_place(self):
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "page.md")
            resp = FakeResponse(["Hello ", "world"])
            result = infer.collect_stream_silent(resp, out)

            self.assertEqual(result["text"], "Hello world")
            self.assertEqual(result["tokens"], 2)
            self.assertTrue(os.path.exists(out))
            self.assertFalse(os.path.exists(out + ".part"))
            with open(out, encoding="utf-8") as f:
                self.assertEqual(f.read(), "Hello world")

    def test_mid_stream_failure_leaves_no_final_file(self):
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "page.md")
            resp = FakeResponse(["partial"], raise_mid_stream=True)
            with self.assertRaises(ConnectionError):
                infer.collect_stream_silent(resp, out)
            # The final file must not exist; only a leftover .part may remain.
            self.assertFalse(os.path.exists(out))


class InferOneTest(unittest.TestCase):
    def setUp(self):
        # Avoid importing sglang for the n-gram processor string.
        patcher = mock.patch.object(infer, "get_ngram_processor_str", return_value="")
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_skips_completed_page_without_calling_server(self):
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "page.md")
            with open(out, "w", encoding="utf-8") as f:
                f.write("already parsed")

            with mock.patch.object(infer.requests, "post") as post:
                result = infer.infer_one("page_0001.png", out, make_args(), 1)
                post.assert_not_called()

            self.assertEqual(result["status"], "skipped")
            self.assertEqual(result["attempts"], 0)

    def test_overwrite_forces_rerun(self):
        with tempfile.TemporaryDirectory() as d:
            image = make_dummy_image(d)
            out = os.path.join(d, "page.md")
            with open(out, "w", encoding="utf-8") as f:
                f.write("stale")

            with mock.patch.object(
                infer.requests, "post", return_value=FakeResponse(["fresh"])
            ) as post:
                result = infer.infer_one(image, out, make_args(overwrite=True), 1)
                post.assert_called_once()

            self.assertEqual(result["status"], "ok")
            with open(out, encoding="utf-8") as f:
                self.assertEqual(f.read(), "fresh")

    def test_successful_inference_writes_output(self):
        with tempfile.TemporaryDirectory() as d:
            image = make_dummy_image(d)
            out = os.path.join(d, "page.md")
            with mock.patch.object(
                infer.requests, "post", return_value=FakeResponse(["# Title\n", "body"])
            ):
                result = infer.infer_one(image, out, make_args(), 1)

            self.assertEqual(result["status"], "ok")
            self.assertEqual(result["image"], image)
            self.assertTrue(os.path.exists(out))

    def test_failure_after_retries_cleans_partial(self):
        with tempfile.TemporaryDirectory() as d:
            image = make_dummy_image(d)
            out = os.path.join(d, "page.md")
            # Leave a stale .part behind to prove it gets cleaned on hard failure.
            with open(out + ".part", "w", encoding="utf-8") as f:
                f.write("garbage")

            with mock.patch.object(infer.requests, "post", side_effect=ConnectionError("down")), \
                    mock.patch.object(infer.time, "sleep"):
                result = infer.infer_one(image, out, make_args(), 1)

            self.assertEqual(result["status"], "failed")
            self.assertFalse(os.path.exists(out))
            self.assertFalse(os.path.exists(out + ".part"))


class BuildJobsTest(unittest.TestCase):
    def test_image_dir_jobs_and_no_tmp_dir(self):
        with tempfile.TemporaryDirectory() as d:
            for name in ("a.png", "b.jpg", "ignore.txt"):
                with open(os.path.join(d, name), "wb") as f:
                    f.write(b"\x00" * 8)

            out_dir = os.path.join(d, "out")
            args = make_args(image_dir=d, output_dir=out_dir)
            jobs, tmp_dir = infer.build_jobs(args)

            self.assertIsNone(tmp_dir)
            self.assertEqual(len(jobs), 2)  # .txt ignored
            for image_path, output_file in jobs:
                self.assertTrue(output_file.endswith(".md"))

    def test_image_dir_required(self):
        with self.assertRaises(ValueError):
            infer.build_jobs(make_args())


class ManifestTest(unittest.TestCase):
    def test_manifest_structure_and_counts(self):
        with tempfile.TemporaryDirectory() as d:
            args = make_args(output_dir=d, image_mode="base")
            results = [
                {"image": "a.png", "output": "a.md", "tokens": 10, "decode_time": 1.0,
                 "text": "x", "status": "ok", "attempts": 1},
                {"image": "b.png", "output": "b.md", "tokens": 0, "decode_time": 0,
                 "text": "", "status": "skipped", "attempts": 0},
                {"image": "c.png", "output": "c.md", "tokens": 0, "decode_time": 0,
                 "text": "", "status": "failed", "attempts": 5},
            ]
            path = infer.write_manifest(args, results, wall_time=12.5)

            with open(path, encoding="utf-8") as f:
                manifest = json.load(f)

            self.assertEqual(manifest["image_mode"], "base")
            self.assertEqual(manifest["total_tokens"], 10)
            self.assertEqual(manifest["counts"], {"ran": 3, "ok": 1, "skipped": 1, "failed": 1})
            self.assertEqual(len(manifest["jobs"]), 3)
            # Bulky generated text must not be persisted in the manifest.
            self.assertNotIn("text", manifest["jobs"][0])


class ParseArgsValidationTest(unittest.TestCase):
    def _parse(self, *argv):
        with mock.patch.object(sys, "argv", ["infer.py", *argv]):
            return infer.parse_args()

    def test_pdf_with_gundam_is_rejected(self):
        # parser.error() exits with code 2; swallow the usage text on stderr.
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                self._parse("--pdf", "doc.pdf", "--image_mode", "gundam")

    def test_pdf_with_base_is_accepted(self):
        args = self._parse("--pdf", "doc.pdf", "--image_mode", "base")
        self.assertEqual(args.pdf, "doc.pdf")
        self.assertEqual(args.image_mode, "base")

    def test_image_dir_with_gundam_is_accepted(self):
        args = self._parse("--image_dir", "imgs", "--image_mode", "gundam")
        self.assertEqual(args.image_mode, "gundam")


@unittest.skipUnless(
    __import__("importlib").util.find_spec("fitz") is not None,
    "PyMuPDF (fitz) not installed",
)
class PdfToImagesTest(unittest.TestCase):
    def test_renders_into_given_dir_and_cleanup_is_caller_controlled(self):
        import fitz

        with tempfile.TemporaryDirectory() as d:
            pdf_path = os.path.join(d, "doc.pdf")
            doc = fitz.open()
            doc.new_page()
            doc.new_page()
            doc.save(pdf_path)
            doc.close()

            out_dir = os.path.join(d, "pages")
            os.makedirs(out_dir)
            paths = infer.pdf_to_images(pdf_path, out_dir, dpi=72)

            self.assertEqual(len(paths), 2)
            for p in paths:
                self.assertTrue(os.path.exists(p))
                self.assertEqual(os.path.dirname(p), out_dir)


if __name__ == "__main__":
    unittest.main()
