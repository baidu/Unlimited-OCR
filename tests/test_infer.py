import argparse
import os
import shutil
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

# Provide a fitz stub before importing infer so the inline `import fitz`
# inside pdf_to_images resolves to the mock rather than requiring PyMuPDF.
_fitz_stub = MagicMock()
sys.modules.setdefault("fitz", _fitz_stub)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import infer  # noqa: E402


def _make_args(**kwargs):
    defaults = dict(pdf="", image_dir="", output_dir="", image_mode="base")
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


class TestPdfToImages(unittest.TestCase):
    def _mock_fitz(self, page_count=2):
        mock_page = MagicMock()
        mock_doc = MagicMock()
        mock_doc.__iter__ = MagicMock(return_value=iter([mock_page] * page_count))
        _fitz_stub.open.return_value = mock_doc
        _fitz_stub.Matrix.return_value = MagicMock()
        return mock_doc

    def test_returns_tuple_of_paths_and_tmpdir(self):
        """pdf_to_images returns (image_paths, tmp_dir), not just image_paths."""
        self._mock_fitz(page_count=2)
        result = infer.pdf_to_images("fake.pdf", dpi=72)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        image_paths, tmp_dir = result
        self.assertIsInstance(image_paths, list)
        self.assertIsInstance(tmp_dir, str)
        shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_tmpdir_exists_until_caller_removes_it(self):
        """tmpdir is created and stays alive — caller is responsible for cleanup."""
        self._mock_fitz(page_count=1)
        _, tmp_dir = infer.pdf_to_images("fake.pdf", dpi=72)
        self.assertTrue(os.path.isdir(tmp_dir), "tmpdir should exist after pdf_to_images returns")
        shutil.rmtree(tmp_dir)
        self.assertFalse(os.path.exists(tmp_dir), "tmpdir should be gone after cleanup")


class TestBuildJobs(unittest.TestCase):
    def test_pdf_mode_raises_for_gundam_image_mode(self):
        """build_jobs raises ValueError when --image_mode gundam is used with --pdf."""
        args = _make_args(pdf="doc.pdf", image_mode="gundam")
        with self.assertRaises(ValueError, msg="gundam is invalid in PDF mode"):
            infer.build_jobs(args)

    def test_pdf_mode_accepts_base_image_mode(self):
        """build_jobs succeeds with --image_mode base + --pdf."""
        created = tempfile.mkdtemp()
        args = _make_args(pdf="doc.pdf", image_mode="base", output_dir="")
        with patch("infer.pdf_to_images", return_value=([], created)):
            jobs, tmp_dir = infer.build_jobs(args)
        self.assertEqual(jobs, [])
        shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_image_dir_mode_returns_none_tmpdir(self):
        """build_jobs returns None for tmp_dir in image_dir mode — no tempdir is created."""
        args = _make_args(image_dir="/some/dir", output_dir="")
        with patch("infer.collect_dataset_images", return_value=[]):
            jobs, tmp_dir = infer.build_jobs(args)
        self.assertIsNone(tmp_dir)

    def test_error_message_mentions_base(self):
        """ValueError message tells the user to use --image_mode base."""
        args = _make_args(pdf="doc.pdf", image_mode="gundam")
        with self.assertRaises(ValueError) as ctx:
            infer.build_jobs(args)
        self.assertIn("base", str(ctx.exception))


class TestRunCleansUpTmpdir(unittest.TestCase):
    def test_tmpdir_removed_after_run(self):
        """run() removes the PDF tmpdir even when inference raises an exception."""
        tmp_dir = tempfile.mkdtemp()
        args = _make_args(pdf="doc.pdf", image_mode="base", output_dir="", concurrency=1)

        with patch("infer.build_jobs", return_value=([], tmp_dir)), \
             patch("infer.start_server", return_value=None):
            infer.run(args)

        self.assertFalse(os.path.exists(tmp_dir), "run() must clean up tmpdir after completion")

    def test_tmpdir_removed_even_on_exception(self):
        """run() cleans up tmpdir even when inference raises an exception."""
        tmp_dir = tempfile.mkdtemp()
        args = _make_args(pdf="doc.pdf", image_mode="base", output_dir="", concurrency=1)

        def boom(*a, **kw):
            raise RuntimeError("simulated inference failure")

        with patch("infer.build_jobs", return_value=([(tmp_dir + "/p.png", None)], tmp_dir)), \
             patch("infer.infer_one", side_effect=boom):
            with self.assertRaises(RuntimeError):
                infer.run(args)

        self.assertFalse(os.path.exists(tmp_dir), "run() must clean up tmpdir even on failure")


if __name__ == "__main__":
    unittest.main()
