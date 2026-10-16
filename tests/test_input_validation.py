import contextlib
import io
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import infer


class InputValidationTests(unittest.TestCase):
    def run_main(self, arguments):
        stderr = io.StringIO()
        with mock.patch.object(sys, "argv", ["infer.py", *arguments]), contextlib.redirect_stderr(stderr):
            infer.main()

    def test_missing_input_mode_is_rejected_before_server_start(self):
        with (
            mock.patch.object(infer, "start_server") as start_server,
            self.assertRaises(SystemExit) as error,
        ):
            self.run_main([])

        self.assertEqual(error.exception.code, 2)
        start_server.assert_not_called()

    def test_missing_image_directory_is_rejected_before_server_start(self):
        with tempfile.TemporaryDirectory() as temporary:
            missing = Path(temporary) / "missing"
            with (
                mock.patch.object(infer, "start_server") as start_server,
                self.assertRaises(SystemExit) as error,
            ):
                self.run_main(["--image_dir", str(missing)])

        self.assertIn("Image directory does not exist", str(error.exception))
        start_server.assert_not_called()

    def test_missing_pdf_is_rejected_before_server_start(self):
        with tempfile.TemporaryDirectory() as temporary:
            missing = Path(temporary) / "missing.pdf"
            with (
                mock.patch.object(infer, "start_server") as start_server,
                self.assertRaises(SystemExit) as error,
            ):
                self.run_main(["--pdf", str(missing)])

        self.assertIn("PDF file does not exist", str(error.exception))
        start_server.assert_not_called()

    def test_unreadable_pdf_is_rejected_before_server_start(self):
        with tempfile.TemporaryDirectory() as temporary:
            invalid_pdf = Path(temporary) / "invalid.pdf"
            invalid_pdf.write_bytes(b"not a PDF")
            fake_fitz = types.SimpleNamespace(
                open=mock.Mock(side_effect=ValueError("invalid document"))
            )
            with (
                mock.patch.dict(sys.modules, {"fitz": fake_fitz}),
                mock.patch.object(infer, "start_server") as start_server,
                self.assertRaises(SystemExit) as error,
            ):
                self.run_main(["--pdf", str(invalid_pdf)])

        self.assertIn("Unable to read PDF file", str(error.exception))
        start_server.assert_not_called()

    def test_valid_image_directory_starts_server_and_runs(self):
        with tempfile.TemporaryDirectory() as temporary:
            image_dir = Path(temporary) / "images"
            image_dir.mkdir()
            image_path = image_dir / "page.png"
            image_path.write_bytes(b"image")
            events = []
            processed_images = []

            with (
                mock.patch.object(
                    infer,
                    "start_server",
                    side_effect=lambda args: events.append("start") or None,
                ),
                mock.patch.object(
                    infer,
                    "run",
                    side_effect=lambda args, images: processed_images.extend(images)
                    or events.append("run"),
                ),
                mock.patch.object(
                    infer,
                    "stop_server",
                    side_effect=lambda process: events.append("stop"),
                ),
            ):
                self.run_main(["--image_dir", str(image_dir)])

        self.assertEqual(events, ["start", "run", "stop"])
        self.assertEqual(processed_images, [str(image_path)])


if __name__ == "__main__":
    unittest.main()
