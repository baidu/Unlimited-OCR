from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import infer


def make_args(**overrides):
    values = {
        "pdf": "",
        "image_dir": "",
        "output_dir": "",
        "concurrency": 1,
        "image_mode": "gundam",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class InputValidationTests(unittest.TestCase):
    def test_main_rejects_missing_image_directory_before_starting_server(self):
        args = make_args(image_dir="does-not-exist")

        with (
            patch("infer.parse_args", return_value=args),
            patch("infer.start_server") as start_server,
            contextlib.redirect_stderr(io.StringIO()) as stderr,
        ):
            exit_code = infer.main()

        self.assertEqual(exit_code, 2)
        self.assertIn("Image directory not found", stderr.getvalue())
        start_server.assert_not_called()

    def test_main_returns_run_failure_status_and_stops_server(self):
        args = make_args(image_dir="images")
        server_process = object()

        with (
            patch("infer.parse_args", return_value=args),
            patch("infer.build_jobs", return_value=[("image.png", None)]),
            patch("infer.start_server", return_value=server_process),
            patch("infer.run", return_value=1),
            patch("infer.stop_server") as stop_server,
        ):
            exit_code = infer.main()

        self.assertEqual(exit_code, 1)
        stop_server.assert_called_once_with(server_process)

    def test_build_jobs_rejects_empty_image_directory(self):
        with tempfile.TemporaryDirectory() as image_dir:
            with self.assertRaisesRegex(ValueError, "No supported images found"):
                infer.build_jobs(make_args(image_dir=image_dir))

    def test_build_jobs_rejects_missing_pdf_before_conversion(self):
        with self.assertRaisesRegex(ValueError, "PDF file not found"):
            infer.build_jobs(make_args(pdf="missing.pdf"))


class RunExitStatusTests(unittest.TestCase):
    def test_run_returns_nonzero_when_any_request_fails(self):
        args = make_args()
        jobs = [("first.png", None), ("second.png", None)]
        results = [
            {"success": True, "tokens": 3, "decode_time": 1.0, "text": "ok"},
            {"success": False, "tokens": 0, "decode_time": 0.0, "text": ""},
        ]

        with patch("infer.infer_one", side_effect=results):
            exit_code = infer.run(args, jobs)

        self.assertEqual(exit_code, 1)

    def test_run_returns_zero_when_every_request_succeeds(self):
        args = make_args()
        jobs = [("first.png", None), ("second.png", None)]
        results = [
            {"success": True, "tokens": 3, "decode_time": 1.0, "text": "one"},
            {"success": True, "tokens": 2, "decode_time": 2.0, "text": "two"},
        ]

        with patch("infer.infer_one", side_effect=results):
            exit_code = infer.run(args, jobs)

        self.assertEqual(exit_code, 0)


if __name__ == "__main__":
    unittest.main()
