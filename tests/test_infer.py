import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import infer


class DatasetOutputPathTest(unittest.TestCase):
    def _args(self, image_dir, output_dir):
        return SimpleNamespace(
            pdf="",
            image_dir=image_dir,
            output_dir=output_dir,
            concurrency=1,
            image_mode="gundam",
        )

    def test_dataset_outputs_preserve_relative_paths(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            image_dir = os.path.join(tmp_dir, "images")
            output_dir = os.path.join(tmp_dir, "outputs")
            os.makedirs(os.path.join(image_dir, "sub"))
            for path in (
                os.path.join(image_dir, "sub", "doc.png"),
                os.path.join(image_dir, "sub__doc.png"),
            ):
                with open(path, "wb") as image_file:
                    image_file.write(b"image")

            jobs = infer.build_jobs(self._args(image_dir, output_dir))
            output_files = {os.path.normpath(output_file) for _, output_file in jobs}

            self.assertEqual(
                output_files,
                {
                    os.path.normpath(os.path.join(output_dir, "sub", "doc.md")),
                    os.path.normpath(os.path.join(output_dir, "sub__doc.md")),
                },
            )

    def test_run_creates_nested_dataset_output_directories(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            image_dir = os.path.join(tmp_dir, "images")
            output_dir = os.path.join(tmp_dir, "outputs")
            os.makedirs(os.path.join(image_dir, "sub"))
            with open(os.path.join(image_dir, "sub", "doc.png"), "wb") as image_file:
                image_file.write(b"image")

            args = self._args(image_dir, output_dir)
            with patch.object(
                infer,
                "infer_one",
                return_value={"tokens": 1, "decode_time": 0, "text": "ok"},
            ):
                infer.run(args)

            self.assertTrue(os.path.isdir(os.path.join(output_dir, "sub")))


if __name__ == "__main__":
    unittest.main()