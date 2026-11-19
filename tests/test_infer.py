"""Unit tests for infer.py utility functions.

Tests cover pure-logic helpers that do NOT require GPU, SGLang, or model weights.
Run with: python -m pytest tests/ -v
"""

import base64
import os
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# encode_image
# ---------------------------------------------------------------------------

class TestEncodeImage:
    """Tests for the encode_image helper."""

    def test_png_mime(self, tmp_path):
        """PNG files should get image/png MIME type."""
        from infer import encode_image

        png_file = tmp_path / "test.png"
        # 1x1 red pixel PNG
        png_data = (
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
            b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00"
            b"\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00"
            b"\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        png_file.write_bytes(png_data)

        result = encode_image(str(png_file))
        assert result["type"] == "image_url"
        assert result["image_url"]["url"].startswith("data:image/png;base64,")

    def test_jpg_mime(self, tmp_path):
        """JPEG files should get image/jpeg MIME type."""
        from infer import encode_image

        jpg_file = tmp_path / "test.jpg"
        jpg_file.write_bytes(b"\xff\xd8\xff\xe0fake_jpeg")

        result = encode_image(str(jpg_file))
        assert "data:image/jpeg;base64," in result["image_url"]["url"]

    def test_jpeg_mime(self, tmp_path):
        """Files with .jpeg extension should also get image/jpeg."""
        from infer import encode_image

        jpeg_file = tmp_path / "test.jpeg"
        jpeg_file.write_bytes(b"\xff\xd8\xff\xe0fake_jpeg")

        result = encode_image(str(jpeg_file))
        assert "data:image/jpeg;base64," in result["image_url"]["url"]

    def test_base64_content(self, tmp_path):
        """Base64-encoded content should match the original file bytes."""
        from infer import encode_image

        content = b"hello world test content"
        img_file = tmp_path / "test.png"
        img_file.write_bytes(content)

        result = encode_image(str(img_file))
        url = result["image_url"]["url"]
        b64_part = url.split(",", 1)[1]
        decoded = base64.b64decode(b64_part)
        assert decoded == content

    def test_webp_mime(self, tmp_path):
        """WebP files should get image/webp MIME type."""
        from infer import encode_image

        webp_file = tmp_path / "test.webp"
        webp_file.write_bytes(b"RIFF\x00\x00\x00\x00WEBP")

        result = encode_image(str(webp_file))
        assert "data:image/webp;base64," in result["image_url"]["url"]


# ---------------------------------------------------------------------------
# build_content
# ---------------------------------------------------------------------------

class TestBuildContent:
    """Tests for the build_content helper."""

    def test_structure(self, tmp_path):
        """Should return list with text block followed by image block."""
        from infer import build_content

        img_file = tmp_path / "img.png"
        img_file.write_bytes(b"\x89PNG")

        result = build_content(str(img_file))
        assert len(result) == 2
        assert result[0]["type"] == "text"
        assert result[0]["text"] == "document parsing."
        assert result[1]["type"] == "image_url"

    def test_prompt_text(self, tmp_path):
        """First element should contain the PROMPT constant."""
        from infer import PROMPT, build_content

        img_file = tmp_path / "img.png"
        img_file.write_bytes(b"\x89PNG")

        result = build_content(str(img_file))
        assert result[0]["text"] == PROMPT


# ---------------------------------------------------------------------------
# server_ready
# ---------------------------------------------------------------------------

class TestServerReady:
    """Tests for the server_ready health check."""

    @patch("infer.requests.get")
    def test_healthy_server(self, mock_get):
        """Should return True when server responds 200."""
        from infer import server_ready

        mock_get.return_value = MagicMock(status_code=200)
        assert server_ready("http://localhost:10000") is True

    @patch("infer.requests.get")
    def test_unhealthy_server(self, mock_get):
        """Should return False when server responds 500."""
        from infer import server_ready

        mock_get.return_value = MagicMock(status_code=500)
        assert server_ready("http://localhost:10000") is False

    @patch("infer.requests.get")
    def test_connection_error(self, mock_get):
        """Should return False when connection fails."""
        import requests as req
        from infer import server_ready

        mock_get.side_effect = req.RequestException("Connection refused")
        assert server_ready("http://localhost:10000") is False


# ---------------------------------------------------------------------------
# collect_dataset_images
# ---------------------------------------------------------------------------

class TestCollectDatasetImages:
    """Tests for the collect_dataset_images file scanner."""

    def test_finds_images(self, tmp_path):
        """Should discover image files with supported extensions."""
        from infer import collect_dataset_images

        (tmp_path / "a.png").write_bytes(b"png")
        (tmp_path / "b.jpg").write_bytes(b"jpg")
        (tmp_path / "c.txt").write_bytes(b"txt")  # not an image

        result = collect_dataset_images(str(tmp_path))
        assert len(result) == 2
        names = {os.path.basename(p) for p in result}
        assert names == {"a.png", "b.jpg"}

    def test_supported_extensions(self, tmp_path):
        """Should support png, jpg, jpeg, webp, bmp."""
        from infer import collect_dataset_images

        exts = [".png", ".jpg", ".jpeg", ".webp", ".bmp"]
        for i, ext in enumerate(exts):
            (tmp_path / f"img{i}{ext}").write_bytes(b"x")

        result = collect_dataset_images(str(tmp_path))
        assert len(result) == len(exts)

    def test_empty_directory(self, tmp_path):
        """Should return empty list for directory with no images."""
        from infer import collect_dataset_images

        (tmp_path / "readme.txt").write_bytes(b"text")

        result = collect_dataset_images(str(tmp_path))
        assert result == []

    def test_sorted_alphabetically(self, tmp_path):
        """Results should be sorted (current impl: by size desc)."""
        from infer import collect_dataset_images

        # Create files with different sizes
        (tmp_path / "small.png").write_bytes(b"x")
        (tmp_path / "medium.png").write_bytes(b"xxx")
        (tmp_path / "large.png").write_bytes(b"xxxxx")

        result = collect_dataset_images(str(tmp_path))
        assert len(result) == 3
        # Current implementation sorts by size descending
        names = [os.path.basename(p) for p in result]
        assert names == ["large.png", "medium.png", "small.png"]

    def test_nested_directories(self, tmp_path):
        """Should walk subdirectories recursively."""
        from infer import collect_dataset_images

        sub = tmp_path / "sub"
        sub.mkdir()
        (tmp_path / "root.png").write_bytes(b"x")
        (sub / "nested.jpg").write_bytes(b"y")

        result = collect_dataset_images(str(tmp_path))
        assert len(result) == 2


# ---------------------------------------------------------------------------
# build_jobs
# ---------------------------------------------------------------------------

class TestBuildJobs:
    """Tests for the build_jobs job builder."""

    def test_image_dir_mode(self, tmp_path):
        """Should create jobs from image directory."""
        from infer import build_jobs

        img_dir = tmp_path / "images"
        img_dir.mkdir()
        (img_dir / "a.png").write_bytes(b"x")
        (img_dir / "b.png").write_bytes(b"y")

        out_dir = tmp_path / "outputs"

        args = MagicMock()
        args.pdf = ""
        args.image_dir = str(img_dir)
        args.output_dir = str(out_dir)

        jobs = build_jobs(args)
        assert len(jobs) == 2
        # Each job is (image_path, output_file)
        for image_path, output_file in jobs:
            assert os.path.exists(image_path)
            assert output_file.startswith(str(out_dir))

    def test_no_args_raises(self):
        """Should raise ValueError when neither --pdf nor --image_dir given."""
        from infer import build_jobs

        args = MagicMock()
        args.pdf = ""
        args.image_dir = ""

        with pytest.raises(ValueError, match="Either --image_dir or --pdf"):
            build_jobs(args)

    def test_output_dir_none(self, tmp_path):
        """Should handle None output_dir (no output files)."""
        from infer import build_jobs

        img_dir = tmp_path / "images"
        img_dir.mkdir()
        (img_dir / "a.png").write_bytes(b"x")

        args = MagicMock()
        args.pdf = ""
        args.image_dir = str(img_dir)
        args.output_dir = None

        jobs = build_jobs(args)
        assert len(jobs) == 1
        assert jobs[0][1] is None  # output_file should be None


# ---------------------------------------------------------------------------
# collect_stream_silent
# ---------------------------------------------------------------------------

class TestCollectStreamSilent:
    """Tests for the SSE stream parser."""

    def test_parses_sse_lines(self, tmp_path):
        """Should parse SSE data lines and extract content."""
        from infer import collect_stream_silent

        # Mock response with SSE lines
        lines = [
            'data: {"choices":[{"delta":{"content":"Hello"}}]}',
            'data: {"choices":[{"delta":{"content":" World"}}]}',
            "data: [DONE]",
        ]

        mock_resp = MagicMock()
        mock_resp.iter_lines.return_value = [line.encode() for line in lines]

        out_file = str(tmp_path / "out.md")
        result = collect_stream_silent(mock_resp, out_file)

        assert result["tokens"] == 2
        assert result["text"] == "Hello World"
        assert os.path.exists(out_file)

    def test_empty_response(self, tmp_path):
        """Should handle empty stream gracefully."""
        from infer import collect_stream_silent

        mock_resp = MagicMock()
        mock_resp.iter_lines.return_value = [b"data: [DONE]"]

        out_file = str(tmp_path / "out.md")
        result = collect_stream_silent(mock_resp, out_file)

        assert result["tokens"] == 0
        assert result["text"] == ""

    def test_no_output_file(self):
        """Should work without writing to file when output_file is None."""
        from infer import collect_stream_silent

        lines = [
            'data: {"choices":[{"delta":{"content":"test"}}]}',
            "data: [DONE]",
        ]

        mock_resp = MagicMock()
        mock_resp.iter_lines.return_value = [line.encode() for line in lines]

        result = collect_stream_silent(mock_resp, None)
        assert result["tokens"] == 1
        assert result["text"] == "test"

    def test_malformed_json_skipped(self, tmp_path):
        """Should skip malformed JSON lines without crashing."""
        from infer import collect_stream_silent

        lines = [
            "data: not-valid-json",
            'data: {"choices":[{"delta":{"content":"ok"}}]}',
            "data: [DONE]",
        ]

        mock_resp = MagicMock()
        mock_resp.iter_lines.return_value = [line.encode() for line in lines]

        result = collect_stream_silent(mock_resp, None)
        assert result["tokens"] == 1
        assert result["text"] == "ok"


# ---------------------------------------------------------------------------
# stop_server
# ---------------------------------------------------------------------------

class TestStopServer:
    """Tests for the stop_server cleanup helper."""

    def test_none_process(self):
        """Should handle None process gracefully."""
        from infer import stop_server

        # Should not raise
        stop_server(None)

    def test_terminates_process(self):
        """Should terminate and wait for process."""
        from infer import stop_server

        mock_proc = MagicMock()
        mock_proc.wait.return_value = None

        stop_server(mock_proc)

        mock_proc.terminate.assert_called_once()
        mock_proc.wait.assert_called_once_with(timeout=30)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestConstants:
    """Verify expected default constants exist and have sane values."""

    def test_server_constants(self):
        from infer import HOST, PORT, SERVER_URL

        assert PORT == 10000
        assert HOST == "0.0.0.0"
        assert "10000" in SERVER_URL

    def test_inference_constants(self):
        from infer import (
            CONTEXT_LENGTH,
            MAX_RETRIES,
            NGRAM_WINDOW,
            NO_REPEAT_NGRAM_SIZE,
            REQUEST_TIMEOUT,
            TEMPERATURE,
        )

        assert TEMPERATURE == 0
        assert NO_REPEAT_NGRAM_SIZE == 35
        assert NGRAM_WINDOW == 128
        assert CONTEXT_LENGTH == 32768
        assert MAX_RETRIES == 5
        assert REQUEST_TIMEOUT > 0

    def test_pdf_constants(self):
        from infer import PDF_DPI

        assert PDF_DPI == 300

    def test_model_name(self):
        from infer import SERVED_MODEL_NAME

        assert SERVED_MODEL_NAME == "Unlimited-OCR"
