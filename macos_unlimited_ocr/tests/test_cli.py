import pytest

from macos_unlimited_ocr.cli import build_inference_config, build_parser, validate_args


def test_parser_accepts_image_file_and_output_dir(tmp_path):
    image = tmp_path / "sample.png"
    image.write_bytes(b"png")
    args = build_parser().parse_args(
        ["--image", str(image), "--output-dir", str(tmp_path / "out"), "--device", "cpu"]
    )

    validate_args(args)

    assert args.image == image
    assert args.device == "cpu"


def test_validate_args_requires_one_input(tmp_path):
    args = build_parser().parse_args(["--output-dir", str(tmp_path / "out")])

    with pytest.raises(SystemExit):
        validate_args(args)


def test_validate_args_rejects_multiple_inputs(tmp_path):
    image = tmp_path / "sample.png"
    image.write_bytes(b"png")
    pdf = tmp_path / "sample.pdf"
    pdf.write_bytes(b"%PDF")
    args = build_parser().parse_args(
        ["--image", str(image), "--pdf", str(pdf), "--output-dir", str(tmp_path / "out")]
    )

    with pytest.raises(SystemExit):
        validate_args(args)


def test_build_inference_config_uses_pdf_defaults(tmp_path):
    pdf = tmp_path / "sample.pdf"
    pdf.write_bytes(b"%PDF")
    args = build_parser().parse_args(["--pdf", str(pdf), "--output-dir", str(tmp_path / "out")])

    config = build_inference_config(args)

    assert config.image_mode == "gundam"
    assert config.image_size == 640
    assert config.ngram_window == 128
    assert config.max_length == 32768
    assert config.write_text_result is False


def test_build_inference_config_uses_single_image_defaults(tmp_path):
    image = tmp_path / "sample.png"
    image.write_bytes(b"png")
    args = build_parser().parse_args(["--image", str(image), "--output-dir", str(tmp_path / "out")])

    config = build_inference_config(args)

    assert config.image_mode == "gundam"
    assert config.image_size == 640
    assert config.ngram_window == 128
    assert config.max_length == 32768


def test_build_inference_config_respects_explicit_pdf_image_mode(tmp_path):
    pdf = tmp_path / "sample.pdf"
    pdf.write_bytes(b"%PDF")
    args = build_parser().parse_args(
        ["--pdf", str(pdf), "--output-dir", str(tmp_path / "out"), "--image-mode", "gundam"]
    )

    config = build_inference_config(args)

    assert config.image_mode == "gundam"
    assert config.image_size == 640
    assert config.ngram_window == 128


def test_build_inference_config_respects_explicit_max_length(tmp_path):
    pdf = tmp_path / "sample.pdf"
    pdf.write_bytes(b"%PDF")
    args = build_parser().parse_args(
        ["--pdf", str(pdf), "--output-dir", str(tmp_path / "out"), "--max-length", "2048"]
    )

    config = build_inference_config(args)

    assert config.max_length == 2048


def test_build_inference_config_can_enable_text_sidecar(tmp_path):
    image = tmp_path / "sample.png"
    image.write_bytes(b"png")
    args = build_parser().parse_args(
        ["--image", str(image), "--output-dir", str(tmp_path / "out"), "--write-text-result"]
    )

    config = build_inference_config(args)

    assert config.write_text_result is True
