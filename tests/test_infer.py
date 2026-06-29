import argparse
import sys
from unittest.mock import patch

import infer


def test_parse_args_defaults():
    with patch.object(sys, "argv", ["infer.py", "--pdf", "test.pdf"]):
        args = infer.parse_args()
    assert args.attention_backend == "fa3"
    assert args.page_size == 1
    assert args.mem_fraction_static == 0.8
    assert args.concurrency == 8
    assert args.gpu == "0"
    assert args.image_mode == "gundam"
    assert args.output_dir == "./outputs"
    assert args.server_log == "./log/sglang_server.log"


def test_parse_args_attention_backend_flashinfer():
    with patch.object(
        sys,
        "argv",
        ["infer.py", "--pdf", "test.pdf", "--attention_backend", "flashinfer"],
    ):
        args = infer.parse_args()
    assert args.attention_backend == "flashinfer"


def test_parse_args_attention_backend_triton():
    with patch.object(
        sys, "argv", ["infer.py", "--pdf", "test.pdf", "--attention_backend", "triton"]
    ):
        args = infer.parse_args()
    assert args.attention_backend == "triton"


def test_parse_args_attention_backend_all_valid_choices():
    valid = ("fa3", "flashinfer", "triton", "fa4", "flashmla", "cutlass")
    for backend in valid:
        with patch.object(
            sys,
            "argv",
            ["infer.py", "--pdf", "test.pdf", "--attention_backend", backend],
        ):
            args = infer.parse_args()
        assert args.attention_backend == backend


def test_parse_args_attention_backend_invalid_choice():
    with patch.object(
        sys,
        "argv",
        ["infer.py", "--pdf", "test.pdf", "--attention_backend", "unknown"],
    ):
        try:
            infer.parse_args()
            assert False, "Expected SystemExit for invalid choice"
        except SystemExit:
            pass


def test_parse_args_page_size():
    with patch.object(
        sys, "argv", ["infer.py", "--pdf", "test.pdf", "--page_size", "4"]
    ):
        args = infer.parse_args()
    assert args.page_size == 4
    assert isinstance(args.page_size, int)


def test_parse_args_mem_fraction_static():
    with patch.object(
        sys,
        "argv",
        ["infer.py", "--pdf", "test.pdf", "--mem_fraction_static", "0.6"],
    ):
        args = infer.parse_args()
    assert args.mem_fraction_static == 0.6
    assert isinstance(args.mem_fraction_static, float)


def test_parse_args_image_mode_gundam():
    with patch.object(
        sys, "argv", ["infer.py", "--pdf", "test.pdf", "--image_mode", "gundam"]
    ):
        args = infer.parse_args()
    assert args.image_mode == "gundam"


def test_parse_args_image_mode_base():
    with patch.object(
        sys, "argv", ["infer.py", "--pdf", "test.pdf", "--image_mode", "base"]
    ):
        args = infer.parse_args()
    assert args.image_mode == "base"


def test_parse_args_image_mode_invalid():
    with patch.object(
        sys,
        "argv",
        ["infer.py", "--pdf", "test.pdf", "--image_mode", "invalid"],
    ):
        try:
            infer.parse_args()
            assert False, "Expected SystemExit for invalid choice"
        except SystemExit:
            pass


def test_parse_args_concurrency():
    with patch.object(
        sys, "argv", ["infer.py", "--pdf", "test.pdf", "--concurrency", "16"]
    ):
        args = infer.parse_args()
    assert args.concurrency == 16


def test_parse_args_gpu():
    with patch.object(sys, "argv", ["infer.py", "--pdf", "test.pdf", "--gpu", "1"]):
        args = infer.parse_args()
    assert args.gpu == "1"


def test_parse_args_output_dir():
    with patch.object(
        sys,
        "argv",
        ["infer.py", "--pdf", "test.pdf", "--output_dir", "/tmp/outputs"],
    ):
        args = infer.parse_args()
    assert args.output_dir == "/tmp/outputs"


def test_parse_args_image_dir():
    with patch.object(
        sys,
        "argv",
        ["infer.py", "--image_dir", "/tmp/images"],
    ):
        args = infer.parse_args()
    assert args.image_dir == "/tmp/images"


def test_parse_args_missing_input_raises():
    with patch.object(sys, "argv", ["infer.py"]):
        args = infer.parse_args()
    assert not args.image_dir
    assert not args.pdf
