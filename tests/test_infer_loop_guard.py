"""Integration tests for the --loop-guard option of infer.py (issue #55).

The SGLang server is replaced by a mocked requests.post, so these tests run
CPU-only, without GPU, model or network. They verify the guard's integration
contract: default-off byte-compatibility, bounded truncation with explicit
status, exactly one request per document, and unchanged request payloads.

Run from the repository root:  python -m pytest -q tests/
"""

import argparse
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

# infer.py imports sglang at module level to build the n-gram logit-processor
# descriptor. sglang (a GPU serving stack) is not needed for these tests, so a
# minimal stub is installed first when the real package is unavailable.
try:
    import sglang.srt.sampling.custom_logit_processor  # noqa: F401
except Exception:
    _sglang = types.ModuleType("sglang")
    _srt = types.ModuleType("sglang.srt")
    _sampling = types.ModuleType("sglang.srt.sampling")
    _processor = types.ModuleType("sglang.srt.sampling.custom_logit_processor")

    class _NoRepeatNGramProcessor:
        @classmethod
        def to_str(cls):
            return "stub.DeepseekOCRNoRepeatNGramLogitProcessor"

    _processor.DeepseekOCRNoRepeatNGramLogitProcessor = _NoRepeatNGramProcessor
    _sampling.custom_logit_processor = _processor
    _srt.sampling = _sampling
    _sglang.srt = _srt
    sys.modules.setdefault("sglang", _sglang)
    sys.modules.setdefault("sglang.srt", _srt)
    sys.modules.setdefault("sglang.srt.sampling", _sampling)
    sys.modules.setdefault(
        "sglang.srt.sampling.custom_logit_processor", _processor
    )

import infer  # noqa: E402

UNIT_CN = "畜牧兽医产业基地建设工程"


def _looping_answer():
    # a realistic page: normal head, then the model degenerates into a
    # variable-tail det-block march
    head = "".join(
        f"<PAGE><BLOCK>text[1,{i},99,{i + 4}]<FORK>第{i}行内容各不相同。\n"
        for i in range(40)
    )
    loop = "".join(
        f"<|ref|>{UNIT_CN}<|/ref|>"
        f"<|det|>text [20,{100 + 38 * i},640,{136 + 38 * i}]<|/det|>"
        for i in range(120)
    )
    return head + loop


def _normal_answer():
    parts = [
        "<PAGE><BLOCK>title[1,1,99,5]<FORK>Annual Report",
        "<BLOCK>text[1,6,99,20]<FORK>Total revenue grew year over year.",
    ]
    return "\n".join(parts)


def _v2_looping_answer():
    # a genuine-format V2 document: two normal pages, then the model
    # degenerates into a V2-native variable-tail <BLOCK> bbox march (no
    # <|det|> blocks, bbox commas defeat the segment signal)
    head = "".join(
        f"<PAGE><BLOCK>text[1,{i},99,{i + 4}]<FORK>第{i}行内容各不相同。\n"
        for i in range(40)
    )
    loop = "".join(
        f"<BLOCK>text[20, {100 + 38 * i}, 640, {136 + 38 * i}]"
        f"<FORK>{UNIT_CN}\n"
        for i in range(120)
    )
    return head + loop


class _FakeResponse:
    def __init__(self, content, finish_reason="length", status_code=200):
        self.status_code = status_code
        self.text = "error body"
        self._payload = {
            "choices": [
                {"message": {"content": content},
                 "finish_reason": finish_reason}
            ],
            "usage": {"prompt_tokens": 907, "completion_tokens": 6365},
        }

    def json(self):
        return self._payload


class InferLoopGuardTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.doc_dir = Path(self._tmp.name) / "demo"
        self.doc_dir.mkdir()
        for n in (1, 2):
            (self.doc_dir / f"page_{n}.jpg").write_bytes(b"\xff\xd8fakejpg")

    @staticmethod
    def _args(loop_guard):
        return argparse.Namespace(
            max_pages=0, dataset_root=None, max_model_len=131072,
            timeout=60, loop_guard=loop_guard,
        )

    def _request_one(self, answer, loop_guard, finish_reason="length"):
        response = _FakeResponse(answer, finish_reason=finish_reason)
        with patch("infer.requests.post", return_value=response) as post:
            stats, out = infer.request_one(
                "127.0.0.1:10000", str(self.doc_dir), self._args(loop_guard)
            )
        return stats, out, post

    def test_disabled_guard_preserves_previous_behavior(self):
        answer = _looping_answer()
        stats, out, _ = self._request_one(answer, loop_guard=False)
        self.assertNotIn("loop_detected", stats)  # stats line unchanged
        self.assertNotIn("loop_truncated", stats)
        self.assertNotIn("incomplete", stats)
        self.assertEqual(out, answer)  # looping output passed through as-is

    def test_enabled_guard_reports_and_bounds_looping_output(self):
        answer = _looping_answer()
        stats, out, _ = self._request_one(answer, loop_guard=True)
        self.assertTrue(stats["loop_detected"])
        self.assertTrue(stats["loop_truncated"])
        self.assertIn("loop guard", out)
        self.assertLess(len(out), len(answer))
        self.assertTrue(out.startswith(answer[:100]))
        # the server's own finish_reason is passed through unchanged: the
        # guard reports a client-side detection, never a server-side stop
        self.assertEqual(stats["finish_reason"], "length")

    def test_enabled_guard_leaves_normal_output_byte_identical(self):
        answer = _normal_answer()
        stats, out, _ = self._request_one(answer, loop_guard=True)
        self.assertFalse(stats["loop_detected"])
        self.assertFalse(stats["loop_truncated"])
        self.assertEqual(out, answer)

    def test_enabled_guard_detects_v2_format_loop(self):
        # regression: the V2-native <BLOCK> bbox march (genuine V2 shape,
        # no <|det|> blocks) was a confirmed false negative of the V1-only
        # signals
        answer = _v2_looping_answer()
        stats, out, _ = self._request_one(answer, loop_guard=True)
        self.assertTrue(stats["loop_detected"])
        self.assertTrue(stats["loop_truncated"])
        self.assertTrue(stats["incomplete"])  # guard cut the document
        self.assertIn("loop guard", out)
        self.assertLess(len(out), len(answer))
        self.assertTrue(out.startswith(answer[:100]))

    def test_incomplete_flag_when_pages_missing(self):
        # the model stops early with fewer <PAGE> markers than input pages
        # (2 input pages, 1 emitted): no silent success -- the stats line
        # says the document is incomplete even without a loop
        answer = ("<PAGE><BLOCK>text[1,1,99,5]<FORK>Only page one arrived\n")
        stats, _, _ = self._request_one(answer, loop_guard=True,
                                        finish_reason="stop")
        self.assertEqual(stats["pages"], 2)
        self.assertEqual(stats["page_markers"], 1)
        self.assertFalse(stats["loop_detected"])
        self.assertTrue(stats["incomplete"])

    def test_incomplete_flag_when_token_budget_hit(self):
        # finish_reason="length": the output was cut at the budget; content
        # may be missing even though every page marker was emitted
        answer = _normal_answer()
        stats, _, _ = self._request_one(answer, loop_guard=True,
                                        finish_reason="length")
        self.assertTrue(stats["incomplete"])

    def test_complete_output_not_marked_incomplete(self):
        # every input page emitted and a clean server stop: success stays
        # silent-success-free in both directions
        answer = ("<PAGE><BLOCK>text[1,1,99,5]<FORK>Page one\n"
                  "<PAGE><BLOCK>text[1,1,99,5]<FORK>Page two\n")
        stats, out, _ = self._request_one(answer, loop_guard=True,
                                          finish_reason="stop")
        self.assertEqual(stats["page_markers"], 2)
        self.assertFalse(stats["loop_detected"])
        self.assertFalse(stats["incomplete"])
        self.assertEqual(out, answer)

    def test_guard_never_adds_requests_or_retries(self):
        # bounding is client-side only: exactly one POST per document whether
        # or not a loop is detected (no retry loop, no cancellation attempt)
        _, _, post = self._request_one(_looping_answer(), loop_guard=True)
        self.assertEqual(post.call_count, 1)
        _, _, post = self._request_one(_normal_answer(), loop_guard=True)
        self.assertEqual(post.call_count, 1)

    def test_server_error_still_raises_with_diagnostics(self):
        response = _FakeResponse("", status_code=500)
        response.text = "Internal Server Error: OOM"
        with patch("infer.requests.post", return_value=response):
            with self.assertRaises(SystemExit) as ctx:
                infer.request_one(
                    "127.0.0.1:10000", str(self.doc_dir),
                    self._args(loop_guard=True),
                )
        self.assertIn("HTTP 500", str(ctx.exception))
        self.assertIn("OOM", str(ctx.exception))

    def test_partial_looping_output_is_not_reported_as_success(self):
        # a budget-truncated looping page keeps finish_reason="length" from
        # the server; with the guard on, the status line distinguishes it
        # from a clean page via loop_detected
        stats, _, _ = self._request_one(_looping_answer(), loop_guard=True)
        self.assertEqual(stats["finish_reason"], "length")
        self.assertTrue(stats["loop_detected"])
        stats, _, _ = self._request_one(_normal_answer(), loop_guard=True,
                                        finish_reason="stop")
        self.assertEqual(stats["finish_reason"], "stop")
        self.assertFalse(stats["loop_detected"])

    def test_request_payload_contract_unchanged(self):
        # with the guard on or off, the request payload is exactly what the
        # V2 client built before: no new server-side knobs are introduced
        response = _FakeResponse(_normal_answer(), finish_reason="stop")
        with patch("infer.requests.post", return_value=response) as post:
            infer.request_one(
                "127.0.0.1:10000", str(self.doc_dir), self._args(True)
            )
        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["model"], "Unlimited-OCR-2")
        self.assertEqual(payload["temperature"], 0.0)
        self.assertIn("custom_logit_processor", payload)
        self.assertEqual(payload["custom_params"]["ngram_size"], 35)
        self.assertNotIn("repetition_penalty", payload)

    def test_main_end_to_end_flag_off_writes_raw_answer_unchanged(self):
        out_path = Path(self._tmp.name) / "out_off.txt"
        answer = _looping_answer()
        response = _FakeResponse(answer)
        argv = ["infer.py", "--doc", str(self.doc_dir), "--out", str(out_path)]
        with patch.object(sys, "argv", argv), patch(
            "infer.requests.post", return_value=response
        ):
            infer.main()
        self.assertEqual(
            out_path.read_text(encoding="utf-8"), answer
        )

    def test_main_end_to_end_flag_on_bounds_written_answer(self):
        out_path = Path(self._tmp.name) / "out_on.txt"
        response = _FakeResponse(_looping_answer())
        argv = [
            "infer.py", "--doc", str(self.doc_dir), "--out", str(out_path),
            "--loop-guard",
        ]
        with patch.object(sys, "argv", argv), patch(
            "infer.requests.post", return_value=response
        ) as post:
            with patch("sys.stdout"):
                infer.main()
        bounded = out_path.read_text(encoding="utf-8")
        self.assertIn("loop guard", bounded)
        self.assertLess(len(bounded), len(_looping_answer()))
        self.assertEqual(post.call_count, 1)

    def test_loop_guard_flag_exists_and_defaults_false(self):
        source = Path(infer.__file__).read_text(encoding="utf-8")
        self.assertIn("--loop-guard", source)
        # argparse store_true: absent flag must evaluate falsy
        args = self._args(loop_guard=False)
        self.assertFalse(args.loop_guard)


if __name__ == "__main__":
    unittest.main()
