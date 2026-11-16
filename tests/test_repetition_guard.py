"""Unit tests for repetition_guard.py (issue #55 loop detector and bounding).

No GPU, no network, no third-party dependencies. Run from the repository
root:  python -m pytest -q tests/
"""

import tempfile
import unittest
from pathlib import Path

from repetition_guard import (
    TRUNCATION_MARKER,
    is_looping_output,
    looping_onset,
    main,
    truncate_looping_output,
)

UNIT_CN = "畜牧兽医产业基地建设工程"

_PHRASES = [
    "正文随段落变化甲",
    "第二种文字内容乙",
    "第三个不同句子丙",
    "另有不同表述丁",
    "再来一句变化戊",
]


def _det_loop(iterations, start_y=100):
    return "".join(
        f"<|ref|>{UNIT_CN}<|/ref|>"
        f"<|det|>text [20,{start_y + 38 * i},640,{start_y + 38 * i + 36}]"
        f"<|/det|>"
        for i in range(iterations)
    )


def _counter_loop(count, start_year=2500):
    return "，".join(f"{y}年出生" for y in range(start_year, start_year + count))


def _varied_head(n=120):
    """Head text that varies beyond its digits (a normal page does)."""
    return "".join(
        f"第{i}行内容各不相同，{_PHRASES[i % 5]}。\n" for i in range(n)
    )


class TestLoopDetector(unittest.TestCase):
    def test_exact_periodic_with_det_tags_detected(self):
        # identical units, identical bboxes (pure periodic repetition)
        text = "".join(
            f"<|ref|>{UNIT_CN}<|/ref|><|det|>text [20,100,640,136]<|/det|>"
            for _ in range(120)
        )
        self.assertTrue(is_looping_output(text))

    def test_variable_tail_bbox_march_detected(self):
        self.assertTrue(is_looping_output(_det_loop(120)))

    def test_counter_loop_detected(self):
        self.assertTrue(is_looping_output(_counter_loop(400)))

    def test_english_sentence_loop_with_marching_bbox_detected(self):
        sent = (
            "The sale is subject to the strict lien upon the sale of the "
            "sale to the purchaser in the event additional funds are "
            "tendered before the sale."
        )
        text = "".join(
            f"<|det|>text [10,{50 + 20 * i},990,{66 + 20 * i}]<|/det|>{sent}"
            for i in range(60)
        )
        self.assertTrue(is_looping_output(text))

    def test_valid_repeated_section_headers_not_detected(self):
        # the same header pattern appears in many sections, but the content
        # between them differs -- not consecutive repetition
        parts = []
        for i in range(30):
            parts.append(f"<|det|>title [10,{i},50,{i + 5}]<|/det|>第{i}章 概述")
            parts.append("本章正文内容各不相同。" * 10)
        self.assertFalse(is_looping_output("".join(parts)))

    def test_valid_table_rows_below_threshold_not_detected(self):
        # 29 identical rows (with distinct bboxes) stay below the calibrated
        # threshold of 30; measured normal pages never exceeded 1 block /
        # 4 pattern consecutive repeats
        rows = "".join(
            f"<|det|>table [20,{100 + 38 * i},640,{136 + 38 * i}]"
            f"<|/det|>|合计|1,204.50|"
            for i in range(29)
        )
        self.assertFalse(is_looping_output("# 表\n" + rows))

    def test_table_rows_at_threshold_detected(self):
        rows = "".join(
            f"<|det|>table [20,{100 + 38 * i},640,{136 + 38 * i}]"
            f"<|/det|>|合计|1,204.50|"
            for i in range(40)
        )
        self.assertTrue(is_looping_output("# 表\n" + rows))

    def test_short_loop_output_not_flagged(self):
        self.assertFalse(is_looping_output(_det_loop(10)))  # below min_chars

    def test_empty_and_none(self):
        self.assertFalse(is_looping_output(""))
        self.assertFalse(is_looping_output(None))

    def test_malformed_bbox_no_crash(self):
        bad = "<|det|>text [1,2<|/det|>" * 200
        bad += "<|det|>text [1,2,3<|/det|>" * 100
        self.assertTrue(is_looping_output(bad))  # unterminated coords collapse

    def test_unicode_mixed_normal_page_not_detected(self):
        normal = (
            "# 实验报告 Report 2026\n"
            "<|det|>title [1,1,99,5]<|/det|>深度学习在OCR中的应用\n"
            "<|det|>text [1,6,99,20]<|/det|>本文提出一种新的检测方法，"
            + "其在多个数据集上表现稳定。" * 8
            + "\n"
            "<|det|>formula [1,21,99,25]<|/det|>$E = mc^2$\n"
            "<|det|>text [1,26,99,40]<|/det|>Experimental results show that "
            "our method outperforms all baselines on dense documents. "
            "Ablation studies confirm the contribution of each component. "
            "Further analysis reveals robustness to rotation."
        )
        self.assertFalse(is_looping_output(normal))

    def test_no_separator_tight_loop_is_known_limitation(self):
        # pure text loop without det tags and without segment separators:
        # neither signal fires. Documented limitation; observed issue #55
        # loops all carry det structure or separators.
        self.assertFalse(is_looping_output(UNIT_CN * 800))

    def test_streaming_window_boundaries_do_not_change_detection(self):
        # detection is a property of the decoded text, not of the transport:
        # a streaming client that starts decoding at any offset inside the
        # normal head must still see the loop in its buffer
        head = _varied_head(140)
        text = head + _det_loop(80)
        self.assertTrue(is_looping_output(text))
        for start in (0, 1, 7, 64, len(head) // 2, len(head) - 1):
            window = text[start:]
            self.assertTrue(
                is_looping_output(window),
                f"window starting at {start} lost the verdict",
            )
        # and the verdict needs the qualifying run: windows that cut into
        # the loop below the repeat threshold must not fire
        self.assertFalse(is_looping_output(text[:len(head) + 10]))

    def test_degenerate_thresholds_are_handled(self):
        # invalid/extreme configuration must stay defined and never raise
        text = _varied_head(30)
        self.assertIsInstance(
            is_looping_output(text, min_chars=-1, block_repeats=-3,
                              pattern_repeats=-3),
            bool,
        )
        self.assertEqual(
            truncate_looping_output(None, min_chars=-1), None
        )
        self.assertEqual(truncate_looping_output("", block_repeats=0), "")


class TestLoopGuardTruncation(unittest.TestCase):
    def test_truncate_adds_marker_and_cuts_garbage(self):
        head = _varied_head(140)
        loop = _det_loop(60)
        text = head + loop
        self.assertTrue(is_looping_output(text))
        out = truncate_looping_output(text)
        self.assertIn("loop guard", out)
        self.assertIn(TRUNCATION_MARKER, out)
        self.assertLess(len(out), len(text))
        self.assertTrue(out.startswith(head[:50]))

    def test_normal_output_untouched(self):
        normal = "<|det|>title [1,1,99,5]<|/det|>标题\n" + _varied_head(150)
        self.assertEqual(truncate_looping_output(normal), normal)
        self.assertEqual(truncate_looping_output(None), None)

    def test_onset_det_loop(self):
        head = _varied_head(140)
        text = head + _det_loop(60)
        onset = looping_onset(text)
        self.assertIsNotNone(onset)
        self.assertEqual(text[onset:onset + 7], "<|det|>")

    def test_onset_counter_loop(self):
        head = _varied_head(140)
        text = head + _counter_loop(200)
        onset = looping_onset(text)
        self.assertIsNotNone(onset)
        self.assertTrue(text[onset:].startswith("2500"))

    def test_truncation_never_drops_a_normal_output(self):
        # even with absurdly sensitive thresholds the function only ever
        # shortens a *detected* run; onset-locatable, marker-appended
        text = _varied_head(50)
        out = truncate_looping_output(text, min_chars=0, block_repeats=10000,
                                      pattern_repeats=10000)
        self.assertEqual(out, text)


def _v2_loop(iterations, start_y=100, sep="\n"):
    # Unlimited-OCR-2 native shape (genuine gfx1151 outputs): blocks with
    # ", "-separated bboxes, identical recognized content, marching
    # coordinates -- the V2 analog of the issue #55 variable-tail loop.
    return sep.join(
        f"<BLOCK>text[20, {start_y + 38 * i}, 640, {start_y + 38 * i + 36}]"
        f"<FORK>{UNIT_CN}"
        for i in range(iterations))


def _v2_normal_doc(pages=3):
    # a genuine-format multi-page document whose content varies per block
    # (english test card from the gfx1151 evidence, extended)
    parts = []
    for p in range(pages):
        parts.append(
            f"<PAGE><BLOCK>text[56, 95, 294, 167]<FORK>Page {p} 头部区域\n")
        for i in range(12):
            parts.append(
                f"<BLOCK>text[56, {200 + 40 * i}, 400, {230 + 40 * i}]"
                f"<FORK>第{p}页第{i}行，内容各不相同甲乙丙丁。\n")
    return "".join(parts)


class TestV2BlockFormat(unittest.TestCase):
    """The <PAGE><BLOCK>label[bbox]<FORK> documents of Unlimited-OCR-2.

    Regression tests for the confirmed false negatives of the V1-only
    signals: bbox commas split the segment signal inside the coordinates
    (max consecutive masked run 2) and no <|det|> blocks exist, so the
    V2-native variable-tail loop below was previously undetectable.
    """

    def test_v2_variable_tail_bbox_march_detected(self):
        self.assertTrue(is_looping_output(_v2_loop(120)))

    def test_v2_concat_bbox_march_detected(self):
        # no separators between blocks at all
        self.assertTrue(is_looping_output(_v2_loop(120, sep="")))

    def test_v2_exact_repeat_detected(self):
        text = ("<BLOCK>text[20, 100, 640, 136]<FORK>" + UNIT_CN) * 120
        self.assertTrue(is_looping_output(text))

    def test_v2_page_marker_loop_detected(self):
        text = "".join(
            f"<PAGE><BLOCK>text[20, {100 + 38 * i}, 640, {136 + 38 * i}]"
            f"<FORK>{UNIT_CN}"
            for i in range(40))
        self.assertTrue(is_looping_output(text))

    def test_v2_loop_below_threshold_not_detected(self):
        # 29 long-content units exceed min_chars but stay below the
        # calibrated 30-repeat threshold (genuine normal documents never
        # exceeded 2 consecutive normalized units)
        body = "各不相同的较长正文内容，用于跨过最小长度门槛。" * 3
        text = "\n".join(
            f"<BLOCK>text[20, {100 + 38 * i}, 640, {136 + 38 * i}]<FORK>{body}"
            for i in range(29))
        self.assertGreater(len(text), 2000)
        self.assertFalse(is_looping_output(text))

    def test_v2_loop_at_threshold_detected(self):
        body = "各不相同的较长正文内容，用于跨过最小长度门槛。" * 3
        text = "\n".join(
            f"<BLOCK>text[20, {100 + 38 * i}, 640, {136 + 38 * i}]<FORK>{body}"
            for i in range(30))
        self.assertTrue(is_looping_output(text))

    def test_genuine_v2_normal_document_not_detected(self):
        self.assertFalse(is_looping_output(_v2_normal_doc(pages=6)))

    def test_genuine_v2_evidence_sample_not_detected(self):
        # byte-shape of a genuine gfx1151 output (english test card):
        # consecutive same-label blocks with different content and bboxes
        sample = ("<PAGE><BLOCK>text[59, 99, 354, 156]<FORK>ROCm OCR Test\n"
                  "<BLOCK>text[59, 240, 442, 301]<FORK>Hello from Windows.\n"
                  "<BLOCK>text[59, 383, 290, 443]<FORK>Invoice 2026\n"
                  "<BLOCK>text[59, 525, 278, 583]<FORK>Total 123.45\n")
        self.assertFalse(is_looping_output(sample * 12))  # repeated card

    def test_v2_counter_loop_with_separators_detected(self):
        # a counter inside <FORK> content is segment-signal territory once
        # separators exist; the block signal alone would not see it because
        # the content differs on every iteration
        text = ("<BLOCK>text[20, 100, 640, 136]<FORK>"
                + "，".join(f"{y}年出生" for y in range(2500, 2900)))
        self.assertTrue(is_looping_output(text))

    def test_v2_content_counter_without_separators_is_known_limitation(self):
        # content varies per iteration AND no separators: documented blind
        # spot (same class as the V1 no-separator limitation)
        text = "".join(
            f"<BLOCK>text[20, {100 + 38 * i}, 640, {136 + 38 * i}]"
            f"<FORK>{y}年出生"
            for i, y in enumerate(range(2500, 2900)))
        self.assertFalse(is_looping_output(text))

    def test_v2_malformed_bbox_no_crash(self):
        bad = "<BLOCK>text[1,2\n" * 200 + "<BLOCK>text[1,2,3]<FORK>甲" * 100
        self.assertIsInstance(is_looping_output(bad), bool)

    def test_v2_onset_points_at_block_start(self):
        head = _v2_normal_doc(pages=2)
        text = head + _v2_loop(60)
        onset = looping_onset(text)
        self.assertIsNotNone(onset)
        self.assertEqual(text[onset:onset + 7], "<BLOCK>")

    def test_v2_truncation_keeps_head_and_adds_marker(self):
        head = _v2_normal_doc(pages=2)
        text = head + _v2_loop(60)
        out = truncate_looping_output(text)
        self.assertIn(TRUNCATION_MARKER, out)
        self.assertTrue(out.startswith(head))
        self.assertLess(len(out), len(text))

    def test_mixed_document_prefers_det_block_onset(self):
        # a V2-format head followed by a V1-style det loop (the shape used
        # in the infer integration tests): the det onset must win
        head = _v2_normal_doc(pages=2)
        text = head + _det_loop(60)
        onset = looping_onset(text)
        self.assertIsNotNone(onset)
        self.assertEqual(text[onset:onset + 7], "<|det|>")


class TestCommandLineInterface(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.normal_file = self.root / "normal.txt"
        self.normal_text = _varied_head(150)
        self.normal_file.write_text(self.normal_text, encoding="utf-8")
        self.loop_file = self.root / "loop.txt"
        self.loop_text = _varied_head(140) + _det_loop(60)
        self.loop_file.write_text(self.loop_text, encoding="utf-8")

    def test_report_mode_detects_and_keeps_files(self):
        status = main([str(self.loop_file), str(self.normal_file)])
        self.assertEqual(status, 1)
        self.assertEqual(
            self.loop_file.read_text(encoding="utf-8"), self.loop_text
        )

    def test_write_mode_truncates_only_looping_files(self):
        status = main(["--write", str(self.loop_file), str(self.normal_file)])
        self.assertEqual(status, 0)
        bounded = self.loop_file.read_text(encoding="utf-8")
        self.assertIn(TRUNCATION_MARKER, bounded)
        self.assertLess(len(bounded), len(self.loop_text))
        self.assertEqual(
            self.normal_file.read_text(encoding="utf-8"), self.normal_text
        )

    def test_missing_file_is_an_io_error_not_a_crash(self):
        status = main([str(self.root / "does-not-exist.txt")])
        self.assertEqual(status, 2)


if __name__ == "__main__":
    unittest.main()
