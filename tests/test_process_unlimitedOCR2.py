"""Downstream contract tests: process_unlimitedOCR2.convert on loop-guard
output (issue #55).

The guard annotates truncated raw outputs with an explicit marker; the
evaluation Markdown must never inherit that annotation (it would pollute
text-comparison scoring and glue onto the last truncated cell).  These tests
run CPU-only, no GPU, no network.
"""

import importlib.util
import unittest
from pathlib import Path

from repetition_guard import TRUNCATION_MARKER, truncate_looping_output

_spec = importlib.util.spec_from_file_location(
    "process_unlimitedOCR2",
    Path(__file__).resolve().parents[1] / "utils" / "process_unlimitedOCR2.py")
process_unlimitedOCR2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(process_unlimitedOCR2)
convert = process_unlimitedOCR2.convert

UNIT_CN = "畜牧兽医产业基地建设工程"

DOC_2P = (
    "<PAGE><BLOCK>title<LEVEL1>[347, 135, 653, 155]<FORK>Annual Report\n"
    "<BLOCK>text[201, 162, 797, 180]<FORK>Revenue grew year over year.\n"
    "<BLOCK>image[100, 300, 400, 500]<FORK>\n"
    "<PAGE><BLOCK>text[201, 162, 797, 180]<FORK>Second page body text.\n"
)


def _v2_loop(iterations=120):
    return "".join(
        f"<BLOCK>text[20, {100 + 38 * i}, 640, {136 + 38 * i}]"
        f"<FORK>{UNIT_CN}\n"
        for i in range(iterations))


class LoopGuardMarkerTests(unittest.TestCase):
    def test_marker_literal_stays_in_sync_with_repetition_guard(self):
        # drift guard: the converter strips the exact literal the guard
        # appends; if TRUNCATION_MARKER ever changes, this fails first
        pattern = process_unlimitedOCR2.LOOP_GUARD_MARKER_RE.pattern
        self.assertIn(TRUNCATION_MARKER, pattern)

    def test_marker_is_stripped_from_evaluation_markdown(self):
        raw = truncate_looping_output(DOC_2P + _v2_loop())
        self.assertIn(TRUNCATION_MARKER, raw)  # raw keeps the annotation
        md = convert(raw)
        self.assertNotIn("loop guard", md)
        self.assertNotIn("<!--", md)

    def test_marker_never_glues_onto_a_truncated_table_cell(self):
        raw = ("<PAGE><BLOCK>table[41, 118, 959, 636]<FORK>"
               "<table><tr><td>Item</td><td>Qty</td></tr>"
               "<tr><td>Widget</td><td>12" + TRUNCATION_MARKER)
        md = convert(raw)
        self.assertNotIn("loop guard", md)
        self.assertIn("Widget", md)

    def test_document_without_marker_converts_unchanged(self):
        self.assertEqual(convert(DOC_2P), convert(DOC_2P + ""))
        md = convert(DOC_2P)
        self.assertIn("Annual Report", md)
        self.assertIn("page1_100_300_400_500.jpg", md)  # page numbering intact

    def test_truncated_document_keeps_page_numbering_of_surviving_pages(self):
        # loop starts on page 2: the guard cuts there; page-1 content (and
        # its page-1 image coordinates) must survive with numbering intact
        raw = truncate_looping_output(DOC_2P + _v2_loop())
        self.assertEqual(raw.count("<PAGE>"), 2)  # both markers before onset
        md = convert(raw)
        self.assertIn("page1_100_300_400_500.jpg", md)
        self.assertIn("Annual Report", md)

    def test_blank_trailing_page_does_not_shift_numbering(self):
        # a dangling <PAGE> left by the cut becomes a blank page; the
        # converter keeps blank pages so numbers do not shift
        md = convert(DOC_2P + "<PAGE>")
        self.assertIn("page1_100_300_400_500.jpg", md)
        self.assertIn("Second page body text.", md)


if __name__ == "__main__":
    unittest.main()
