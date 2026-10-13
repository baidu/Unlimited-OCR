"""
Unit test for repetition penalty and ngram parameters in infer.py (Fixes #55).
"""

import sys
import unittest
from unittest.mock import MagicMock, patch

import infer


class TestInferRepetitionControl(unittest.TestCase):
    def test_parse_args_defaults(self):
        with patch.object(sys, "argv", ["infer.py"]):
            args = infer.parse_args()
            self.assertEqual(args.repetition_penalty, 1.0)
            self.assertEqual(args.no_repeat_ngram_size, 35)
            self.assertEqual(args.ngram_window, 128)

    def test_parse_args_custom_repetition(self):
        test_argv = [
            "infer.py",
            "--repetition_penalty",
            "1.1",
            "--no_repeat_ngram_size",
            "15",
            "--ngram_window",
            "256",
        ]
        with patch.object(sys, "argv", test_argv):
            args = infer.parse_args()
            self.assertEqual(args.repetition_penalty, 1.1)
            self.assertEqual(args.no_repeat_ngram_size, 15)
            self.assertEqual(args.ngram_window, 256)

    @patch("infer.requests.post")
    @patch("infer.get_ngram_processor_str", return_value="MockNGramProcessor")
    @patch("infer.build_content", return_value=[{"type": "text", "text": "test"}])
    def test_infer_one_payload_building(self, mock_build_content, mock_get_processor, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_lines.return_value = [b"data: [DONE]"]
        mock_post.return_value = mock_response

        args = MagicMock()
        args.image_mode = "gundam"
        args.repetition_penalty = 1.1
        args.no_repeat_ngram_size = 20
        args.ngram_window = 64

        infer.infer_one("dummy.png", None, args, 1)

        self.assertTrue(mock_post.called)
        _, kwargs = mock_post.call_args
        import json

        payload = json.loads(kwargs["data"])

        self.assertEqual(payload.get("repetition_penalty"), 1.1)
        self.assertEqual(payload.get("custom_logit_processor"), "MockNGramProcessor")
        self.assertEqual(payload["custom_params"]["ngram_size"], 20)
        self.assertEqual(payload["custom_params"]["window_size"], 64)


if __name__ == "__main__":
    unittest.main()
