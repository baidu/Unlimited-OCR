from __future__ import annotations

from html import unescape
from html.parser import HTMLParser
import re
from pathlib import Path

IMAGE_LINK_RE = re.compile(r"^\s*!\[[^\]]*\]\([^)]+\)\s*$")
IMAGE_DETECTION_RE = re.compile(r"^\s*<\|det\|>image(?:\s|\[|$)")
TOO_MANY_BLANK_LINES_RE = re.compile(r"\n{3,}")
HTML_TABLE_RE = re.compile(r"<table>.*?</table>", re.IGNORECASE | re.DOTALL)


class _TableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[str]] = []
        self._current_row: list[str] | None = None
        self._current_cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag.lower() == "tr":
            self._current_row = []
        elif tag.lower() in {"td", "th"} and self._current_row is not None:
            self._current_cell = []

    def handle_data(self, data: str) -> None:
        if self._current_cell is not None:
            self._current_cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"td", "th"} and self._current_row is not None and self._current_cell is not None:
            cell = " ".join("".join(self._current_cell).split())
            self._current_row.append(unescape(cell))
            self._current_cell = None
        elif tag == "tr" and self._current_row is not None:
            if self._current_row:
                self.rows.append(self._current_row)
            self._current_row = None


def _html_table_to_markdown(html_table: str) -> str:
    parser = _TableParser()
    parser.feed(html_table)
    rows = parser.rows
    if not rows:
        return html_table

    width = max(len(row) for row in rows)
    padded = [row + [""] * (width - len(row)) for row in rows]
    header = padded[0]
    body = padded[1:]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * width) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in body)
    return "\n".join(lines)


def normalize_result_markdown(markdown: str) -> str:
    return HTML_TABLE_RE.sub(lambda match: _html_table_to_markdown(match.group(0)), markdown)


def strip_markdown_images(markdown: str) -> str:
    lines = [
        line
        for line in markdown.splitlines()
        if not IMAGE_LINK_RE.match(line) and not IMAGE_DETECTION_RE.match(line)
    ]
    text = TOO_MANY_BLANK_LINES_RE.sub("\n\n", "\n".join(lines).strip())
    return text + "\n"


def write_text_only_result(result_path: Path) -> Path | None:
    if not result_path.exists():
        return None
    normalized = normalize_result_markdown(result_path.read_text(encoding="utf-8"))
    text = strip_markdown_images(normalized)
    text_path = result_path.with_name("result_text.md")
    text_path.write_text(text, encoding="utf-8")
    return text_path
