from macos_unlimited_ocr.output import normalize_result_markdown, strip_markdown_images, write_text_only_result


def test_strip_markdown_images_removes_standalone_image_links():
    markdown = "Title\n![](images/0.jpg)\n\nPrice 1.00\n![](images/1.jpg)\n"

    assert strip_markdown_images(markdown) == "Title\n\nPrice 1.00\n"


def test_strip_markdown_images_removes_truncated_image_detection_lines():
    markdown = "Title\n<|det|>image [\n\n\nPrice\n"

    assert strip_markdown_images(markdown) == "Title\n\nPrice\n"


def test_write_text_only_result_creates_sibling_file(tmp_path):
    result = tmp_path / "result.md"
    result.write_text("Title\n![](images/0.jpg)\nPrice\n", encoding="utf-8")

    text_result = write_text_only_result(result)

    assert text_result == tmp_path / "result_text.md"
    assert text_result.read_text(encoding="utf-8") == "Title\nPrice\n"


def test_normalize_result_markdown_converts_simple_html_table():
    markdown = (
        "Table 3\n"
        "<table><tr><td>Metric Pages</td><td>2</td><td>5</td></tr>"
        "<tr><td>Distinct-20 ↑</td><td>99.76%</td><td>99.78%</td></tr>"
        "<tr><td>Edit Distance ↓</td><td>0.0362</td><td>0.0452</td></tr></table>\n"
        "After\n"
    )

    assert normalize_result_markdown(markdown) == (
        "Table 3\n"
        "| Metric Pages | 2 | 5 |\n"
        "| --- | --- | --- |\n"
        "| Distinct-20 ↑ | 99.76% | 99.78% |\n"
        "| Edit Distance ↓ | 0.0362 | 0.0452 |\n"
        "After\n"
    )


def test_write_text_only_result_normalizes_tables_before_stripping_images(tmp_path):
    result = tmp_path / "result.md"
    original = (
        "<table><tr><td>A</td><td>B</td></tr><tr><td>1</td><td>2</td></tr></table>\n"
        "![](images/0.jpg)\n"
    )
    result.write_text(original, encoding="utf-8")

    text_result = write_text_only_result(result)

    assert result.read_text(encoding="utf-8") == original
    assert text_result.read_text(encoding="utf-8") == (
        "| A | B |\n"
        "| --- | --- |\n"
        "| 1 | 2 |\n"
    )
