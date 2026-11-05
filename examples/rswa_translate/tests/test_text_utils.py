"""segment_text 的纯逻辑单测（无需 torch / GPU / 权重）。"""

from text_utils import segment_text


def test_empty_returns_empty():
    assert segment_text("") == []


def test_whitespace_only_returns_empty():
    assert segment_text("   \n  \n\t ") == []


def test_single_short_block():
    assert segment_text("hello world", max_chars=512) == ["hello world"]


def test_det_blocks_merged_when_short():
    txt = "a <|det|>x</|det|> b <|det|>y</|det|>"
    segs = segment_text(txt)
    assert len(segs) == 1
    assert "<|det|>" in segs[0] and "</|det|>" in segs[0]


def test_blank_blocks_dropped_between_det():
    txt = "<|det|>keep</|det|>\n\n   \n<|det|>also</|det|>"
    segs = segment_text(txt)
    assert len(segs) == 1
    assert "keep" in segs[0] and "also" in segs[0]


def test_long_block_split_by_lines():
    # 单块超过 max_chars，应按行继续切，且每行非空才保留。
    txt = "\n".join(f"line{i}" for i in range(300))  # 每行 ~6 字符
    segs = segment_text(txt, max_chars=100)
    assert segs, "不应返回空"
    assert all(s.strip() for s in segs)
    # 每行都 <= max_chars，所以每段也应 <= max_chars（除单行长于上限的行）。
    assert all(len(s) <= 100 for s in segs if len(s.splitlines()[0]) <= 100)


def test_max_chars_respected_for_short_lines():
    txt = "\n".join(f"short segment {i}" for i in range(200))
    segs = segment_text(txt, max_chars=120)
    assert all(len(s) <= 120 for s in segs)


def test_keeps_order():
    txt = "first <|det|>A</|det|> second <|det|>B</|det|> third"
    segs = segment_text(txt)
    joined = "".join(segs)
    assert joined.index("first") < joined.index("second") < joined.index("third")
