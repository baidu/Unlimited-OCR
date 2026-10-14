"""OCR→翻译示例的纯文本工具（无 torch 依赖，便于单测）。

把 ``segment_text`` 从 ``ocr_translate.py`` 抽到这里，避免测试时被迫 import
torch / transformers；同时让分段逻辑成为可独立验证的纯函数。
"""

from __future__ import annotations

import re

# Unlimited-OCR 的版式块分隔符：<det>...</det>
_DET_BLOCK_RE = re.compile(r"(<\|det\|>.*?<\|/det\|>)", flags=re.DOTALL)


def segment_text(text: str, max_chars: int = 512) -> list[str]:
    """按 ``<det>`` 区块切分，过长单块再按行切。

    参数
    ----
    text: 待切分的带版式文本（可能含 ``<|det|>...</|det|>`` 块）。
    max_chars: 单段最大字符数；超过则按行继续切。

    返回
    ----
    非空片段列表，顺序与原文一致。
    """
    blocks = _DET_BLOCK_RE.split(text)
    segs: list[str] = []
    buf = ""
    for blk in blocks:
        if not blk.strip():
            continue
        if len(buf) + len(blk) <= max_chars:
            buf += blk
        else:
            if buf:
                segs.append(buf)
            buf = blk
    if buf:
        segs.append(buf)

    out: list[str] = []
    for s in segs:
        if len(s) <= max_chars:
            out.append(s)
        else:
            for line in s.splitlines() or [s]:
                if line.strip():
                    out.append(line)
    return [s for s in out if s.strip()]


def build_translate_prompt(segment: str, tgt_lang: str = "Chinese") -> str:
    """构建单段翻译 prompt（纯函数，无 torch 依赖，便于单测）。

    参数
    ----
    segment: 待翻译的源文片段。
    tgt_lang: 目标语言名（如 ``Chinese`` / ``Japanese``）。

    返回
    ----
    可直接喂给翻译模型的 prompt 字符串。
    """
    return f"Translate the following text to {tgt_lang}:\n{segment}\n\nTranslation:"
