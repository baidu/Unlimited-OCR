#!/usr/bin/env python3
"""
Generate a sample test image for OCR testing.

Usage:
    python3 scripts/generate_test_image.py tests/sample.png
"""
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


def generate(out_path: Path) -> None:
    img = Image.new("RGB", (1024, 1024), "white")
    d = ImageDraw.Draw(img)
    try:
        fontb = ImageFont.truetype("/System/Library/Fonts/Hiragino Sans GB.ttc", 48)
        font = ImageFont.truetype("/System/Library/Fonts/Hiragino Sans GB.ttc", 32)
        font_small = ImageFont.truetype("/System/Library/Fonts/Hiragino Sans GB.ttc", 24)
    except OSError:
        # Fallback for Linux/Windows
        font = ImageFont.load_default()
        fontb = font
        font_small = font
    d.text((40, 40), "Unlimited OCR Test Image", fill="black", font=fontb)
    lines = [
        "Baidu released Unlimited-OCR on 2026-06-22.",
        "Based on Deepseek V2 architecture with MoE routing.",
        "Total ~6.7B parameters, ~2.5B activated.",
        "",
        "Main features:",
        "- One-shot long-horizon parsing",
        "- Grounding OCR (bbox + text output)",
        "- Multi-page PDF support",
    ]
    y = 120
    for ln in lines:
        d.text((40, y), ln, fill="black", font=font_small)
        y += 36
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    print(f"saved: {out_path}  ({img.size})")


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("tests/sample.png")
    generate(out)
