#!/usr/bin/env python3
"""
MCP (Model Context Protocol) server for Unlimited OCR.

Exposes 3 tools to any MCP-compatible client (Claude Code, OpenAI Codex CLI, etc.):
  - ocr_image(path)            → JSON of detections in one image
  - ocr_directory(path)        → batch OCR all images in a directory
  - model_status()             → check if model is loaded and what device

Usage:
  # First install + download model:
  bash scripts/install.sh

  # Run the MCP server (will be invoked automatically by MCP clients):
  python3 mcp_server/server.py
  # or
  bash scripts/mcp.sh

  # Add to Claude Code / Codex config (claude_desktop_config.json or ~/.codex/config.toml):
  # { "mcpServers": { "unlimited-ocr": { "command": "python3",
  #                                       "args": ["/path/to/mcp_server/server.py"] } } }
"""
from __future__ import annotations
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

# Make run_mac.py importable
REPO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_DIR))

# FastMCP is the standard server framework
try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    print("[mcp] ERROR: mcp package not installed. Run: pip install mcp", file=sys.stderr)
    sys.exit(1)

# Lazy import the heavy model
_model = None
_tokenizer = None
_device = None
_model_load_time: Optional[float] = None


def _ensure_model():
    """Lazy-load the model on first use. Takes ~5s on M4 Pro after first download."""
    global _model, _tokenizer, _device, _model_load_time
    if _model is not None:
        return

    from run_mac import pick_device, apply_patches, load_model

    _device = pick_device()
    print(f"[mcp] loading model on {_device} ...", file=sys.stderr)
    t0 = time.time()
    patched_dir = apply_patches(_device)
    _model, _tokenizer = load_model(patched_dir, _device, dtype=None)  # default bf16
    _model_load_time = time.time() - t0
    print(f"[mcp] model ready in {_model_load_time:.1f}s", file=sys.stderr)


def _parse_ocr_output(raw: str) -> list[dict]:
    """Parse the model's raw <|det|>...<|/det|>text string into structured JSON."""
    import re
    detections = []
    pattern = re.compile(
        r'<\|det\|>(\w+)\s*\[([^\]]+)\]<\|/det\|>(.*?)(?=<\|det\|>|\\Z)',
        re.DOTALL,
    )
    for m in pattern.finditer(raw):
        det_type = m.group(1)
        bbox_str = m.group(2).strip()
        text = m.group(3).strip()
        try:
            coords = [int(float(x)) for x in re.split(r'[,\s]+', bbox_str) if x]
        except Exception:
            coords = []
        detections.append({
            "type": det_type,
            "bbox": coords,
            "text": text,
        })
    return detections


# === MCP Server Definition ===

mcp = FastMCP("unlimited-ocr")


@mcp.tool()
def ocr_image(
    path: str,
    prompt: str = "<image>document parsing.",
    base_size: int = 1024,
    image_size: int = 640,
    crop_mode: bool = True,
    max_length: int = 8192,
) -> dict:
    """
    Run Unlimited OCR on a single image file.

    Returns:
        {
          "path": "/abs/path/to/image.png",
          "elapsed_seconds": 18.4,
          "raw_output": "<|det|>title [...]...<|/det|>...",  # raw model output
          "detections": [                                # parsed structured
            {"type": "title", "bbox": [x1,y1,x2,y2], "text": "..."},
            ...
          ]
        }
    """
    _ensure_model()
    p = Path(path).expanduser().resolve()
    if not p.exists():
        return {"error": f"file not found: {p}"}
    if not p.is_file():
        return {"error": f"not a file: {p}"}

    out_dir = REPO_DIR / "mcp_output"
    out_dir.mkdir(exist_ok=True)
    t0 = time.time()
    raw = _model.infer(
        _tokenizer,
        prompt=prompt,
        image_file=str(p),
        output_path=str(out_dir),
        base_size=base_size,
        image_size=image_size,
        crop_mode=crop_mode,
        eval_mode=True,
        max_length=max_length,
        no_repeat_ngram_size=35,
        ngram_window=128,
        temperature=0.0,
        save_results=False,
    )
    elapsed = time.time() - t0
    return {
        "path": str(p),
        "elapsed_seconds": round(elapsed, 2),
        "raw_output": raw,
        "detections": _parse_ocr_output(raw),
    }


@mcp.tool()
def ocr_directory(
    path: str,
    output_dir: Optional[str] = None,
    max_length: int = 8192,
) -> dict:
    """
    Run Unlimited OCR on all images in a directory.

    Supported extensions: .png .jpg .jpeg .webp .bmp

    Returns:
        {
          "input_dir": "...",
          "image_count": 5,
          "results": [ {"path": ..., "elapsed_seconds": ..., "detections": [...], "raw_output": ...}, ... ]
        }
    """
    _ensure_model()
    in_dir = Path(path).expanduser().resolve()
    if not in_dir.is_dir():
        return {"error": f"not a directory: {in_dir}"}

    out_dir = Path(output_dir).expanduser().resolve() if output_dir else (REPO_DIR / "mcp_output" / in_dir.name)
    out_dir.mkdir(parents=True, exist_ok=True)

    exts = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
    images = sorted(
        p for p in in_dir.iterdir()
        if p.is_file() and p.suffix.lower() in exts
    )
    results = []
    for img in images:
        try:
            r = ocr_image(
                path=str(img),
                max_length=max_length,
            )
            # Save .md
            md_path = out_dir / f"{img.stem}.md"
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(f"# OCR: {img.name}\n\n```\n{r.get('raw_output', '')}\n```\n")
            r["saved_to"] = str(md_path)
            results.append(r)
        except Exception as e:
            results.append({"path": str(img), "error": str(e)})

    return {
        "input_dir": str(in_dir),
        "output_dir": str(out_dir),
        "image_count": len(images),
        "results": results,
    }


@mcp.tool()
def model_status() -> dict:
    """Check if the model is loaded and on which device."""
    return {
        "loaded": _model is not None,
        "device": str(_device) if _device else None,
        "load_time_seconds": round(_model_load_time, 2) if _model_load_time else None,
    }


if __name__ == "__main__":
    mcp.run()
