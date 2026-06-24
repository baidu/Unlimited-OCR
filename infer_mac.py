"""
Run Unlimited-OCR on a PDF on macOS (Apple Silicon, MPS or CPU).

Usage:
    python infer_mac.py <pdf_path> [--device mps|cpu] [--dtype fp32|bf16]
"""
import argparse
import os
import sys
import tempfile
import time

import fitz  # PyMuPDF
import torch
from transformers import AutoModel, AutoTokenizer

MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "model_local"))
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "outputs"))


def pdf_to_images(pdf_path: str, dpi: int = 300) -> list[str]:
    doc = fitz.open(pdf_path)
    tmp_dir = tempfile.mkdtemp(prefix="ocr_mac_")
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    paths = []
    for i, page in enumerate(doc):
        out = os.path.join(tmp_dir, f"page_{i + 1:04d}.png")
        page.get_pixmap(matrix=mat).save(out)
        paths.append(out)
    doc.close()
    return paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", help="Path to the PDF to OCR")
    parser.add_argument("--device", default="auto", choices=["auto", "mps", "cpu"])
    parser.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"],
                        help="fp32 is safer on MPS; bf16 halves memory but may have numerical issues.")
    parser.add_argument("--model_dir", default=MODEL_DIR,
                        help="Path to a Mac-patched model directory (see prepare_mac_model.py).")
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    else:
        device = torch.device(args.device)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32

    print(f"[init] device={device}, dtype={dtype}, model_dir={args.model_dir}")
    print(f"[init] pdf={args.pdf}")

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.model_dir,
        trust_remote_code=True,
        use_safetensors=True,
        torch_dtype=dtype,
    )
    model = model.eval().to(device)
    print(f"[load] {time.time() - t0:.1f}s  model.dtype={model.dtype}  device={next(model.parameters()).device}")

    img_paths = pdf_to_images(args.pdf, dpi=args.dpi)
    print(f"[pdf]  {len(img_paths)} page(s) rendered at {args.dpi} DPI")

    os.makedirs(args.output_dir, exist_ok=True)

    t1 = time.time()
    if len(img_paths) == 1:
        print("[ocr]  single-page → infer() with gundam config")
        model.infer(
            tokenizer,
            prompt="<image>document parsing.",
            image_file=img_paths[0],
            output_path=args.output_dir,
            base_size=1024, image_size=640, crop_mode=True,
            max_length=32768,
            no_repeat_ngram_size=35, ngram_window=128,
            save_results=True,
        )
    else:
        print("[ocr]  multi-page → infer_multi() with base config")
        model.infer_multi(
            tokenizer,
            prompt="<image>Multi page parsing.",
            image_files=img_paths,
            output_path=args.output_dir,
            image_size=1024,
            max_length=32768,
            no_repeat_ngram_size=35, ngram_window=1024,
            save_results=True,
        )
    print(f"\n[done] OCR took {time.time() - t1:.1f}s")

    md = os.path.join(args.output_dir, "result.md")
    if os.path.exists(md):
        print(f"[out]  markdown: {md}")
        print("=" * 60)
        with open(md, encoding="utf-8") as f:
            sys.stdout.write(f.read())
        print()
        print("=" * 60)


if __name__ == "__main__":
    main()
