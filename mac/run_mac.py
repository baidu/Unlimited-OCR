#!/usr/bin/env python3
"""
Mac (Apple Silicon) launcher for Unlimited-OCR.

Unlimited-OCR 的 modeling_unlimitedocr.py 里有 13+ 处 `.cuda()` 硬编码和
`torch.autocast("cuda", ...)`,官方只支持 NVIDIA GPU。在 Mac 上跑要做:

1. 把 .cuda() 替换成 `.to(target_device)`  (target_device = mps | cpu)
2. 把 `torch.autocast("cuda", ...)` 换成 `torch.autocast(device_type=..., device=...)`
3. 模型加载到 MPS(Apple GPU 后端)而不是 CUDA

策略: 在加载模型前, 就地对 `mac_model/modeling_unlimitedocr.py` 和
`mac_model/deepencoder.py` 做 search/replace,生成 mac_patched_*.py,然后把
原始的 .py 备份覆盖 → 加载。这样 trust_remote_code=True 拿到的就是 patch 过的代码。

⚠️ 这是个 work-around,不是上游支持。如果哪天 Baidu 改 infer() 把 device 参数化,
  就可以丢这个 patch 了。

用法:
  source "/Volumes/Mac mini/文稿 · 云端/Model/Unlimited OCR/.venv-ocr/bin/activate"
  python3 run_mac.py --image /path/to/test.png
  python3 run_mac.py --image_dir /path/to/images/ --output_dir /path/to/out/
"""
from __future__ import annotations
import argparse
import os
import re
import shutil
import sys
import time
from pathlib import Path

import torch

REPO_DIR = Path(__file__).resolve().parent
# Source of patches (in this repo). install.sh copies them over the downloaded HF model.
PATCHES_DIR = REPO_DIR / "patches"
# Where the patched model lives. install.sh sets this up at <repo>/model_dir
MODEL_DIR = REPO_DIR / "model_dir"
HF_HUB_CACHE = Path(
    os.environ.get(
        "HF_HUB_CACHE",
        str(Path.home() / ".cache" / "huggingface" / "hub"),
    )
)


def pick_device() -> torch.device:
    """Pick best available device. MPS > CPU. CUDA never works on macOS."""
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")


def apply_patches(target_device: torch.device) -> Path:
    """
    Copy MODEL_DIR (HF model + applied patches/) → PATCHED_DIR, rewriting .cuda() → .to(<device>).
    Idempotent: if PATCHED_DIR exists with mtime >= MODEL_DIR, reuse.

    install.sh should have populated MODEL_DIR by:
      1. downloading baidu/Unlimited-OCR from HuggingFace
      2. overlaying our `patches/` (the .py source files with our Mac fixes)

    This function then does the final device-specific rewrite (.cuda → .to('mps')).
    Returns the patched dir to load with from_pretrained.
    """
    if not MODEL_DIR.exists():
        raise FileNotFoundError(
            f"{MODEL_DIR} not found. Run scripts/install.sh first to download + patch the model."
        )

    patched_dir = MODEL_DIR.parent / "model_dir_patched"
    if patched_dir.exists():
        src_mtime = max(
            (MODEL_DIR / f).stat().st_mtime
            for f in ("modeling_unlimitedocr.py", "deepencoder.py", "modeling_deepseekv2.py")
            if (MODEL_DIR / f).exists()
        )
        dst_mtime = (patched_dir / "modeling_unlimitedocr.py").stat().st_mtime
        if dst_mtime >= src_mtime:
            print(f"[patch] reusing {patched_dir}")
            return patched_dir
        shutil.rmtree(patched_dir)

    shutil.copytree(MODEL_DIR, patched_dir)
    device_str = str(target_device)  # 'mps' or 'cpu'
    autocast_device = device_str  # for any leftover autocast calls

    for fname in ("modeling_unlimitedocr.py", "modeling_deepseekv2.py", "deepencoder.py"):
        p = patched_dir / fname
        if not p.exists():
            continue
        src = p.read_text()
        orig = src

        # .cuda() (no-arg) → .to(<device>) — covers `tensor.cuda()` patterns
        src = re.sub(r"\.cuda\(\)", f".to('{device_str}')", src)
        # .cuda(non_blocking=True) etc. → .to(device, non_blocking=True)
        src = re.sub(
            r"\.cuda\(([^)]*)\)",
            lambda m: f".to('{device_str}'" + (f", {m.group(1)}" if m.group(1).strip() else "") + ")",
            src,
        )
        # torch.autocast("cuda", ...) → torch.autocast(device_type=..., ...)
        src = re.sub(
            r'torch\.autocast\("cuda"',
            f'torch.autocast(device_type="{autocast_device}"',
            src,
        )
        # torch.get_autocast_gpu_dtype() → torch.get_autocast_dtype(device.type)
        # PyTorch deprecated get_autocast_gpu_dtype(); the device-aware API is correct
        # for any device (CUDA, MPS, etc.), not just CUDA.
        src = re.sub(
            r"torch\.get_autocast_gpu_dtype\(\)",
            "torch.get_autocast_dtype(query_states.device.type)",
            src,
        )

        if src != orig:
            p.write_text(src)
            print(f"[patch] {fname}: rewrote .cuda → .to('{device_str}')")

    print(f"[patch] done → {patched_dir}")
    return patched_dir


def load_model(patched_dir: Path, target_device: torch.device, dtype: torch.dtype = torch.bfloat16):
    """Load AutoModel + AutoTokenizer from patched dir, move weights to device."""
    from transformers import AutoModel, AutoTokenizer

    print(f"[load] tokenizer from {patched_dir}")
    tokenizer = AutoTokenizer.from_pretrained(str(patched_dir), trust_remote_code=True)

    print(f"[load] model from {patched_dir} (this can take 30-60s)...")
    t0 = time.time()
    print(f"[load] using dtype={dtype}")
    model = AutoModel.from_pretrained(
        str(patched_dir),
        trust_remote_code=True,
        use_safetensors=True,
        torch_dtype=dtype,
    )
    print(f"[load] model loaded in {time.time()-t0:.1f}s, moving to {target_device}...")
    model = model.eval().to(target_device)
    print(f"[load] ready on {target_device}, dtype={dtype}")
    return model, tokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=str, help="single image to OCR")
    ap.add_argument("--image_dir", type=str, help="directory of images to OCR")
    ap.add_argument("--output_dir", type=str, default="./out", help="where to write .md")
    ap.add_argument("--prompt", type=str, default="<image>document parsing.")
    ap.add_argument("--cpu", action="store_true", help="force CPU even if MPS available")
    ap.add_argument("--base_size", type=int, default=1024)
    ap.add_argument("--image_size", type=int, default=640)
    ap.add_argument("--crop_mode", action="store_true", default=True)
    ap.add_argument("--max_length", type=int, default=8192, help="max tokens to generate (default 8192; 32768 is paper's setting)")
    ap.add_argument("--no_repeat_ngram_size", type=int, default=35, help="ngram size for repetition block (35 in README; 0 disables)")
    ap.add_argument("--ngram_window", type=int, default=128, help="sliding window for ngram block")
    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16", help="model dtype (bf16 default; fp16 may be more stable on MPS)")
    ap.add_argument("--temperature", type=float, default=0.0, help="0 = greedy; >0 enables sampling (e.g. 0.1 may help break MPS bf16 repetition loops)")
    ap.add_argument("--stream", action="store_true", help="stream output to terminal (NOTE: stream mode triggers a PyTorch MPS 'Placeholder storage' bug; use eval_mode for now)")
    args = ap.parse_args()

    if not (args.image or args.image_dir):
        ap.error("need --image or --image_dir")

    target_device = torch.device("cpu") if args.cpu else pick_device()
    print(f"[device] {target_device}")

    if target_device.type == "cpu":
        print("[!] WARNING: CPU inference is extremely slow for this 6.7B VLM.")
        print("[!] Expect minutes per image. Prefer MPS unless you have no choice.")

    patched_dir = apply_patches(target_device)
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    chosen_dtype = dtype_map.get(args.dtype, torch.bfloat16) if target_device.type in ("mps", "cuda") else torch.float32
    model, tokenizer = load_model(patched_dir, target_device, chosen_dtype)

    os.makedirs(args.output_dir, exist_ok=True)

    if args.image:
        images = [args.image]
    else:
        exts = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
        images = sorted(
            os.path.join(args.image_dir, f)
            for f in os.listdir(args.image_dir)
            if f.lower().endswith(exts)
        )

    print(f"[infer] {len(images)} image(s), base_size={args.base_size}, image_size={args.image_size}")
    for i, img in enumerate(images, 1):
        stem = Path(img).stem
        out = os.path.join(args.output_dir, f"{stem}.md")
        print(f"[infer] [{i}/{len(images)}] {img} → {out}")
        t0 = time.time()
        try:
            result = model.infer(
                tokenizer,
                prompt=args.prompt,
                image_file=img,
                output_path=args.output_dir,
                base_size=args.base_size,
                image_size=args.image_size,
                crop_mode=args.crop_mode,
                eval_mode=not args.stream,  # Mac-MPS workaround: stream mode hits PyTorch MPS bug
                max_length=args.max_length,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                ngram_window=args.ngram_window,
                temperature=args.temperature,
                save_results=False,  # Mac path: we write the .md ourselves from eval_mode string
            )
            # In eval_mode, the model returns a raw string (no .md written by the model code).
            # Write it ourselves.
            with open(out, 'w', encoding='utf-8') as f:
                f.write(f"# OCR result: {Path(img).name}\n\n```\n{result}\n```\n")
                print(f"  wrote {out}")
        except Exception as e:
            print(f"[infer] FAILED on {img}: {e}")
            import traceback; traceback.print_exc()
            continue
        print(f"[infer] [{i}/{len(images)}] done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
