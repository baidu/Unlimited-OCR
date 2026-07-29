"""Local single-process inference with Hugging Face Transformers.

The SGLang path in ``infer.py`` requires CUDA. This script runs the model
directly through ``transformers`` on CUDA, Apple Silicon (MPS), or CPU, one
image at a time. It expects a local snapshot that has been prepared with
``patch_model_for_local.py`` (which removes the hardcoded CUDA calls and fixes
a silent image-embedding corruption on MPS; see that script's docstring).

Usage:
    hf download baidu/Unlimited-OCR --local-dir ./Unlimited-OCR-local
    python patch_model_for_local.py ./Unlimited-OCR-local
    python infer_transformers.py --model_dir ./Unlimited-OCR-local \
        --image_dir ./my_pages --output_dir ./outputs

Two input modes mirror ``infer.py``: ``--image_dir`` sends every image found
under a directory, ``--pdf`` converts each page of a PDF first.
"""

import argparse
import os
import pathlib
import tempfile
import time

PROMPT = "<image>document parsing."
NO_REPEAT_NGRAM_SIZE = 35
NGRAM_WINDOW = 128
PDF_DPI = 300
IMAGE_MODES = {
    # matches the gundam / base presets used by the SGLang path
    "gundam": {"base_size": 1024, "image_size": 640, "crop_mode": True},
    "base": {"base_size": 1024, "image_size": 1024, "crop_mode": False},
}


def pick_device(requested: str) -> str:
    import torch

    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def check_patched(model_dir: str) -> None:
    modeling = pathlib.Path(model_dir) / "modeling_unlimitedocr.py"
    if modeling.exists() and ".cuda()" in modeling.read_text(encoding="utf-8"):
        raise SystemExit(
            f"{modeling} still contains hardcoded CUDA calls. Run\n"
            f"    python patch_model_for_local.py {model_dir}\n"
            "first (required for MPS/CPU; a no-op change on CUDA)."
        )


def pdf_to_images(pdf_path: str, dpi: int = PDF_DPI) -> list:
    import fitz

    doc = fitz.open(pdf_path)
    tmp_dir = tempfile.mkdtemp(prefix="pdf_ocr_")
    image_paths = []
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    for i, page in enumerate(doc):
        out_path = os.path.join(tmp_dir, f"page_{i + 1:04d}.png")
        page.get_pixmap(matrix=mat).save(out_path)
        image_paths.append(out_path)
    doc.close()
    return image_paths


def collect_images(image_dir: str) -> list:
    exts = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
    image_files = []
    for root, _, files in os.walk(image_dir):
        for name in files:
            if name.lower().endswith(exts):
                image_files.append(os.path.join(root, name))
    return sorted(image_files)


def load_model(model_dir: str, device: str):
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_dir,
        trust_remote_code=True,
        use_safetensors=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    )
    return tokenizer, model.eval().to(device)


def run(args) -> None:
    device = pick_device(args.device)
    check_patched(args.model_dir)
    if device == "mps":
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    if args.pdf:
        images = pdf_to_images(args.pdf)
    elif args.image_dir:
        images = collect_images(args.image_dir)
    else:
        raise SystemExit("either --image_dir or --pdf is required")
    if not images:
        raise SystemExit("no input images found")

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"device={device}, images={len(images)}, image_mode={args.image_mode}")

    tokenizer, model = load_model(args.model_dir, device)
    mode = IMAGE_MODES[args.image_mode]

    import torch

    total_tokens = 0
    wall_start = time.time()
    for i, image_path in enumerate(images, 1):
        name = os.path.splitext(os.path.basename(image_path))[0]
        t0 = time.time()
        with torch.no_grad():
            text = model.infer(
                tokenizer,
                prompt=PROMPT,
                image_file=image_path,
                output_path=args.output_dir,
                base_size=mode["base_size"],
                image_size=mode["image_size"],
                crop_mode=mode["crop_mode"],
                no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
                ngram_window=NGRAM_WINDOW,
                max_length=args.max_length,
                save_results=False,
                eval_mode=True,
            )
        elapsed = time.time() - t0
        n_tokens = len(tokenizer(text, add_special_tokens=False)["input_ids"])
        total_tokens += n_tokens
        out_file = os.path.join(args.output_dir, f"{name}.md")
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"  [{i}/{len(images)}] {name}: {n_tokens} tokens, "
              f"{elapsed:.1f}s ({n_tokens / max(elapsed, 1e-6):.1f} tok/s)")

    wall = time.time() - wall_start
    print(f"done: {len(images)} image(s), {total_tokens} tokens, {wall:.1f}s")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Local Transformers inference (CUDA / Apple Silicon / CPU).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model_dir", required=True,
                        help="Local snapshot prepared by patch_model_for_local.py")
    parser.add_argument("--image_dir", default="", help="Directory of images")
    parser.add_argument("--pdf", default="", help="PDF file to convert and parse")
    parser.add_argument("--output_dir", default="./outputs")
    parser.add_argument("--image_mode", choices=tuple(IMAGE_MODES), default="gundam")
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"),
                        default="auto")
    parser.add_argument("--max_length", type=int, default=8192)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
