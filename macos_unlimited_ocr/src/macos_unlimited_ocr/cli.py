from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .model import InferenceConfig, UnlimitedOCRModel
from .runtime import collect_images, ensure_project_venv, pdf_to_images


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run baidu/Unlimited-OCR locally on macOS via Transformers.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    inputs = parser.add_argument_group("inputs")
    inputs.add_argument("--image", type=Path, help="Single image file to process")
    inputs.add_argument("--image-dir", type=Path, help="Directory of images to process")
    inputs.add_argument("--pdf", type=Path, help="PDF file to convert and process")

    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--model-name", default="baidu/Unlimited-OCR")
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    parser.add_argument("--image-mode", choices=("gundam", "base"), default="gundam")
    parser.add_argument("--prompt", default="<image>document parsing.")
    parser.add_argument("--pdf-dpi", type=int, default=200)
    parser.add_argument("--max-pages", type=int, default=None)
    parser.add_argument("--ngram-window", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--write-text-result", action="store_true")
    parser.add_argument("--allow-outside-venv", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    selected = [value for value in (args.image, args.image_dir, args.pdf) if value is not None]
    if len(selected) != 1:
        raise SystemExit("Choose exactly one of --image, --image-dir, or --pdf.")

    for path in selected:
        if not path.exists():
            raise SystemExit(f"Input path does not exist: {path}")


def build_image_jobs(args: argparse.Namespace) -> list[Path]:
    if args.image:
        return [args.image]
    if args.image_dir:
        return collect_images(args.image_dir)

    page_dir = args.output_dir / "_pdf_pages"
    return pdf_to_images(args.pdf, page_dir, dpi=args.pdf_dpi, max_pages=args.max_pages)


def build_inference_config(args: argparse.Namespace) -> InferenceConfig:
    image_mode = args.image_mode
    image_size = 640 if image_mode == "gundam" else 1024
    ngram_window = args.ngram_window
    if ngram_window is None:
        ngram_window = 128
    max_length = args.max_length
    if max_length is None:
        max_length = 32768

    return InferenceConfig(
        model_name=args.model_name,
        device=args.device,
        prompt=args.prompt,
        image_mode=image_mode,
        output_dir=args.output_dir,
        image_size=image_size,
        max_length=max_length,
        ngram_window=ngram_window,
        write_text_result=args.write_text_result,
    )


def run(args: argparse.Namespace) -> int:
    validate_args(args)
    if not args.allow_outside_venv:
        ensure_project_venv()

    images = build_image_jobs(args)
    if not images:
        print("No images found.", file=sys.stderr)
        return 1

    config = build_inference_config(args)
    model = UnlimitedOCRModel(config)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Images: {len(images)}")
    print(f"Output: {args.output_dir}")
    for idx, image_path in enumerate(images, start=1):
        page_output = args.output_dir / image_path.stem
        print(f"[{idx}/{len(images)}] {image_path}")
        model.infer_image(image_path, page_output)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run(args)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
