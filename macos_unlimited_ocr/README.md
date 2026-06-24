# macOS Unlimited OCR

Apple Silicon macOS runner for `baidu/Unlimited-OCR` using Hugging Face Transformers instead of the original CUDA/SGLang path.

## Privacy

Inputs are processed locally by this CLI. The first real model run downloads model weights and model Python files from Hugging Face or a compatible cache. `trust_remote_code=True` means that downloaded model code is executed locally; it does not mean your PDF or image is uploaded for OCR.

## Setup

Use only the project virtual environment:

```bash
cd macos_unlimited_ocr
./scripts/setup_venv.sh
source .venv/bin/activate
```

## Quick Tests

```bash
pytest -q
```

Create a local sample image:

```bash
python scripts/make_sample_image.py
```

## Real Inference

```bash
macos-unlimited-ocr --image path/to/image.png --output-dir outputs
macos-unlimited-ocr --pdf path/to/document.pdf --output-dir outputs --max-pages 1
```

By default the CLI uses `mps` when available and falls back to `cpu`. Use `--device cpu` to force CPU.

By default, this CLI preserves the local upstream `infer.py` behavior as closely as possible:

- `--image-mode gundam`
- `--ngram-window 128`
- `--max-length 32768`
- `--prompt "<image>document parsing."`

Use explicit flags if you want to deviate from upstream behavior, for example `--image-mode base`, `--ngram-window 1024`, or a smaller `--max-length`.

Each page output directory contains:

- `result.md`: upstream markdown with text and extracted image links
- `result_with_boxes.jpg`: visual detection overlay

Pass `--write-text-result` to additionally create `result_text.md`, a sidecar file with extracted image links removed and simple HTML tables converted to Markdown tables. This does not modify `result.md`.

## Hardware Notes

The original project targets NVIDIA CUDA. This macOS runner uses the Transformers API and may be slower. Some remote model operations may not be supported by Apple MPS; use `--device cpu` if MPS fails.

For the first run, expect a large model download. Keep the terminal inside this directory's `.venv`; the CLI refuses to run from other Python environments by default.

## Verified On

Tested on:

- macOS 26.5.1 arm64
- Python 3.12 virtual environment at `macos_unlimited_ocr/.venv`
- PyTorch 2.12.1
- Transformers 4.57.1

Commands run:

```bash
pytest -q
python scripts/make_sample_image.py
macos-unlimited-ocr --image examples/sample_receipt.png --output-dir outputs/smoke_cpu --device cpu
macos-unlimited-ocr --pdf examples/sample_invoice.pdf --output-dir outputs/smoke_pdf_cpu --max-pages 1 --device cpu
```

Observed sample image output:

```text
Unlimited OCR macOS smoke test
Invoice No: MAC-2026-0623
Date: 2026-06-23
Item Qty Price
Notebook 2 12.50
Coffee 1 4.20
Total 29.20
```

Observed sample PDF output:

```text
Unlimited OCR PDF smoke test
Order: PDF-2026-0623
Total: 42.00
```

## Known Limitations

- The upstream model's `infer()` implementation assumes CUDA. This project applies a local compatibility shim during inference so `.cuda()` calls run on the selected local device.
- PDF processing is page-by-page, matching the repository's batch `infer.py` style. The upstream `infer_multi()` multi-page joint parsing path is not yet reimplemented.
- On the tested machine, PyTorch was built with MPS support but reported `torch.backends.mps.is_available() == False`, so the verified real runs used CPU.
- CPU inference works for small smoke tests, but it is slow compared with NVIDIA GPU inference.
- The first real run downloads several GB of model files from Hugging Face.
- The upstream model code emits warnings about `position_ids`, attention masks, and `torch_dtype`; the smoke outputs were still correct for the local samples.
