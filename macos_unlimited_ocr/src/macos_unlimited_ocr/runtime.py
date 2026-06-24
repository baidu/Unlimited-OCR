from __future__ import annotations

import base64
import os
import sys
import tempfile
from pathlib import Path
from typing import Iterable

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def ensure_project_venv(project_dir: Path | None = None, sys_prefix: str | None = None) -> None:
    root = (project_dir or project_root()).resolve()
    prefix = Path(sys_prefix or sys.prefix).resolve()
    expected = root / ".venv"
    if prefix != expected:
        raise RuntimeError(
            "This command must run inside the project virtual environment.\n"
            f"Expected: {expected}\n"
            f"Current:  {prefix}\n"
            f"Run: source {expected}/bin/activate"
        )


def select_torch_device(torch_module, requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    mps = getattr(getattr(torch_module, "backends", None), "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def detect_image_mime(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in (".jpg", ".jpeg"):
        return "image/jpeg"
    if ext == ".png":
        return "image/png"
    if ext == ".webp":
        return "image/webp"
    if ext == ".bmp":
        return "image/bmp"
    raise ValueError(f"Unsupported image extension: {path}")


def encode_image_data_url(path: Path) -> str:
    mime = detect_image_mime(path)
    data = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{data}"


def collect_images(image_dir: Path) -> list[Path]:
    if not image_dir.exists():
        raise FileNotFoundError(image_dir)
    if not image_dir.is_dir():
        raise NotADirectoryError(image_dir)

    paths: list[Path] = []
    for root, _, names in os.walk(image_dir):
        for name in names:
            path = Path(root) / name
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                paths.append(path)
    return sorted(paths, key=lambda p: str(p.relative_to(image_dir)).lower())


def pdf_to_images(pdf_path: Path, output_dir: Path | None = None, dpi: int = 200, max_pages: int | None = None) -> list[Path]:
    import fitz

    if not pdf_path.exists():
        raise FileNotFoundError(pdf_path)

    target_dir = output_dir or Path(tempfile.mkdtemp(prefix="macos_unlimited_ocr_pdf_"))
    target_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(pdf_path)
    try:
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pages: Iterable = doc
        if max_pages is not None:
            pages = [doc[i] for i in range(min(max_pages, doc.page_count))]

        image_paths: list[Path] = []
        for i, page in enumerate(pages):
            out_path = target_dir / f"page_{i + 1:04d}.png"
            page.get_pixmap(matrix=mat).save(out_path)
            image_paths.append(out_path)
        return image_paths
    finally:
        doc.close()
