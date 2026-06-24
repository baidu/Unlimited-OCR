from pathlib import Path

import pytest

from macos_unlimited_ocr.runtime import (
    collect_images,
    detect_image_mime,
    ensure_project_venv,
    pdf_to_images,
    select_torch_device,
)


def test_detect_image_mime_for_common_extensions():
    assert detect_image_mime(Path("scan.jpg")) == "image/jpeg"
    assert detect_image_mime(Path("scan.jpeg")) == "image/jpeg"
    assert detect_image_mime(Path("scan.png")) == "image/png"
    assert detect_image_mime(Path("scan.webp")) == "image/webp"


def test_collect_images_returns_supported_files_sorted(tmp_path):
    nested = tmp_path / "nested"
    nested.mkdir()
    (tmp_path / "b.txt").write_text("ignore", encoding="utf-8")
    (tmp_path / "a.png").write_bytes(b"png")
    (nested / "c.JPG").write_bytes(b"jpg")

    assert [p.name for p in collect_images(tmp_path)] == ["a.png", "c.JPG"]


def test_collect_images_rejects_missing_directory(tmp_path):
    with pytest.raises(FileNotFoundError):
        collect_images(tmp_path / "missing")


def test_select_torch_device_prefers_mps(monkeypatch):
    class FakeMps:
        @staticmethod
        def is_available():
            return True

    class FakeBackends:
        mps = FakeMps()

    class FakeTorch:
        backends = FakeBackends()

    assert select_torch_device(FakeTorch()) == "mps"


def test_select_torch_device_falls_back_to_cpu(monkeypatch):
    class FakeMps:
        @staticmethod
        def is_available():
            return False

    class FakeBackends:
        mps = FakeMps()

    class FakeTorch:
        backends = FakeBackends()

    assert select_torch_device(FakeTorch()) == "cpu"


def test_ensure_project_venv_accepts_expected_prefix(tmp_path):
    project_dir = tmp_path / "project"
    venv = project_dir / ".venv"
    venv.mkdir(parents=True)

    ensure_project_venv(project_dir, sys_prefix=str(venv))


def test_ensure_project_venv_rejects_other_prefix(tmp_path):
    project_dir = tmp_path / "project"
    project_dir.mkdir()

    with pytest.raises(RuntimeError, match="project virtual environment"):
        ensure_project_venv(project_dir, sys_prefix="/usr/bin/python")


def test_pdf_to_images_converts_pages(tmp_path):
    fitz = pytest.importorskip("fitz")
    pdf_path = tmp_path / "one-page.pdf"
    doc = fitz.open()
    page = doc.new_page(width=200, height=100)
    page.insert_text((20, 50), "Unlimited OCR macOS test")
    doc.save(pdf_path)
    doc.close()

    image_paths = pdf_to_images(pdf_path, tmp_path / "pages", dpi=72)

    assert len(image_paths) == 1
    assert image_paths[0].exists()
    assert image_paths[0].suffix == ".png"
