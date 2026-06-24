# macOS Unlimited OCR Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an isolated macOS-friendly Unlimited-OCR runner that uses a project-local Python virtual environment and attempts real Hugging Face Transformers inference on Apple Silicon MPS or CPU.

**Architecture:** The original repository stays untouched. The `macos_unlimited_ocr/` subproject contains a small installable Python package, CLI, tests, and setup scripts. Runtime code separates environment checks, image/PDF preparation, model loading, and inference orchestration.

**Tech Stack:** Python 3.11/3.12 preferred, PyTorch, Transformers, Pillow, PyMuPDF, pytest.

---

## Chunk 1: Local Utilities and CLI

### Task 1: Device and Input Helpers

**Files:**
- Create: `macos_unlimited_ocr/src/macos_unlimited_ocr/runtime.py`
- Create: `macos_unlimited_ocr/tests/test_runtime.py`

- [ ] Write tests for virtual environment detection, device selection, image collection, MIME detection, and PDF conversion.
- [ ] Run tests and confirm they fail because implementation is missing.
- [ ] Implement minimal helper functions.
- [ ] Run tests and confirm they pass.

### Task 2: Transformers OCR CLI

**Files:**
- Create: `macos_unlimited_ocr/src/macos_unlimited_ocr/cli.py`
- Create: `macos_unlimited_ocr/src/macos_unlimited_ocr/model.py`
- Create: `macos_unlimited_ocr/tests/test_cli.py`

- [ ] Write tests for argument parsing and venv guard behavior.
- [ ] Run tests and confirm they fail because implementation is missing.
- [ ] Implement CLI and model wrapper with MPS/CPU selection.
- [ ] Run tests and confirm they pass.

## Chunk 2: Packaging and Docs

### Task 3: Project Shell

**Files:**
- Create: `macos_unlimited_ocr/pyproject.toml`
- Create: `macos_unlimited_ocr/requirements.txt`
- Create: `macos_unlimited_ocr/scripts/setup_venv.sh`
- Create: `macos_unlimited_ocr/README.md`

- [ ] Add install metadata and pinned dependency ranges.
- [ ] Add setup script that creates `.venv` inside this subproject only.
- [ ] Document privacy, model download, hardware expectations, and commands.
- [ ] Run package tests in the venv.

## Chunk 3: Real Smoke Test

### Task 4: Local Sample and Inference Attempt

**Files:**
- Create: `macos_unlimited_ocr/examples/`

- [ ] Generate a tiny local OCR sample image without network.
- [ ] Run unit tests.
- [ ] If dependencies and model download are available, run one real inference with `baidu/Unlimited-OCR`.
- [ ] Record observed behavior and limitations.
