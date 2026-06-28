#!/usr/bin/env bash
# Unlimited-OCR environment setup script.
#
# Usage:
#   source install.sh          # creates .venv and installs deps
#   source .venv/bin/activate  # activate afterwards
#
# Prerequisites:
#   - Python 3.12
#   - CUDA 12.9 toolkit and drivers
#   - uv (https://docs.astral.sh/uv/getting-started/installation/)

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$APP_DIR" || return 1

echo "==> Creating Python 3.12 virtualenv with uv ..."
uv venv --python 3.12 || return 1

echo "==> Installing the custom SGLang wheel (required — standard pip install sglang WILL NOT WORK) ..."
uv pip install "wheel/sglang-0.0.0.dev11416+g92e8bb79e-py3-none-any.whl" || return 1

echo "==> Installing pinned dependencies ..."
uv pip install -r requirements.txt || return 1

echo ""
echo "Done. Activate the environment with:  source .venv/bin/activate"
