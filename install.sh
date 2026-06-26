#!/usr/bin/env bash
set -euo pipefail

# Unlimited-OCR environment setup script.
#
# Prerequisites:
#   - Python 3.12
#   - CUDA 12.9 toolkit and drivers
#   - uv (https://docs.astral.sh/uv/getting-started/installation/)
#
# Usage:
#   source install.sh          # creates .venv and installs deps
#   source .venv/bin/activate  # activate afterwards

APP_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$APP_DIR"

echo "==> Creating Python 3.12 virtualenv with uv ..."
uv venv --python 3.12

echo "==> Activating virtualenv ..."
# shellcheck disable=SC1091
source .venv/bin/activate

echo "==> Installing the custom SGLang wheel (required — standard pip install sglang WILL NOT WORK) ..."
uv pip install "wheel/sglang-0.0.0.dev11416+g92e8bb79e-py3-none-any.whl"

echo "==> Installing pinned dependencies ..."
uv pip install -r requirements.txt

echo ""
echo "Done. Activate the environment with:  source .venv/bin/activate"
