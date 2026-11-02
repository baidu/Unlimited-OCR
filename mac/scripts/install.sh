#!/usr/bin/env bash
# 一键安装 Unlimited OCR Mac 适配版 / One-click install for Unlimited OCR Mac adaptation
#
# Usage:
#   bash scripts/install.sh                    # install to ./model_dir (in repo)
#   bash scripts/install.sh /path/to/model     # install model to a custom path
#   HF_HUB_CACHE=/path bash scripts/install.sh # use custom HF cache (e.g., external SSD)
#
# This script will:
#   1. Check Python 3.12+ (3.13 recommended)
#   2. Create a venv in .venv-ocr/ and install dependencies
#   3. Download the baidu/Unlimited-OCR model (~6.7 GB bf16)
#   4. Apply the Mac-specific patches from patches/ into model_dir/

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="${1:-$REPO_DIR/model_dir}"
HF_HUB_CACHE="${HF_HUB_CACHE:-$HOME/.cache/huggingface/hub}"
PYTHON_BIN="${PYTHON_BIN:-python3.13}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
err()   { echo -e "${RED}[ERR]${NC} $*" >&2; }

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║   Unlimited OCR · Mac (Apple Silicon) Installer          ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# 1. Check Python
info "Checking Python ..."
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    if command -v python3.12 >/dev/null 2>&1; then
        PYTHON_BIN=python3.12
        warn "python3.13 not found, falling back to python3.12"
    elif command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN=python3
        warn "python3.13 not found, falling back to python3 (may not work)"
    else
        err "No Python 3 found. Install via: brew install python@3.13"
        exit 1
    fi
fi
PY_VERSION=$($PYTHON_BIN -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
info "Using $PYTHON_BIN (Python $PY_VERSION)"

# 2. Check macOS
if [[ "$(uname)" != "Darwin" ]]; then
    warn "Not macOS. MPS will not be available, will fall back to CPU (very slow)."
fi
ARCH=$(uname -m)
if [[ "$ARCH" != "arm64" ]]; then
    warn "Not Apple Silicon (arm64). MPS won't work, will fall back to CPU."
fi

# 3. Create venv
VENV_DIR="$REPO_DIR/.venv-ocr"
if [[ -d "$VENV_DIR" ]]; then
    info "Reusing existing venv: $VENV_DIR"
else
    info "Creating venv at $VENV_DIR ..."
    $PYTHON_BIN -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# 4. Install deps
info "Installing Python dependencies (this may take a few minutes) ..."
pip install --quiet --upgrade pip
pip install --quiet \
    "torch>=2.4" \
    torchvision \
    "transformers==4.57.1" \
    "huggingface_hub>=0.30" \
    Pillow matplotlib einops addict easydict pymupdf psutil \
    accelerate safetensors requests tqdm

# 5. Download model
info "Downloading baidu/Unlimited-OCR model (6.67 GB, this may take a while) ..."
mkdir -p "$HF_HUB_CACHE"
HF_HUB_CACHE="$HF_HUB_CACHE" python3 -c "
import os
from huggingface_hub import snapshot_download
p = snapshot_download(
    repo_id='baidu/Unlimited-OCR',
    cache_dir=os.environ['HF_HUB_CACHE'],
    allow_patterns=['*.json','*.py','*.safetensors','*.txt','*.md'],
    max_workers=4,
)
print('snapshot:', p)
"

# 6. Apply patches
info "Applying Mac-specific patches ..."
mkdir -p "$MODEL_DIR"

# Find the latest snapshot in HF cache
SNAPSHOT_DIR=$(ls -d "$HF_HUB_CACHE"/models--baidu--Unlimited-OCR/snapshots/*/ 2>/dev/null | head -1)
if [[ -z "$SNAPSHOT_DIR" || ! -d "$SNAPSHOT_DIR" ]]; then
    err "Could not find downloaded snapshot in $HF_HUB_CACHE"
    exit 1
fi

# Copy the model (with symlinks for big files)
for f in "$SNAPSHOT_DIR"/*; do
    name=$(basename "$f")
    target="$MODEL_DIR/$name"
    if [[ -e "$target" ]]; then
        continue
    fi
    if [[ "$name" == *.py ]]; then
        cp "$f" "$target"
    else
        ln -sf "$f" "$target"
    fi
done

# Overlay our Mac patches (overwrite the .py files + config.json)
for f in "$REPO_DIR"/patches/*; do
    name=$(basename "$f")
    cp "$f" "$MODEL_DIR/$name"
done

info "Model ready at: $MODEL_DIR"
info "HF cache:        $HF_HUB_CACHE"
info "Venv:            $VENV_DIR"
echo ""
echo "✓ Installation complete!"
echo ""
echo "Next:"
echo "  bash scripts/run.sh /path/to/your/image.png"
echo "  bash scripts/run.sh --image_dir /path/to/images/"
echo "  bash scripts/run.sh --help  # see all options"
echo ""
