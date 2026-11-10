#!/usr/bin/env bash
# 一键运行 Unlimited OCR / One-click run Unlimited OCR
#
# Usage:
#   bash scripts/run.sh /path/to/image.png                       # auto-detected as --image
#   bash scripts/run.sh --image_dir /path/to/dir/                # batch
#   bash scripts/run.sh /path/to/image.png --output_dir /tmp/out # mixed
#   bash scripts/run.sh --help                                   # all options
#
# If the first argument is a file (exists on disk), it's auto-converted to --image PATH.
# If it's a directory, it's auto-converted to --image_dir PATH.
# Otherwise, args are passed through unchanged.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$REPO_DIR/.venv-ocr"

if [[ ! -d "$VENV_DIR" ]]; then
    echo "[ERR] venv not found at $VENV_DIR"
    echo "[ERR] Run scripts/install.sh first."
    exit 1
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# Auto-convert a leading positional arg to --image / --image_dir
ARGS=()
if [[ $# -gt 0 ]]; then
    first="$1"
    if [[ "$first" != --* ]] && { [[ -f "$first" ]] || [[ -d "$first" ]]; }; then
        if [[ -f "$first" ]]; then
            ARGS=(--image "$first")
            shift
        elif [[ -d "$first" ]]; then
            ARGS=(--image_dir "$first")
            shift
        fi
    fi
fi

cd "$REPO_DIR"
exec python3 run_mac.py "${ARGS[@]}" "$@"
