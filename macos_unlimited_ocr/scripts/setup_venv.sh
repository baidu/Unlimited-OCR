#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "${PYTHON_BIN}" ]]; then
  if command -v python3.12 >/dev/null 2>&1; then
    PYTHON_BIN="python3.12"
  elif command -v python3.11 >/dev/null 2>&1; then
    PYTHON_BIN="python3.11"
  else
    echo "Python 3.11 or 3.12 is required. Set PYTHON_BIN=/path/to/python." >&2
    exit 1
  fi
fi

"${PYTHON_BIN}" -m venv "${PROJECT_DIR}/.venv"
"${PROJECT_DIR}/.venv/bin/python" -m pip install --upgrade pip "setuptools<82" wheel
"${PROJECT_DIR}/.venv/bin/python" -m pip install -r "${PROJECT_DIR}/requirements.txt"

cat <<EOF
Virtual environment ready:
  ${PROJECT_DIR}/.venv

Activate it with:
  source "${PROJECT_DIR}/.venv/bin/activate"
EOF
