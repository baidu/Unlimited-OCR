#!/usr/bin/env bash
# 启动 MCP server / Launch the MCP server
#
# MCP-compatible clients (Claude Code, OpenAI Codex CLI, etc.) will spawn this
# process automatically when configured. To test manually:
#   bash scripts/mcp.sh          # blocks, waiting for MCP client connections via stdio
#
# To register with Claude Code, add to ~/.claude.json or project .mcp.json:
#   {
#     "mcpServers": {
#       "unlimited-ocr": {
#         "command": "python3",
#         "args": ["/abs/path/to/unlimited-ocr-mac/mcp_server/server.py"]
#       }
#     }
#   }
#
# To register with OpenAI Codex CLI (~/.codex/config.toml):
#   [mcp_servers.unlimited-ocr]
#   command = "python3"
#   args = ["/abs/path/to/unlimited-ocr-mac/mcp_server/server.py"]

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

cd "$REPO_DIR"
exec python3 mcp_server/server.py
