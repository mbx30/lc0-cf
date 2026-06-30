#!/usr/bin/env bash
# Install the vendored Maia-3 Python package from the maia3 git submodule.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MAIA3_DIR="${ROOT}/maia3"

if [[ ! -f "${MAIA3_DIR}/pyproject.toml" ]]; then
  echo "Maia-3 submodule not found. Initializing..." >&2
  git -C "${ROOT}" submodule update --init --recursive maia3
fi

if [[ ! -f "${MAIA3_DIR}/pyproject.toml" ]]; then
  echo "Error: ${MAIA3_DIR}/pyproject.toml missing after submodule init." >&2
  exit 1
fi

echo "Installing Maia-3 from ${MAIA3_DIR}..."
python3 -m pip install --break-system-packages -e "${MAIA3_DIR}"

echo ""
echo "Maia-3 installed. Available commands:"
echo "  maia3-uci --list-models"
echo "  maia3-cache --model maia3-5m"
echo "  maia3-5m | maia3-23m | maia3-79m"
echo ""
echo "See CHESSFORMER.md and maia3/README.md for usage."
