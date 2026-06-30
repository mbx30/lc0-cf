#!/usr/bin/env bash
# 4sight engine bring-up (documented, NOT a CI gate).
#
# Wires up the real OPTIMAL + HUMAN axes on a GPU box:
#   1. init the maia3 submodule + install Maia-3 (scripts/setup-maia3.sh)
#   2. build Lc0 (meson/ninja -> build/release/lc0)
#   3. point you at the Leela-CF + Maia-3 nets
#   4. write resolved paths into foursight/config/settings.toml
#
# The Phase-0 CPU gate does NOT need any of this — the registry falls back to
# the dependency-free MockEngine. See CHESSFORMER.md for the upstream quick-start.
set -euo pipefail

FOURSIGHT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${FOURSIGHT_DIR}/.." && pwd)"
SETTINGS="${FOURSIGHT_DIR}/config/settings.toml"
EXAMPLE="${FOURSIGHT_DIR}/config/settings.example.toml"

echo "==> 4sight engine bring-up"
echo "    repo root: ${REPO_ROOT}"
echo "    foursight: ${FOURSIGHT_DIR}"

# --- 1. Maia-3 (HUMAN axis) -------------------------------------------------
echo "==> [1/4] Maia-3 submodule + install"
if [[ -f "${REPO_ROOT}/scripts/setup-maia3.sh" ]]; then
  ( cd "${REPO_ROOT}" && git submodule update --init --recursive maia3 || true )
  bash "${REPO_ROOT}/scripts/setup-maia3.sh"
else
  echo "    NOTE: ${REPO_ROOT}/scripts/setup-maia3.sh not found."
  echo "          This branch may predate the maia3 integration (c/integrate-maia3-2cf5)."
  echo "          Skipping Maia-3 install; the HUMAN axis will use the mock."
fi

# --- 2. Lc0 + Leela-CF (OPTIMAL axis) ---------------------------------------
echo "==> [2/4] Build Lc0 (engine strength)"
if [[ -f "${REPO_ROOT}/build.sh" ]]; then
  echo "    Building with gcc (clang can fail with 'cannot find -lstdc++')."
  ( cd "${REPO_ROOT}" && CC=gcc CXX=g++ ./build.sh ) || {
    echo "    WARNING: Lc0 build failed; install CUDA/cuDNN/ONNX and retry." >&2
  }
else
  echo "    NOTE: build.sh not found; cannot build Lc0 here."
fi

LC0_BIN="${REPO_ROOT}/build/release/lc0"
if [[ -x "${LC0_BIN}" ]]; then
  echo "    Lc0 binary: ${LC0_BIN}"
else
  echo "    Lc0 binary not present at ${LC0_BIN} (build did not complete)."
fi

# --- 3. Nets ----------------------------------------------------------------
echo "==> [3/4] Networks"
echo "    Leela-CF strength net (.pb.gz): https://huggingface.co/LeelaChessZero"
echo "    Maia-3 weights (.pt) ship with the maia3-* commands (run: maia3-cache --model maia3-5m)."
NETS_DIR="${REPO_ROOT}/nets"
mkdir -p "${NETS_DIR}"
echo "    Place the Leela-CF net at: ${NETS_DIR}/leela-cf.pb.gz"

# --- 4. settings.toml -------------------------------------------------------
echo "==> [4/4] Writing ${SETTINGS}"
if [[ -f "${SETTINGS}" ]]; then
  echo "    settings.toml already exists; leaving it untouched."
else
  cp "${EXAMPLE}" "${SETTINGS}"
  echo "    Created from settings.example.toml. Edit paths/device as needed."
fi

cat <<EOF

==> Done. Verify the live swap with:
      export FOURSIGHT_LIVE=1
      cd ${FOURSIGHT_DIR} && uv run foursight engines doctor
      uv run pytest tests/test_engines_live.py
EOF
