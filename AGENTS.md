# Agent Development Guide

Instructions for cloud agents and contributors working on this Chessformer Lc0 fork.

Lc0 (Leela Chess Zero) is a single C++20 product: a UCI-compliant chess engine
that evaluates positions with a neural network. It is a CLI app driven over
stdin/stdout (UCI protocol), not a web/service app. Build system is Meson +
Ninja (`meson.build`, `meson_options.txt`); see `README.md`/`CONTRIBUTING.md`
for the canonical build docs.

## Repository Layout

- **Root** — Lc0 C++ engine (Meson build). Supports attention-body / Smolgen networks for Chessformer engine weights.
- **`maia3/`** — Human move prediction (git submodule). Python package with UCI entry points.
- **`CHESSFORMER.md`** — Integration overview, quick start, and architecture notes.

## Build / test / run Lc0

System dependencies (Debian/Ubuntu):

```bash
apt-get install -y libopenblas-dev pkg-config ninja-build
pip3 install --break-system-packages meson ninja
```

Build:

```bash
CC=gcc CXX=g++ ./build.sh
```

- IMPORTANT: the default `c++`/`cc` is clang and the build FAILS with
  `cannot find -lstdc++` (no clang libstdc++ in this image). Always set
  `CC=gcc CXX=g++`. g++ 13 satisfies the C++20 requirement.
- Binary: `build/release/lc0`
- Incremental dev build (after first `build.sh`): `ninja -C build/release lc0`.

Run tests:

```bash
ninja -C build/release test
```

Hello-world UCI smoke test (CPU backend):

```bash
printf 'uci\nposition startpos\ngo nodes 200\n'; sleep 8; printf 'quit\n' | \
  ./build/release/lc0 --weights=build/release/weights_test.pb.gz --backend=blas
```

`lc0` needs a network weights file to evaluate. Send `quit` only after the search
finishes (input is read on a separate thread, so an immediate `quit` aborts the
search).

## Install Maia-3

```bash
git submodule update --init --recursive
./scripts/setup-maia3.sh
```

Requires Python 3.10+, PyTorch, python-chess, and huggingface-hub. The script installs the vendored `maia3/` submodule in editable mode.

Verify:

```bash
maia3-uci --list-models
python -c "import importlib.metadata; print(importlib.metadata.version('maia3'))"
```

Pre-cache a model (avoids GUI timeouts on first launch):

```bash
maia3-cache --model maia3-5m
```

## Common Tasks

| Task | Command |
| --- | --- |
| Update maia3 submodule | `git submodule update --remote maia3 && pip install ./maia3` |
| Describe a network file | `./build/release/describenet --weights=file.pb.gz` |
| Run Maia-3 UCI engine | `maia3-5m` |
| Run Lc0 with Chessformer weights | `./build/release/lc0 --weights=BT4.pb.gz --backend=blas` |

## Branch Naming

Feature branches: `c/<descriptive-name>-2cf5`

Base branch for PRs: `cf-integration`

## Gotchas

1. Always use `CC=gcc CXX=g++` for Lc0 builds on this VM.
2. `meson` is installed in `~/.local/bin`. `build.sh` adds that to PATH itself,
   but direct `meson`/`ninja` commands need `PATH="$PATH:$HOME/.local/bin"` if
   meson isn't found (ninja from apt is already on PATH at `/usr/bin/ninja`).
3. CPU backend is `blas` (OpenBLAS, installed via `libopenblas-dev`); there is no
   GPU in this environment. `eigen`/`trivial`/`random` backends are also built.
4. A test network is already present at `build/release/weights_test.pb.gz`
   (256x20 legacy net, ~55 MB). It is not committed; download more from
   <https://lczero.org/play/networks/bestnets/> if needed. Eval on CPU/BLAS is
   slow (single-digit nps), so use small `go nodes N` for quick checks.
5. Meson auto-fetches subprojects (abseil, protobuf, zlib, gtest, eigen) from git
   during the first `meson setup`; this needs network access.
6. Maia-3 and Lc0 are separate UCI engines — do not expect Maia-3 `.pt` weights to load in Lc0 without conversion.
7. Maia-3 is for human modeling (`go nodes 1`); Lc0 Chessformer nets use full search for strength.
8. The `build/` directory is gitignored; do not commit build artifacts or downloaded weights.
