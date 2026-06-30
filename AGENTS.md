# Agent Development Guide

Instructions for cloud agents and contributors working on this Chessformer Lc0 fork.

## Repository Layout

- **Root** — Lc0 C++ engine (Meson build). Supports attention-body / Smolgen networks for Chessformer engine weights.
- **`maia3/`** — Human move prediction (git submodule). Python package with UCI entry points.
- **`CHESSFORMER.md`** — Integration overview, quick start, and architecture notes.

## Build Lc0

System dependencies (Debian/Ubuntu):

```bash
apt-get install -y libopenblas-dev pkg-config ninja-build
pip3 install --break-system-packages meson ninja
```

**Important:** Build with gcc, not the default clang (clang fails linking `-lstdc++`):

```bash
CC=gcc CXX=g++ ./build.sh
```

Binary: `build/release/lc0`

Run tests:

```bash
ninja -C build/release test
```

Hello-world UCI smoke test (CPU backend, small network):

```bash
# Download a small test network if needed, then:
printf 'uci\nisready\nposition startpos\ngo nodes 1\nquit\n' | ./build/release/lc0 --backend=blas --weights=/path/to/network.pb.gz
```

## Install Maia-3

```bash
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
2. Maia-3 and Lc0 are separate UCI engines — do not expect Maia-3 `.pt` weights to load in Lc0 without conversion.
3. Maia-3 is for human modeling (`go nodes 1`); Lc0 Chessformer nets use full search for strength.
4. The `build/` directory is gitignored; do not commit build artifacts or downloaded weights.
