# AGENTS.md

## Cursor Cloud specific instructions

Lc0 (Leela Chess Zero) is a single C++20 product: a UCI-compliant chess engine
that evaluates positions with a neural network. It is a CLI app driven over
stdin/stdout (UCI protocol), not a web/service app. Build system is Meson +
Ninja (`meson.build`, `meson_options.txt`); see `README.md`/`CONTRIBUTING.md`
for the canonical build docs.

### Build / test / run

- Build: `CC=gcc CXX=g++ ./build.sh` → binary at `build/release/lc0`.
  - IMPORTANT: the default `c++`/`cc` is clang and the build FAILS with
    `cannot find -lstdc++` (no clang libstdc++ in this image). Always set
    `CC=gcc CXX=g++`. g++ 13 satisfies the C++20 requirement.
- Incremental dev build (after first `build.sh`): `ninja -C build/release lc0`.
- Tests: `ninja -C build/release test` (gtest suites; ~8 suites, all should pass).
- Run / hello-world: `lc0` needs a network weights file to evaluate. Example:
  `printf 'uci\nposition startpos\ngo nodes 200\n'; sleep 8; printf 'quit\n'`
  piped into `./build/release/lc0 --weights=<net> --backend=blas` → prints
  `bestmove ...`. Send `quit` only after the search finishes (input is read on a
  separate thread, so an immediate `quit` aborts the search).

### Non-obvious gotchas

- `meson` is installed in `~/.local/bin`. `build.sh` adds that to PATH itself,
  but direct `meson`/`ninja` commands need `PATH="$PATH:$HOME/.local/bin"` if
  meson isn't found (ninja from apt is already on PATH at `/usr/bin/ninja`).
- CPU backend is `blas` (OpenBLAS, installed via `libopenblas-dev`); there is no
  GPU in this environment. `eigen`/`trivial`/`random` backends are also built.
- A test network is already present at `build/release/weights_test.pb.gz`
  (256x20 legacy net, ~55 MB). It is not committed; download more from
  <https://lczero.org/play/networks/bestnets/> if needed. Eval on CPU/BLAS is
  slow (single-digit nps), so use small `go nodes N` for quick checks.
- Meson auto-fetches subprojects (abseil, protobuf, zlib, gtest, eigen) from git
  during the first `meson setup`; this needs network access.
