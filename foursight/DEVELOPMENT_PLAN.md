# 4sight — Development Plan

4sight predicts the **most likely real-world outcome** by comparing three axes —
**OPTIMAL**, **HUMAN**, and **ACTUAL** — and measuring the gaps between them.
Chess is the proving ground: with the Chessformer integration in this repo
(`lc0-cf`) the two engines we need come from one place.

This package was migrated here from `stockfish-chess/maia-v2/` when `lc0-cf`
became the home repo, and revised to the three-axis model below. **Stockfish is
removed**; Leela-CF takes the OPTIMAL axis with a native WDL value head (no
centipawn→win% sigmoid).

## The three axes

| Axis | Chess | Real-world / market |
| --- | --- | --- |
| **OPTIMAL** | Leela-CF (Chessformer strength net in Lc0, MCTS, native WDL) | best-EV action |
| **HUMAN** | Maia-3 (Elo-conditioned Chessformer human net) | behavioral / crowd model |
| **ACTUAL** | move played + realized result, from real PGNs | realized outcome |

The three pairwise gaps are the product in miniature — the same signals the
market engine will later compute:

* **optimal vs human** → where humans are *predictably* suboptimal.
* **human vs actual** → how well the human model predicts reality.
* **optimal vs actual** → how often reality tracked the optimal.

## Key decisions

| # | Decision | Choice |
| --- | --- | --- |
| D1 | Keep both engines, side-by-side? | YES — Leela-CF (OPTIMAL) + Maia-3 (HUMAN) |
| D2 | Optimal axis | Leela-CF replaces Stockfish (native WDL) |
| D3 | Human axis | Maia-3 adopted (Elo-conditioned, history n=7) — not retrained |
| D4 | Third axis | ACTUAL = played move + result, first-class |
| D5 | Home repo | `mbx30/lc0-cf`; package `foursight` at repo root |
| D6 | Engine source | in-repo Lc0 (+Leela-CF net) and the `maia3` submodule; config can override |
| D7 | Naming / compute | package `foursight`; CPU-green gate via mock; GPU path scripted, not gated |

## Phase 0 — scaffold (this milestone)

A runnable `foursight` package that (1) exposes Leela-CF and Maia-3 as
symmetric, side-by-side UCI adapters, (2) ingests the ACTUAL axis from real
games, (3) emits a structured `ThreeWayComparison` per position, and (4) is
**CPU-green in CI today** via a mock engine, with a scripted path to the real
GPU engines. **No training in Phase 0.**

Delivered:

1. **Project setup** — `pyproject.toml` (uv), `config/settings.example.toml`,
   additive `.gitignore` entries.
2. **Engine layer** — `engines/{base,leela_cf,maia3,mock,registry}.py`; the
   registry builds the `{optimal, human}` pair and falls back to the mock when a
   binary/net is absent.
3. **ACTUAL ingest** — `ingest/actual.py` (PGN via `chess.pgn`, or FEN + move
   list) into per-position `ActualRecord`s.
4. **Comparison layer** — `compare.py`: `ThreeWayComparison` with agree@1,
   JS/KL divergence between optimal & human policies, and Δexpectation (optimal
   EV regret) for the human and actual moves.
5. **CLI** — `engines doctor`, `compare`, `replay`; a docstring-only seam toward
   a later `serving/` FastAPI endpoint (<100 ms target for the small nets).
6. **Engine bring-up** — `scripts/setup_engines.sh` wraps submodule init +
   `scripts/setup-maia3.sh`, builds Lc0, points at both nets, writes
   `settings.toml`. Documented, not a gate.
7. **Tests & CI** — mock-based gate (`test_engines_mock`, `test_compare`,
   `test_actual_ingest`) plus opt-in `test_engines_live`; path-filtered
   `.github/workflows/foursight-ci.yml` that never touches Lc0's C++ CI.

### Phase-0 verification

```bash
cd foursight
uv run ruff check . && uv run mypy foursight && uv run pytest   # green on CPU (mock)
uv run foursight engines doctor                                  # real vs mock per axis
uv run foursight compare "<FEN>" --elo 1500 --played e2e4        # three-way + gaps
uv run foursight replay game.pgn --elo auto                      # per-move stream
# optional GPU swap:
foursight/scripts/setup_engines.sh && FOURSIGHT_LIVE=1 uv run pytest tests/test_engines_live.py
```

## Explicitly deferred (not Phase 0)

Bulk data ingestion (Lichess / markets / FRED), the tiny trend net, the market
three-way engine, logistics & finance, any training, a web dashboard, and
deleting Stockfish source.

## Roadmap beyond Phase 0 (sketch)

1. **Calibration & metrics** — aggregate the per-position gaps over many games:
   human predictive accuracy, optimal-regret distributions by Elo band.
2. **Bulk ingest** — Lichess PGN/Parquet pipelines reusing the `ActualRecord`
   schema; the `to_dataframe`/`write_parquet` seam is already in place.
3. **Serving** — the `serving/` FastAPI endpoint behind a sub-100 ms target.
4. **Market engine** — port the same three-axis comparison to market data
   (optimal = best-EV, human = crowd belief, actual = realized).
