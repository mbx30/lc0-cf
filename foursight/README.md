# 4sight (Maia v2)

**4sight predicts the most likely real-world outcome by lining up up to four
sights and comparing them.** The name is deliberate: **OPTIMAL**, **HUMAN**,
**ACTUAL**, and **WORST** (worst possible outcome under adversarial
multi-actor coordination).

Chess is the **two-actor proving ground** and therefore runs **three sights
only** — WORST does not apply in zero-sum two-player games (the adversarial
worst case is already captured by OPTIMAL). The three pairwise gaps computed
here are the core signals the market engine will extend to six gaps when actor
count ≥ 3.

| Sight | Chess proving ground (this repo) | Real-world / market analogue |
| --- | --- | --- |
| **OPTIMAL** | **Leela-CF** — Chessformer strength net in Lc0 (MCTS, native WDL value head), ~100 Elo > Stockfish | best-EV / "correct in hindsight" action |
| **HUMAN** | **Maia-3** — Elo-conditioned Chessformer human net (Python UCI, history n=7) | behavioral model / crowd belief |
| **ACTUAL** | the move actually played + the realized game result (from real PGNs) | realized real-world outcome |
| **WORST** | *N/A for 2-actor chess* | worst coalition outcome (game theory, 3+ actors) |

Stockfish is replaced: **Leela-CF takes the OPTIMAL sight** (native WDL — no
centipawn→win% sigmoid) and **Maia-3 keeps the HUMAN sight**. Both engines stay,
organized side-by-side — comparing **optimal vs human vs actual** is the point.

The three chess pairwise gaps:

* **optimal vs human** (Leela vs Maia) → where humans are *predictably* suboptimal.
* **human vs actual** (Maia vs realized) → how well the human model predicts reality.
* **optimal vs actual** (Leela vs realized) → how often reality tracked the optimal.

See [`DEVELOPMENT_PLAN.md`](DEVELOPMENT_PLAN.md) for the full phased roadmap
(chess metrics → bulk ingest → serving → **market four-sight engine** with
Polymarket/Kalshi data tiers → logistics → finance → optional chess N-actor
extension last). Product vision:
[Maia v2 on Notion](https://app.notion.com/p/berrymichael/Maia-v2-38e9cb079ddb804f8843e182ffb7101c).

**Market data (Phase 4, sources TBD):** prediction-market ingest targets
**Lightweight** (outcomes + metadata, < 100 MB) plus **Medium** (15-minute price
history, laptop-scale) — not TB tick archives. Polymarket and Kalshi may be
combined into a synthesis; see the plan for candidate datasets and sizing.

## Quick start (CPU — no GPU, no nets)

The Phase-0 gate runs entirely on a built-in `MockEngine`, so the package is
green on CPU with nothing installed:

```bash
cd foursight
uv sync
FOURSIGHT_FORCE_MOCK=1 uv run pytest        # registry -> ingest -> three-way compare
uv run ruff check . && uv run mypy foursight
```

### Commands

```bash
# Which axis resolves to a real engine vs the mock?
uv run foursight engines doctor

# One three-way comparison for a position (mock by default):
uv run foursight compare "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1" \
    --elo 1500 --played c7c5 --result 1-0

# Per-move three-way stream over a real game:
uv run foursight replay game.pgn --elo auto

# Aggregate calibration metrics by Elo band and phase:
uv run foursight calibrate game.pgn --elo auto --json-out report.json
```

Each comparison is one structured `ThreeWayComparison` object (see
`foursight/compare.py`) with `optimal`, `human`, `actual`, and the three `gaps`.

## Live engines (GPU path — documented, not a CI gate)

```bash
foursight/scripts/setup_engines.sh    # init maia3 submodule, build Lc0, fetch nets
cp foursight/config/settings.example.toml foursight/config/settings.toml  # then edit
FOURSIGHT_LIVE=1 uv run pytest tests/test_engines_live.py
```

The registry transparently falls back to the mock whenever a binary or net is
missing, so partial setups still run. See the repo's `CHESSFORMER.md` for the
upstream Lc0 + Maia-3 quick-start.

## Layout

```
foursight/
  pyproject.toml              uv-managed; minimal deps
  config/settings.example.toml
  scripts/setup_engines.sh    GPU bring-up (not a gate)
  foursight/
    config.py                 layered TOML + env settings
    engines/
      base.py                 ChessEngine ABC + EngineResult/EngineMove
      leela_cf.py             OPTIMAL — Lc0 + Leela-CF, native WDL
      maia3.py                HUMAN   — Maia-3 UCI, SelfElo/OppoElo
      mock.py                 CPU gate — deterministic stand-in
      registry.py             builds {optimal, human}, mock fallback
    ingest/actual.py          ACTUAL sight from PGN / FEN+moves
    compare.py                ThreeWayComparison + pairwise gaps
    cli.py                    doctor / compare / replay
    market/                   Phase 4 — prediction-market four-sight engine
      records.py              MarketRecord + Parquet I/O
      ingest/                 Polymarket + Kalshi CSV adapters (Light/Medium)
      synthesis.py            multi-platform merge, resample, walk-forward
      trend_net.py            DOWN/FLAT/UP tiny net (numpy, CPU)
      compare.py              FourWayComparison + six gaps
    game_theory/              WORST sight — adversarial-coalition solver
  tests/                      mock-based gate + market/game_theory tests
```
