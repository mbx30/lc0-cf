# 4sight (Maia v2)

**4sight predicts the most likely real-world outcome by lining up three things
and comparing them.** Chess is a direct prototype of the market product: the
three pairwise gaps it computes here are exactly the signals the market engine
will later compute.

| Axis | Chess proving ground (this repo) | Real-world / market analogue |
| --- | --- | --- |
| **OPTIMAL** | **Leela-CF** — Chessformer strength net in Lc0 (MCTS, native WDL value head), ~100 Elo > Stockfish | best-EV / "correct in hindsight" action |
| **HUMAN** | **Maia-3** — Elo-conditioned Chessformer human net (Python UCI) | behavioral model / crowd belief |
| **ACTUAL (IRL)** | the move actually played + the realized game result (from real PGNs) | realized real-world outcome |

Stockfish is replaced: **Leela-CF takes the OPTIMAL axis** (native WDL — no
centipawn→win% sigmoid) and **Maia-3 keeps the HUMAN axis**. Both engines stay,
organized side-by-side — comparing **optimal vs human vs actual** is the point.

The three pairwise gaps:

* **optimal vs human** (Leela vs Maia) → where humans are *predictably* suboptimal.
* **human vs actual** (Maia vs realized) → how well the human model predicts reality.
* **optimal vs actual** (Leela vs realized) → how often reality tracked the optimal.

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
    ingest/actual.py          ACTUAL axis from PGN / FEN+moves
    compare.py                ThreeWayComparison + pairwise gaps
    cli.py                    doctor / compare / replay
  tests/                      mock-based gate + opt-in live checks
```

See [`DEVELOPMENT_PLAN.md`](DEVELOPMENT_PLAN.md) for the phased roadmap.
