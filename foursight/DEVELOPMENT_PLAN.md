# 4sight — Development Plan

4sight predicts the **most likely real-world outcome** by comparing up to four
**sights** — **OPTIMAL**, **HUMAN**, **ACTUAL**, and **WORST** — and measuring
the gaps between them. Chess is the proving ground: with the Chessformer
integration in this repo (`lc0-cf`) the two engines we need come from one place.

This package was migrated here from `stockfish-chess/maia-v2/` when `lc0-cf`
became the home repo. **Stockfish is removed**; Leela-CF takes the OPTIMAL axis
with a native WDL value head (no centipawn→win% sigmoid). The product vision
lives in the [Maia v2 Notion blueprint](https://app.notion.com/p/berrymichael/Maia-v2-38e9cb079ddb804f8843e182ffb7101c);
this document is the repo-local execution plan.

## The four sights

| Sight | Name | Chess (2 actors) | Real-world / market (3+ actors) |
| --- | --- | --- | --- |
| 1st | **OPTIMAL** | Leela-CF (Chessformer strength net in Lc0, MCTS, native WDL) | best-EV / hindsight-optimal action |
| 2nd | **HUMAN** | Maia-3 (Elo-conditioned Chessformer human net, history n=7) | behavioral / crowd model |
| 3rd | **ACTUAL** | move played + realized result, from real PGNs | realized outcome |
| 4th | **WORST** | **N/A** — see below | worst possible outcome for the focal actor |

### Why chess stays three-axis

Chess is **two-player, zero-sum, perfect-information, deterministic**. In that
regime the adversarial worst case is fully captured by the opponent playing
optimally — minimax collapses into the OPTIMAL sight viewed from the adversary's
side. A separate WORST sight adds no information, so the chess proving ground
implements **three sights only** (`ThreeWayComparison`, three pairwise gaps).

**WORST exists only when there are three or more actors.** Other participants can
coordinate against the focal actor in ways no single rational opponent models.
Evaluate on a game-theory basis:

```
WORST(a_i) = min over joint strategies of all other actors of E[u_i | a_i, a_{-i}]
```

(adversarial-coalition worst case — all non-focal actors coordinate to minimize
your payoff.)

| Actor count | Sights active | Pairwise gaps |
| --- | --- | --- |
| 2 (chess) | OPTIMAL, HUMAN, ACTUAL | 3 |
| 3+ (markets, logistics coalitions) | all four | 6 |

The six market gaps (when WORST applies):

* **optimal vs human** → where humans/crowd are *predictably* suboptimal.
* **human vs actual** → how well the behavioral model predicts reality.
* **optimal vs actual** → how often reality tracked the optimal.
* **worst vs optimal** → tail risk your best-EV action still carries.
* **worst vs human** → does crowd belief ignore adversarial coordination?
* **worst vs actual** → did reality land near the worst-case coalition outcome?

### Architectural layers (from Maia v2 blueprint)

The Notion blueprint frames the full product as three transferable layers on top
of chess engines:

| Layer | Role | Chess instance | Generalizes to |
| --- | --- | --- | --- |
| **B — Feature extraction** | UCI signals (WDL, policy, MultiPV, PV) → structured features | `engines/`, Leela-CF native WDL | any engine-backed evaluator as feature extractor |
| **C — Human / outcome prediction** | Supervised models for move-matching and outcome calibration | Maia-3 (57.1% move-match, 79M params) | crowd-belief / behavioral axis |
| **A — Generalization framework** | MDP + (search over model) + (learned eval); plan over distributions, not enumerated adversary trees | MCTS + value head in Lc0 | logistics MPC, market trend net |

**What transfers vs what breaks** (honest caveats from the blueprint):

* **Transfers:** MCTS, learned value functions, policy/value decomposition,
  search+eval → MPC/model-based RL, engine eval as a feature extractor.
* **Breaks:** minimax/alpha-beta (adversarial 2-player only); exact tree search
  over stochastic futures (finance); stationarity (markets drift); cheap perfect
  simulators (chess rules are free; market simulators approximate and compound
  error).

Recommended staging from the blueprint: **prototype Layer B on chess → Layer C
outcome models → generalize to logistics → finance last**, with ruthless
out-of-sample discipline in finance (walk-forward splits, Deflated Sharpe, PBO).

## Key decisions

| # | Decision | Choice |
| --- | --- | --- |
| D1 | Keep both engines, side-by-side? | YES — Leela-CF (OPTIMAL) + Maia-3 (HUMAN) |
| D2 | Optimal axis | Leela-CF replaces Stockfish (native WDL) |
| D3 | Human axis | Maia-3 adopted (Elo-conditioned, history n=7) — not retrained |
| D4 | Third sight | ACTUAL = played move + result, first-class |
| D5 | Fourth sight | WORST = adversarial-coalition worst outcome; **3+ actors only** |
| D6 | Chess scope | **Three-axis unchanged** — WORST is N/A for 2-actor chess |
| D7 | Home repo | `mbx30/lc0-cf`; package `foursight` at repo root |
| D8 | Engine source | in-repo Lc0 (+Leela-CF net) and the `maia3` submodule; config can override |
| D9 | Naming / compute | package `foursight`; CPU-green gate via mock; GPU path scripted, not gated |

## Phase 0 — scaffold (delivered)

A runnable `foursight` package that (1) exposes Leela-CF and Maia-3 as
symmetric, side-by-side UCI adapters, (2) ingests the ACTUAL sight from real
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
four-sight engine, logistics & finance domains, any training, a web dashboard,
and deleting Stockfish source.

## Roadmap beyond Phase 0

Phases are ordered so **chess stays three-axis through the middle of the
roadmap**. Extending chess itself to four sights would require modeling
multi-actor dynamics (e.g. Swiss-system fields, coaching teams, or synthetic
coalitions) and is **deferred to the very last phase** — it is not needed for
the market product and is substantially more complex than the current
`ThreeWayComparator`.

### Phase 1 — Calibration & metrics

Aggregate per-position gaps over many games:

* human predictive accuracy by Elo band
* optimal-regret distributions
* policy divergence (JS/KL) vs rating and phase

Reuse `ActualRecord` schema; bulk Lichess ingest can follow in Phase 2.

### Phase 2 — Bulk ingest (chess)

Lichess PGN/Parquet pipelines reusing `ActualRecord`; `to_dataframe` /
`write_parquet` seam. Target: millions of positions with engine features
(Layer B) for downstream calibration. Budget for heavy preprocessing (Maia team
reports multi-day PGN→tabular on large shards).

### Phase 3 — Serving

`serving/` FastAPI endpoint behind a sub-100 ms target for Maia-3 / small nets.
Single forward pass at inference (no search) for the HUMAN sight — same pattern
as Maia at one node.

### Phase 4 — Market engine (four sights, 3+ actors)

Port the comparison framework to market / business data where **actor count ≥
3** (e.g. you vs competitors vs market makers vs informed flow):

| Sight | Market instance |
| --- | --- |
| OPTIMAL | best-EV action (MPC / OR baseline where simulators exist) |
| HUMAN | crowd belief — tiny trend net or aggregated sentiment |
| ACTUAL | realized price / sales / demand outcome |
| WORST | adversarial-coalition tail outcome via `game_theory/` module |

**Tiny trend net** (from blueprint): state = rolling window of normalized
features (returns, vol, MA ratio); policy = DOWN / FLAT / UP buckets (mirrors
WDL framing); value = expected magnitude; **walk-forward chronological splits
only** — never shuffle time series. Architecture: small 1D-conv or MLP, tens of
thousands to low millions of parameters, CPU inference.

New code (sketch):

* `foursight/game_theory/` — `GameSpec`, `worst_outcome(focal, action, game)`
* `foursight/market/` — ingest, `FourWayComparison`, six pairwise gaps
* No change to chess `compare.py` in this phase

### Phase 5 — Logistics

Friendly generalization target: buildable simulators, near-deterministic
dynamics, strong OR-Tools baselines. Frame VRP / inventory / dispatch as MDPs;
use search+eval (MCTS or MPC rollouts over demand simulator). **Always benchmark
against OR-Tools** before shipping RL. Map sights:

* OPTIMAL = OR / MPC cost-to-go
* HUMAN = learned dispatch prior or historical operator behavior
* ACTUAL = realized cost / SLA
* WORST = coalition of suppliers/competitors minimizing your payoff (3+ actors)

### Phase 6 — Finance

Hardest generalization: non-stationarity, backtest overfitting, partial
observability. Plan over **distributions/expectations**, never enumerated price
trees. Baselines: Almgren-Chriss, TWAP. Discipline: purged/embargoed CV,
Deflated Sharpe, Probability of Backtest Overfitting; kill if PBO > 0.5 or
out-of-sample Sharpe ≤ 0.

Four sights map to execution / allocation with WORST capturing coordinated
adversarial market impact across multiple liquidity providers.

### Phase 7 — Chess four-sight extension (last, optional)

**Only if needed for research** — e.g. modeling multi-player or team chess,
tournament fields, or explicit coalition scenarios. Would require:

* N-actor game spec on top of chess positions
* New evaluators beyond pairwise UCI (coalition policy, joint opponent models)
* `FourWayComparison` path for chess — **not** a rename of today's
  `ThreeWayComparison`

This phase is intentionally last: the 2-actor chess model **remains unchanged**
through Phases 1–6, and the market product does not depend on it.

## Reference: engine stack

| Component | Role | Notes |
| --- | --- | --- |
| **Leela-CF** | OPTIMAL | 191M-param Chessformer in Lc0; +100 Elo vs CNN baseline; native WDL |
| **Maia-3** | HUMAN | 79M params, 57.1% move-match (Allie test set), n=7 history, SelfElo/OppoElo |
| **Mock** | CI gate | Deterministic CPU stand-in when binaries absent |
| **Stockfish** | removed | replaced by Leela-CF on OPTIMAL axis |

Chessformer (ICLR 2026): encoder-only transformer, 64 square tokens, Geometric
Attention Bias (GAB), source-destination policy head. One architecture serves
strength (Leela-CF), human emulation (Maia-3), and interpretability — the
unification rationale for hosting both nets in this repo.
