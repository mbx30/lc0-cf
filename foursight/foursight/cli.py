"""``foursight`` command-line entry point.

Three commands make the three-axis idea tangible without any GPU:

* ``foursight engines doctor`` — show whether each axis resolves real or mock.
* ``foursight compare "<FEN>" --elo 1500 [--played e2e4]`` — one three-way table.
* ``foursight replay game.pgn --elo auto`` — a per-move three-way stream.
* ``foursight calibrate game.pgn --elo auto`` — aggregate calibration metrics.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import chess
import chess.pgn
import typer

from foursight.compare import ThreeWayComparator, ThreeWayComparison
from foursight.config import load_settings
from foursight.engines.registry import build_engines, doctor
from foursight.ingest.actual import (
    ActualRecord,
    records_from_fen_moves,
    records_from_game,
    records_from_pgn,
)
from foursight.metrics import (
    CalibrationSummary,
    aggregate_by_elo_band,
    aggregate_by_phase,
    comparisons_to_rows,
    write_rows_parquet,
)

app = typer.Typer(
    add_completion=False,
    help="4sight — compare optimal (Leela-CF) vs human (Maia-3) vs actual.",
)
engines_app = typer.Typer(add_completion=False, help="Inspect the engine pair.")
app.add_typer(engines_app, name="engines")
ingest_app = typer.Typer(add_completion=False, help="Bulk-ingest the ACTUAL axis.")
app.add_typer(ingest_app, name="ingest")


def _fmt_prob(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}"


def _print_comparison(c: ThreeWayComparison) -> None:
    opt, hum = c.optimal, c.human
    typer.echo(f"FEN: {c.fen}")
    typer.echo(f"Elo: {c.elo}")
    opt_exp = _fmt_prob(opt.expectation)
    opt_best = opt.bestmove.uci() if opt.bestmove else "—"
    hum_best = hum.bestmove.uci() if hum.bestmove else "—"
    typer.echo(f"  OPTIMAL  best={opt_best:<6} expectation={opt_exp}  wdl={opt.wdl}")
    typer.echo(f"  HUMAN    best={hum_best:<6} (Elo {hum.elo})")
    if c.actual is not None:
        played = c.actual.played_move.uci()
        typer.echo(
            f"  ACTUAL   played={played:<6} "
            f"result={c.actual.game_result} score={_fmt_prob(c.actual.realized_score())}"
        )

    g = c.gaps
    typer.echo("  gaps:")
    typer.echo(
        f"    optimal vs human : agree@1={g.opt_vs_human.agree_at_1} "
        f"JS={_fmt_prob(g.opt_vs_human.js_divergence)} "
        f"KL={_fmt_prob(g.opt_vs_human.kl_divergence)} "
        f"Δexp(human)={_fmt_prob(g.opt_vs_human.delta_expectation)}"
    )
    if c.actual is not None:
        typer.echo(
            f"    human vs actual  : agree@1={g.human_vs_actual.agree_at_1} "
            f"P(actual|human)={_fmt_prob(g.human_vs_actual.target_prob)}"
        )
        typer.echo(
            f"    optimal vs actual: agree@1={g.opt_vs_actual.agree_at_1} "
            f"Δexp(actual)={_fmt_prob(g.opt_vs_actual.delta_expectation)} "
            f"realized={_fmt_prob(g.opt_vs_actual.realized_score)}"
        )


def _print_calibration_table(title: str, summaries: list[CalibrationSummary]) -> None:
    typer.echo(title)
    typer.echo(
        "  bucket        n     hum@1  opt@1  JS     KL     Δexp(h)  Δexp(a)  P(act|hum)"
    )
    for summary in summaries:
        typer.echo(
            f"  {summary.bucket:<12}"
            f"{summary.n_positions:>5}  "
            f"{_fmt_prob(summary.human_agree_at_1):>6}  "
            f"{_fmt_prob(summary.optimal_agree_at_1):>6}  "
            f"{_fmt_prob(summary.mean_js):>5}  "
            f"{_fmt_prob(summary.mean_kl):>5}  "
            f"{_fmt_prob(summary.mean_delta_exp_human):>7}  "
            f"{_fmt_prob(summary.mean_delta_exp_actual):>7}  "
            f"{_fmt_prob(summary.mean_p_actual_given_human):>10}"
        )
    typer.echo("")


def _resolve_pgn_paths(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    if source.is_dir():
        paths = sorted(p for p in source.iterdir() if p.suffix.lower() == ".pgn")
        if paths:
            return paths
        raise typer.BadParameter(f"no .pgn files found in directory: {source}")
    raise typer.BadParameter(f"source path does not exist: {source}")


def _records_for_calibration(
    source: Path,
    *,
    max_games: int | None,
    max_moves: int | None,
) -> list[ActualRecord]:
    records: list[ActualRecord] = []
    games_seen = 0
    for path in _resolve_pgn_paths(source):
        with path.open(encoding="utf-8") as handle:
            while True:
                game = chess.pgn.read_game(handle)
                if game is None:
                    break
                records.extend(records_from_game(game))
                games_seen += 1
                if max_games is not None and games_seen >= max_games:
                    break
        if max_games is not None and games_seen >= max_games:
            break
    if max_moves is not None:
        records = records[:max_moves]
    return records


@engines_app.command("doctor")
def engines_doctor() -> None:
    """Report each engine's resolved backend (real vs mock)."""
    settings = load_settings()
    rows = doctor(settings)
    typer.echo(f"force_mock = {settings.force_mock}")
    for row in rows:
        flag = "MOCK" if row["is_mock"] else "REAL"
        typer.echo(
            f"  [{flag}] {row['role']:<8} backend={row['backend']:<16} "
            f"cmd='{row['command']}'  ({row['detail']})"
        )


@app.command("compare")
def compare_position(
    fen: str = typer.Argument(..., help="Board FEN to analyse."),
    elo: int = typer.Option(1500, help="Human (Maia-3) skill level."),
    played: str | None = typer.Option(None, help="ACTUAL move actually played (UCI)."),
    result: str | None = typer.Option(None, help="Game result, e.g. 1-0 / 0-1 / 1/2-1/2."),
    nodes: int | None = typer.Option(None, help="Search node budget (else config default)."),
) -> None:
    """Emit the three-way comparison for a single position."""
    board = chess.Board(fen)
    actual: ActualRecord | None = None
    if played is not None:
        actual = records_from_fen_moves(fen, [played], game_result=result)[0]

    with build_engines(load_settings()) as pair:
        comparator = ThreeWayComparator(pair.optimal, pair.human)
        comparison = comparator.compare(board, elo=elo, actual=actual, nodes=nodes)
    _print_comparison(comparison)


@app.command("replay")
def replay(
    pgn: Path = typer.Argument(..., help="PGN file to replay."),
    elo: str = typer.Option("auto", help="'auto' (use player Elo) or a fixed integer."),
    max_moves: int | None = typer.Option(None, help="Limit the number of positions."),
    nodes: int | None = typer.Option(None, help="Search node budget per position."),
) -> None:
    """Stream a per-move three-way comparison over a real game."""
    records = list(records_from_pgn(pgn))
    if max_moves is not None:
        records = records[:max_moves]

    with build_engines(load_settings()) as pair:
        comparator = ThreeWayComparator(pair.optimal, pair.human)
        for idx, record in enumerate(records, start=1):
            if elo == "auto":
                use_elo = record.player_elo or 1500
            else:
                use_elo = int(elo)
            board = record.board
            comparison = comparator.compare(
                board, elo=use_elo, actual=record, nodes=nodes
            )
            typer.echo(f"--- ply {idx} ---")
            _print_comparison(comparison)


@app.command("calibrate")
def calibrate(
    source: Path = typer.Argument(..., help="PGN file or directory of PGN files."),
    elo: str = typer.Option("auto", help="'auto' (use mover Elo) or a fixed integer."),
    max_games: int | None = typer.Option(None, help="Limit number of PGN games processed."),
    max_moves: int | None = typer.Option(None, help="Limit number of positions processed."),
    nodes: int | None = typer.Option(None, help="Search node budget per position."),
    json_out: Path | None = typer.Option(
        None, help="Optional JSON report output path (rows + summaries)."
    ),
    parquet_out: Path | None = typer.Option(
        None, help="Optional parquet output path for flattened per-position rows."
    ),
) -> None:
    """Aggregate calibration metrics over one or more games."""
    records = _records_for_calibration(source, max_games=max_games, max_moves=max_moves)
    if not records:
        typer.echo("No records found for calibration.")
        raise typer.Exit(1)

    fixed_elo: int | None = None
    if elo != "auto":
        try:
            fixed_elo = int(elo)
        except ValueError as exc:
            raise typer.BadParameter("elo must be 'auto' or an integer.") from exc

    comparisons: list[ThreeWayComparison] = []
    with build_engines(load_settings()) as pair:
        comparator = ThreeWayComparator(pair.optimal, pair.human)
        for record in records:
            use_elo = fixed_elo if fixed_elo is not None else (record.player_elo or 1500)
            comparisons.append(
                comparator.compare(record.board, elo=use_elo, actual=record, nodes=nodes)
            )

    rows = comparisons_to_rows(comparisons)
    by_elo = aggregate_by_elo_band(rows)
    by_phase = aggregate_by_phase(rows)

    typer.echo(f"=== Calibration report (n={len(rows)} positions) ===")
    _print_calibration_table("By Elo band:", by_elo)
    _print_calibration_table("By phase:", by_phase)

    if json_out is not None:
        payload = {
            "source": str(source),
            "rows": [asdict(r) for r in rows],
            "by_elo_band": [asdict(s) for s in by_elo],
            "by_phase": [asdict(s) for s in by_phase],
        }
        json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        typer.echo(f"Wrote JSON report to {json_out}")

    if parquet_out is not None:
        write_rows_parquet(rows, parquet_out)
        typer.echo(f"Wrote parquet rows to {parquet_out}")


def main() -> None:
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
