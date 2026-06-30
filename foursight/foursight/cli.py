"""``foursight`` command-line entry point.

Three commands make the three-axis idea tangible without any GPU:

* ``foursight engines doctor`` — show whether each axis resolves real or mock.
* ``foursight compare "<FEN>" --elo 1500 [--played e2e4]`` — one three-way table.
* ``foursight replay game.pgn --elo auto`` — a per-move three-way stream.
"""

from __future__ import annotations

from pathlib import Path

import chess
import typer

from foursight.compare import ThreeWayComparator, ThreeWayComparison
from foursight.config import load_settings
from foursight.engines.registry import build_engines, doctor
from foursight.ingest.actual import ActualRecord, records_from_fen_moves, records_from_pgn

app = typer.Typer(
    add_completion=False,
    help="4sight — compare optimal (Leela-CF) vs human (Maia-3) vs actual.",
)
engines_app = typer.Typer(add_completion=False, help="Inspect the engine pair.")
app.add_typer(engines_app, name="engines")


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


def main() -> None:
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
