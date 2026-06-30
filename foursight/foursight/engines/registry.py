"""Build the {optimal, human} engine pair from config, with mock fallback.

The registry is the only place that decides *real vs mock*. For each role it
checks whether the configured binary (and, for Leela-CF, the net) actually
exists; if not — or if ``force_mock`` is set, or construction fails — it
substitutes a :class:`~foursight.engines.mock.MockEngine` and records why. This
is what keeps the Phase-0 gate green on a CPU box with no engines installed.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass

from foursight.config import EngineConfig, Settings, load_settings
from foursight.engines.base import ChessEngine
from foursight.engines.leela_cf import LeelaCFEngine
from foursight.engines.maia3 import Maia3Engine
from foursight.engines.mock import MockEngine


@dataclass
class EnginePair:
    """The two side-by-side axes plus a record of how each was resolved."""

    optimal: ChessEngine
    human: ChessEngine

    def close(self) -> None:
        for engine in (self.optimal, self.human):
            engine.close()

    def __enter__(self) -> EnginePair:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _resolves(program: str) -> bool:
    """True if ``program`` is on PATH or is an existing file."""
    if shutil.which(program) is not None:
        return True
    return os.path.exists(program)


def _optimal_available(cfg: EngineConfig) -> tuple[bool, str]:
    if not cfg.command:
        return False, "no command configured"
    if not _resolves(cfg.command[0]):
        return False, f"binary not found: {cfg.command[0]}"
    if cfg.net and not os.path.exists(cfg.net):
        return False, f"net not found: {cfg.net}"
    return True, "ok"


def _human_available(cfg: EngineConfig) -> tuple[bool, str]:
    if not cfg.command:
        return False, "no command configured"
    if not _resolves(cfg.command[0]):
        return False, f"binary not found: {cfg.command[0]}"
    return True, "ok"


def _build_optimal(settings: Settings) -> ChessEngine:
    cfg = settings.optimal
    if settings.force_mock:
        return MockEngine("optimal", default_elo=cfg.default_elo)
    ok, _ = _optimal_available(cfg)
    if not ok:
        return MockEngine("optimal", default_elo=cfg.default_elo)
    try:
        return LeelaCFEngine(
            cfg.command, net=cfg.net, device=cfg.device, default_nodes=cfg.nodes
        )
    except Exception as exc:  # noqa: BLE001 - degrade gracefully to mock
        engine = MockEngine("optimal", default_elo=cfg.default_elo)
        engine.backend = "mock (fallback)"
        engine.name = f"mock-optimal (launch failed: {exc})"
        return engine


def _build_human(settings: Settings) -> ChessEngine:
    cfg = settings.human
    if settings.force_mock:
        return MockEngine("human", default_elo=cfg.default_elo)
    ok, _ = _human_available(cfg)
    if not ok:
        return MockEngine("human", default_elo=cfg.default_elo)
    try:
        return Maia3Engine(cfg.command, default_elo=cfg.default_elo)
    except Exception as exc:  # noqa: BLE001 - degrade gracefully to mock
        engine = MockEngine("human", default_elo=cfg.default_elo)
        engine.backend = "mock (fallback)"
        engine.name = f"mock-human (launch failed: {exc})"
        return engine


def build_engines(settings: Settings | None = None) -> EnginePair:
    """Construct the optimal + human pair, falling back to mocks as needed."""
    settings = settings or load_settings()
    return EnginePair(optimal=_build_optimal(settings), human=_build_human(settings))


def doctor(settings: Settings | None = None) -> list[dict[str, object]]:
    """Report, without launching anything real, what each role would resolve to.

    Returns one row per role with the backend and whether it is mock so that
    ``foursight engines doctor`` can tell real from mock at a glance.
    """
    settings = settings or load_settings()
    rows: list[dict[str, object]] = []
    for role, cfg, check in (
        ("optimal", settings.optimal, _optimal_available),
        ("human", settings.human, _human_available),
    ):
        if settings.force_mock:
            available, detail, backend = False, "force_mock", "mock"
        else:
            available, detail = check(cfg)
            backend = ("leela-cf" if role == "optimal" else "maia3") if available else "mock"
        rows.append(
            {
                "role": role,
                "backend": backend,
                "is_mock": not available,
                "command": " ".join(cfg.command) if cfg.command else "(none)",
                "detail": detail,
            }
        )
    return rows
