"""Configuration for the 4sight engine pair.

Settings are layered: built-in defaults (pointing at the in-repo Lc0 binary and
the ``maia3`` submodule's UCI entry) are overridden by a TOML file, which is in
turn overridden by a small set of environment variables. Nothing here launches
an engine — the registry decides, per role, whether a real binary is available
and otherwise falls back to the dependency-free mock.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path

from pydantic import BaseModel, Field

#: Environment variable holding an explicit path to a ``settings.toml``.
ENV_SETTINGS = "FOURSIGHT_SETTINGS"
#: When truthy, the registry never launches a real engine (CI / CPU gate).
ENV_FORCE_MOCK = "FOURSIGHT_FORCE_MOCK"


class EngineConfig(BaseModel):
    """Launch + evaluation parameters for a single engine role."""

    command: list[str] = Field(default_factory=list)
    """argv used to ``popen_uci`` the engine (empty -> no real binary)."""

    net: str | None = None
    """Weights path passed to the engine (Lc0 ``--weights``); ``None`` for Maia."""

    default_elo: int = 1500
    """Elo used when a caller does not pass one explicitly."""

    device: str = "cpu"
    """Backend/device hint (e.g. ``cpu``, ``cuda``); advisory only in Phase 0."""

    nodes: int | None = None
    """Search node budget. Prefer this over wall-clock for reproducibility."""


def _default_optimal() -> EngineConfig:
    # Leela-CF (OPTIMAL): the repo's built Lc0 + a Chessformer strength net.
    return EngineConfig(
        command=["build/release/lc0"],
        net="nets/leela-cf.pb.gz",
        default_elo=1500,
        device="cuda",
        nodes=800,
    )


def _default_human() -> EngineConfig:
    # Maia-3 (HUMAN): the maia3 submodule's Python UCI predictor.
    return EngineConfig(
        command=["maia3-5m"],
        net=None,
        default_elo=1500,
        device="cpu",
        nodes=1,
    )


class Settings(BaseModel):
    """Top-level 4sight configuration: the optimal + human engine pair."""

    force_mock: bool = False
    optimal: EngineConfig = Field(default_factory=_default_optimal)
    human: EngineConfig = Field(default_factory=_default_human)


def _read_toml(path: Path) -> dict:
    with path.open("rb") as fh:
        return tomllib.load(fh)


def default_settings_path() -> Path | None:
    """Resolve the settings file from env or the package's ``config`` dir."""
    env = os.environ.get(ENV_SETTINGS)
    if env:
        return Path(env)
    # foursight/config/settings.toml relative to the project directory.
    here = Path(__file__).resolve().parent.parent
    candidate = here / "config" / "settings.toml"
    return candidate if candidate.exists() else None


def load_settings(path: str | os.PathLike[str] | None = None) -> Settings:
    """Load :class:`Settings`, merging a TOML file over defaults if present.

    ``FOURSIGHT_FORCE_MOCK`` always wins so CI can guarantee the mock path.
    """
    resolved = Path(path) if path is not None else default_settings_path()
    data: dict = {}
    if resolved is not None and resolved.exists():
        data = _read_toml(resolved)

    settings = Settings.model_validate(data) if data else Settings()

    env_force = os.environ.get(ENV_FORCE_MOCK)
    if env_force is not None and env_force.strip().lower() in {"1", "true", "yes", "on"}:
        settings = settings.model_copy(update={"force_mock": True})

    return settings
