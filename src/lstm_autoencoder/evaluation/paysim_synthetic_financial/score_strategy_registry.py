"""Frozen reconstruction-score strategy per (architecture, window), chosen on validation only."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from lstm_autoencoder.model.config import SCORE_STRATEGIES

REGISTRY_FILENAME = "score_strategy.json"


def checkpoint_key(path: str | Path) -> str:
    """Machine-independent identity of a checkpoint: `<version dir>/<file name>`."""
    path = Path(path)
    return f"{path.parent.name}/{path.name}"


def config_key(architecture_tag: str, window: int) -> str:
    """Registry key of one (architecture, window) configuration."""
    return f"{architecture_tag}|w{window}"


@dataclass(frozen=True)
class FrozenScoreStrategy:
    """The score strategy (`mean` or `last`) frozen per configuration BEFORE any test scoring.

    Written once by `select-score-strategy` from validation AP only. Every test-time
    workflow (standalone evaluation, hybrid classifier) reads it instead of choosing
    for itself, and refuses to run if it is missing or was frozen on other checkpoints.
    """

    entries: dict[str, dict[str, object]]
    path: Path

    @staticmethod
    def registry_path(artifacts_dir: Path) -> Path:
        """Location of the registry file inside the model artifacts folder."""
        return artifacts_dir / "model" / "paysim_synthetic_financial" / REGISTRY_FILENAME

    @classmethod
    def load(cls, artifacts_dir: Path) -> FrozenScoreStrategy:
        """Load the registry; raise with the exact command to run when it does not exist."""
        path = cls.registry_path(artifacts_dir)
        if not path.exists():
            raise FileNotFoundError(
                f"No frozen score strategy at {path}. Run `select-score-strategy` on the trained "
                "checkpoints (it uses validation only) before scoring the test split."
            )
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(entries=payload["entries"], path=path)

    def strategy_for(self, architecture_tag: str, window: int, checkpoints: Sequence[str | Path] = ()) -> str:
        """Frozen strategy of one configuration; verifies it was frozen on (a superset of) these checkpoints."""
        key = config_key(architecture_tag, window)
        if key not in self.entries:
            raise KeyError(f"Configuration {key} is not in {self.path}; run `select-score-strategy` for it first.")
        entry = self.entries[key]
        strategy = str(entry["strategy"])
        if strategy not in SCORE_STRATEGIES:
            raise ValueError(f"Frozen strategy {strategy!r} for {key} is not one of {SCORE_STRATEGIES}.")
        if checkpoints:
            frozen_on = set(entry.get("checkpoints", []))  # type: ignore[arg-type]
            current = {checkpoint_key(path) for path in checkpoints}
            if not current <= frozen_on:
                raise ValueError(
                    f"Score strategy of {key} was frozen on checkpoints {sorted(frozen_on)} but is being used "
                    f"with {sorted(current)}; the checkpoints were retrained, so re-run `select-score-strategy`."
                )
        return strategy
