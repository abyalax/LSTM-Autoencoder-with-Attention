"""Pipeline-revision stamp that ties a checkpoint to the pipeline that trained it.

A checkpoint is only meaningful for the exact feature construction, windowing,
padding/packing and split it was trained with. Because every one of those can
change without changing the tensor shapes, `torch.load` happily accepts a stale
checkpoint and evaluation silently mixes incompatible pipelines. Training stamps
`PIPELINE_REVISION` into the version directory's `metadata.json`; evaluation
refuses any checkpoint whose stamp is missing or different.

Bump `PIPELINE_REVISION` whenever features, entity definition, tie handling,
padding/packing, scaling or the split change.
"""

from __future__ import annotations

import json
from pathlib import Path

PIPELINE_REVISION = "2026-11-10-role-aware-strict-step-rightpad-v1"
REVISION_KEY = "pipeline_revision"


class IncompatibleCheckpointError(RuntimeError):
    """Raised when a checkpoint was not trained with the current pipeline revision."""


def read_checkpoint_revision(checkpoint_path: str | Path) -> str | None:
    """Revision stamp recorded next to a checkpoint, or None when absent/unreadable."""
    metadata_path = Path(checkpoint_path).parent / "metadata.json"
    if not metadata_path.exists():
        return None
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    value = payload.get(REVISION_KEY) if isinstance(payload, dict) else None
    return str(value) if value is not None else None


def assert_checkpoint_compatible(checkpoint_path: str | Path) -> None:
    """Raise `IncompatibleCheckpointError` unless the checkpoint carries the current revision."""
    found = read_checkpoint_revision(checkpoint_path)
    if found != PIPELINE_REVISION:
        raise IncompatibleCheckpointError(
            f"Checkpoint {Path(checkpoint_path).parent.name}/{Path(checkpoint_path).name} was trained with "
            f"pipeline revision {found!r}, but the current pipeline is {PIPELINE_REVISION!r}. Its features, "
            "windows, padding or split differ from what is being evaluated, so the numbers would be invalid. "
            "Retrain it with `train-all-configs`, then re-run `select-score-strategy` before evaluating."
        )
