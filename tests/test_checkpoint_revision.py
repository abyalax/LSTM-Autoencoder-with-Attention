"""Stale checkpoints must be rejected before evaluation (torch-free)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lstm_autoencoder.model.revision import (
    PIPELINE_REVISION,
    REVISION_KEY,
    IncompatibleCheckpointError,
    assert_checkpoint_compatible,
    read_checkpoint_revision,
)


def _version_dir(tmp_path: Path, metadata: dict | None) -> Path:
    directory = tmp_path / "version_1_loss"
    directory.mkdir()
    if metadata is not None:
        (directory / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return directory / "lstm_ae_w1_plain_seed42_best.pt"


def test_current_revision_is_accepted(tmp_path: Path) -> None:
    checkpoint = _version_dir(tmp_path, {REVISION_KEY: PIPELINE_REVISION})
    assert read_checkpoint_revision(checkpoint) == PIPELINE_REVISION
    assert_checkpoint_compatible(checkpoint)


@pytest.mark.parametrize("metadata", [None, {}, {REVISION_KEY: "older-revision"}])
def test_missing_or_old_revision_is_rejected(tmp_path: Path, metadata: dict | None) -> None:
    checkpoint = _version_dir(tmp_path, metadata)
    with pytest.raises(IncompatibleCheckpointError, match="train-all-configs"):
        assert_checkpoint_compatible(checkpoint)


def test_corrupt_metadata_is_rejected(tmp_path: Path) -> None:
    checkpoint = _version_dir(tmp_path, None)
    (checkpoint.parent / "metadata.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(IncompatibleCheckpointError):
        assert_checkpoint_compatible(checkpoint)
