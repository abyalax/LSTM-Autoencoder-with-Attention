"""Score-strategy freezing: chosen on validation only, immutable, bound to the checkpoints it was frozen on."""

from __future__ import annotations

import json

import pytest

from lstm_autoencoder.evaluation.paysim_synthetic_financial.score_strategy import ScoreStrategySelector
from lstm_autoencoder.evaluation.paysim_synthetic_financial.score_strategy_registry import (
    FrozenScoreStrategy,
    checkpoint_key,
    config_key,
)
from lstm_autoencoder.evaluation.paysim_synthetic_financial.standalone_evaluator import StandaloneConfig


def _write_registry(artifacts_dir, entries):
    path = FrozenScoreStrategy.registry_path(artifacts_dir)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"entries": entries}), encoding="utf-8")


def test_last_is_chosen_only_when_it_beats_mean_by_more_than_min_gain():
    choose = ScoreStrategySelector.choose_strategy
    assert choose({"mean": 0.020, "last": 0.030}) == "last"
    assert choose({"mean": 0.030, "last": 0.020}) == "mean"
    assert choose({"mean": 0.020, "last": 0.020}) == "mean"  # a tie keeps the training-loss score
    assert choose({"mean": 0.020, "last": 0.021}, min_gain=0.005) == "mean"


def test_missing_registry_fails_loudly_with_the_command_to_run(tmp_path):
    with pytest.raises(FileNotFoundError, match="select-score-strategy"):
        FrozenScoreStrategy.load(tmp_path)


def test_registry_returns_the_frozen_strategy_and_rejects_retrained_checkpoints(tmp_path):
    frozen_on = [checkpoint_key("v1_loss/lstm_ae_w3_plain_seed42_best.pt")]
    _write_registry(tmp_path, {config_key("plain", 3): {"strategy": "last", "checkpoints": frozen_on}})
    registry = FrozenScoreStrategy.load(tmp_path)

    assert registry.strategy_for("plain", 3) == "last"
    assert registry.strategy_for("plain", 3, ["x/v1_loss/lstm_ae_w3_plain_seed42_best.pt"]) == "last"
    with pytest.raises(ValueError, match="retrained"):
        registry.strategy_for("plain", 3, ["x/v2_loss/lstm_ae_w3_plain_seed42_best.pt"])
    with pytest.raises(KeyError, match="select-score-strategy"):
        registry.strategy_for("attn", 3)


def test_invalid_frozen_strategy_is_rejected(tmp_path):
    _write_registry(tmp_path, {config_key("plain", 1): {"strategy": "max", "checkpoints": []}})
    with pytest.raises(ValueError):
        FrozenScoreStrategy.load(tmp_path).strategy_for("plain", 1)


def test_test_split_cannot_override_the_frozen_strategy_by_hand():
    with pytest.raises(ValueError, match="frozen"):
        StandaloneConfig(split="test", score_strategy="last")
    assert StandaloneConfig(split="val", score_strategy="last").score_strategy == "last"
    assert StandaloneConfig(split="test").score_strategy is None
