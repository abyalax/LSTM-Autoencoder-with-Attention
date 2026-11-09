"""Pure-logic tests of the standalone evaluator (no checkpoints or torch model needed)."""

from __future__ import annotations

import numpy as np
import pytest

from lstm_autoencoder.evaluation.metrics import RankingMetrics
from lstm_autoencoder.evaluation.paysim_synthetic_financial.standalone_evaluator import (
    StandaloneConfig,
    StandaloneEvaluator,
    model_config_for,
)
from lstm_autoencoder.model.pipeline import SplitScoringData


def _evaluator(**config) -> StandaloneEvaluator:
    evaluator = StandaloneEvaluator.__new__(StandaloneEvaluator)  # skips data/report setup
    evaluator._config = StandaloneConfig(**config)
    return evaluator


def _split(n: int = 5000, seed: int = 0) -> tuple[SplitScoringData, np.ndarray]:
    rng = np.random.default_rng(seed)
    labels = (rng.random(n) < 0.05).astype(int)
    errors = rng.normal(size=n) + 0.7 * labels
    buckets = rng.choice(["1", "2-4", "5-9", ">=10"], size=n)
    receivers = rng.integers(0, 800, size=n)
    return SplitScoringData(None, labels, np.arange(n), buckets, receivers), errors


@pytest.mark.parametrize(
    ("architecture", "tag"),
    [("attention", "attn"), ("plain", "plain"), ("bottleneck", "bottleneck"), ("attention_pooling", "attn_pool")],
)
def test_architecture_names_map_to_checkpoint_tags(architecture, tag):
    assert model_config_for(architecture).architecture_tag == tag


def test_capacity_suffixes_and_invalid_combinations():
    assert model_config_for("attention_pooling", 16, 4).architecture_tag == "attn_pool_z4_h16"
    with pytest.raises(ValueError):
        model_config_for("plain", 16, 4)  # a latent code needs a single-vector architecture
    with pytest.raises(ValueError, match="Unknown architecture"):
        model_config_for("transformer")


def test_run_scores_match_ranking_metrics_and_buckets_are_consistent():
    data, errors = _split()
    run = _evaluator(n_bootstrap=0)._score_run(errors, data, seed=42, checkpoint="x.pt")
    assert run["metrics"]["auc"] == pytest.approx(RankingMetrics.roc_auc(errors, data.labels))
    assert run["metrics"]["ap"] == pytest.approx(RankingMetrics.average_precision(errors, data.labels))
    assert set(run["metrics"]) == {"auc", "ap", "recall_at_fpr_0.01", "recall_at_fpr_0.001"}
    assert "bootstrap_ci" not in run
    assert sum(b["n"] for b in run["by_bucket"].values()) == run["n"]
    assert sum(b["n_fraud"] for b in run["by_bucket"].values()) == run["n_fraud"]
    for bucket in run["by_bucket"].values():
        assert bucket["fraud_rate"] == pytest.approx(bucket["n_fraud"] / bucket["n"])


def test_receiver_cluster_ci_brackets_the_point_estimate():
    data, errors = _split()
    run = _evaluator(n_bootstrap=300)._score_run(errors, data, seed=1, checkpoint="x.pt")
    ci, metrics = run["bootstrap_ci"], run["metrics"]
    for key in metrics:
        assert ci[f"{key}_ci_low"] <= metrics[key] <= ci[f"{key}_ci_high"], key
    assert 0 < ci["auc_ci_high"] - ci["auc_ci_low"] < 0.2


def test_aggregate_uses_sample_std_across_seeds():
    runs = [
        {"metrics": {"auc": 0.5, "ap": 0.1}},
        {"metrics": {"auc": 0.7, "ap": 0.3}},
        {"metrics": {"auc": 0.6, "ap": 0.2}},
    ]
    aggregate = StandaloneEvaluator._aggregate(runs)
    assert aggregate["auc_mean"] == pytest.approx(0.6)
    assert aggregate["auc_std"] == pytest.approx(0.1)
    assert aggregate["n_seeds"] == 3
    assert StandaloneEvaluator._aggregate([]) == {}
