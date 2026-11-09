"""The weighted bootstrap engine must reproduce `RankingMetrics` on the materialised resample."""

from __future__ import annotations

import numpy as np
import pytest

from lstm_autoencoder.evaluation.bootstrap import WeightedRankingBootstrap
from lstm_autoencoder.evaluation.metrics import RankingMetrics

FPR_TARGETS = (0.01, 0.001)


def _data(seed: int, n: int = 6000, n_receivers: int = 900, ties: bool = False):
    rng = np.random.default_rng(seed)
    labels = (rng.random(n) < 0.04).astype(int)
    receivers = rng.integers(0, n_receivers, size=n)
    base = {"a": rng.normal(size=n) + 0.8 * labels, "b": rng.normal(size=n) + 0.2 * labels}
    if ties:  # coarse scores -> many exact ties, the hard case for AUC/AP/recall@FPR
        base = {name: np.round(values * 2) / 2 for name, values in base.items()}
    base["c"] = rng.normal(size=n)
    return rng, labels, receivers, base


def _reference(scores, labels, weights):
    index = np.repeat(np.arange(len(labels)), weights.astype(int))
    out = {}
    for name, values in scores.items():
        row = {
            "auc": RankingMetrics.roc_auc(values[index], labels[index]),
            "ap": RankingMetrics.average_precision(values[index], labels[index]),
        }
        for fpr in FPR_TARGETS:
            row[f"recall_at_fpr_{fpr:g}"] = RankingMetrics.recall_at_fpr(values[index], labels[index], fpr)
        out[name] = row
    return out


def _assert_equal(engine_metrics, reference):
    for name, row in reference.items():
        for key, expected in row.items():
            assert engine_metrics[name][key] == pytest.approx(expected, rel=1e-9, abs=1e-12), (name, key)


@pytest.mark.parametrize("ties", [False, True])
def test_cluster_draws_match_materialised_resample(ties):
    rng, labels, receivers, scores = _data(0, ties=ties)
    _, inverse = np.unique(receivers, return_inverse=True)
    engine = WeightedRankingBootstrap(labels, scores, FPR_TARGETS)
    for _ in range(8):
        weights = engine.cluster_weights(inverse, int(inverse.max()) + 1, rng)
        _assert_equal(engine.metrics(weights), _reference(scores, labels, weights))


def test_stratified_row_draws_match_materialised_resample():
    rng, labels, _, scores = _data(1)
    engine = WeightedRankingBootstrap(labels, scores, FPR_TARGETS)
    for _ in range(5):
        weights = engine.stratified_row_weights(labels, rng)
        assert int(weights[labels == 1].sum()) == int((labels == 1).sum())  # class counts preserved
        _assert_equal(engine.metrics(weights), _reference(scores, labels, weights))


def test_full_sample_weights_equal_plain_metrics():
    _, labels, _, scores = _data(2)
    engine = WeightedRankingBootstrap(labels, scores, FPR_TARGETS)
    _assert_equal(engine.metrics(np.ones(len(labels))), _reference(scores, labels, np.ones(len(labels))))


def test_draw_without_positives_is_nan():
    _, labels, _, scores = _data(3)
    engine = WeightedRankingBootstrap(labels, scores, FPR_TARGETS)
    weights = (labels == 0).astype(float)
    metrics = engine.metrics(weights)
    assert all(np.isnan(value) for value in metrics["a"].values())


def test_non_finite_scores_are_ignored():
    _, labels, _, scores = _data(4)
    scores["a"] = scores["a"].copy()
    scores["a"][::97] = np.nan
    engine = WeightedRankingBootstrap(labels, scores, FPR_TARGETS)
    weights = np.ones(len(labels))
    finite = np.isfinite(scores["a"])
    expected = RankingMetrics.roc_auc(scores["a"][finite], labels[finite])
    assert engine.metrics(weights)["a"]["auc"] == pytest.approx(expected, rel=1e-9)
