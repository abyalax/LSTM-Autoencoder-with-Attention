"""`HybridClassifierEvaluator._bootstrap` must equal the naive materialised-resample bootstrap, draw for draw."""

from __future__ import annotations

import numpy as np
import pytest

from lstm_autoencoder.evaluation.metrics import RankingMetrics
from lstm_autoencoder.evaluation.paysim_synthetic_financial.hybrid_classifier import (
    BOOTSTRAP_COMPARISONS,
    DIAGNOSTIC_COMPARISONS,
    HybridClassifierConfig,
    HybridClassifierEvaluator,
)

PAIRS = (*BOOTSTRAP_COMPARISONS, *DIAGNOSTIC_COMPARISONS)
NAMES = sorted({name for pair in PAIRS for name in pair})


def _evaluator(mode: str, n_bootstrap: int) -> HybridClassifierEvaluator:
    evaluator = HybridClassifierEvaluator.__new__(HybridClassifierEvaluator)  # skips the heavy constructor
    evaluator._config = HybridClassifierConfig(n_bootstrap=n_bootstrap, bootstrap_mode=mode)
    return evaluator


def _fixture(seed: int = 0, n: int = 4000, n_receivers: int = 700):
    rng = np.random.default_rng(seed)
    labels = (rng.random(n) < 0.05).astype(int)
    receivers = rng.integers(0, n_receivers, size=n)
    scores = {name: rng.normal(size=n) + (0.3 + 0.1 * i) * labels for i, name in enumerate(NAMES)}
    return labels, receivers, scores


def _naive(mode, labels, receivers, scores, config, seed):
    """The pre-optimisation algorithm: materialise every resample, then call RankingMetrics."""
    rng = np.random.default_rng(config.random_seed + 1_000 + seed)
    if mode == "receiver":
        _, inverse = np.unique(receivers, return_inverse=True)
        rows = [np.flatnonzero(inverse == i) for i in range(int(inverse.max()) + 1)]
        draw = lambda: np.concatenate([rows[i] for i in rng.integers(0, len(rows), size=len(rows))])  # noqa: E731
    else:
        pos, neg = np.flatnonzero(labels == 1), np.flatnonzero(labels == 0)
        draw = lambda: np.concatenate([rng.choice(pos, size=len(pos)), rng.choice(neg, size=len(neg))])  # noqa: E731
    out = {f"{c}_vs_{r}": {"auc": [], "ap": []} for c, r in PAIRS}
    for _ in range(config.n_bootstrap):
        index = draw()
        y = labels[index]
        auc = {n: RankingMetrics.roc_auc(scores[n][index], y) for n in NAMES}
        ap = {n: RankingMetrics.average_precision(scores[n][index], y) for n in NAMES}
        for c, r in PAIRS:
            out[f"{c}_vs_{r}"]["auc"].append(auc[c] - auc[r])
            out[f"{c}_vs_{r}"]["ap"].append(ap[c] - ap[r])
    return out


@pytest.mark.parametrize("mode", ["receiver", "row"])
def test_bootstrap_matches_naive_algorithm(mode):
    labels, receivers, scores = _fixture()
    evaluator = _evaluator(mode, n_bootstrap=15)
    summary = evaluator._bootstrap(scores, labels, 7, receivers)
    naive = _naive(mode, labels, receivers, scores, evaluator._config, 7)
    for pair, draws in naive.items():
        low, high = np.percentile(draws["auc"], [2.5, 97.5])
        assert summary[pair]["d_auc_ci_low"] == pytest.approx(low, abs=1e-9)
        assert summary[pair]["d_auc_ci_high"] == pytest.approx(high, abs=1e-9)
        low, high = np.percentile(draws["ap"], [2.5, 97.5])
        assert summary[pair]["d_ap_ci_low"] == pytest.approx(low, abs=1e-9)
        assert summary[pair]["d_ap_ci_high"] == pytest.approx(high, abs=1e-9)
        assert summary[pair]["p_d_ap_le_0"] == pytest.approx(np.mean(np.asarray(draws["ap"]) <= 0.0))


def test_null_comparisons_get_an_ap_interval_too():
    labels, receivers, scores = _fixture(1)
    summary = _evaluator("receiver", n_bootstrap=10)._bootstrap(scores, labels, 0, receivers)
    for candidate, reference in DIAGNOSTIC_COMPARISONS:
        entry = summary[f"{candidate}_vs_{reference}"]
        assert {"d_ap_ci_low", "d_ap_ci_high", "p_d_ap_le_0"} <= set(entry)


def test_unknown_bootstrap_mode_is_rejected():
    labels, receivers, scores = _fixture(2)
    with pytest.raises(ValueError, match="bootstrap_mode"):
        _evaluator("block", n_bootstrap=2)._bootstrap(scores, labels, 0, receivers)
