"""The hybrid scorer must reward a genuinely informative error and give no credit to noise."""

from __future__ import annotations

import numpy as np

from lstm_autoencoder.evaluation.banksim.hybrid_scoring import (
    GBM_VAL_CONTROL,
    HYBRID_GBM_VAL,
    NULL_HYBRID,
    HybridInputs,
    HybridScorer,
    HybridScoringConfig,
    SplitArrays,
)


def _split(rng: np.random.Generator, n_customers: int, per_customer: int) -> tuple[SplitArrays, np.ndarray]:
    n = n_customers * per_customer
    customers = np.repeat(np.arange(n_customers), per_customer)
    visible = rng.normal(size=(n, 3))
    hidden = rng.normal(size=n)  # signal the tabular features cannot see
    logit = -3.2 + 1.2 * visible[:, 0] + 1.6 * hidden
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return SplitArrays(visible, y, customers), hidden


def _inputs(seed: int = 0) -> tuple[HybridInputs, dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    train, _ = _split(rng, 300, 40)
    val, hidden_val = _split(rng, 150, 40)
    test, hidden_test = _split(rng, 150, 40)
    return HybridInputs(train, val, test), {"val": hidden_val, "test": hidden_test}


def test_informative_error_improves_the_gbm_and_noise_does_not() -> None:
    inputs, hidden = _inputs()
    scorer = HybridScorer(inputs, HybridScoringConfig(n_bootstrap=40))

    informative = {k: np.exp(v + np.random.default_rng(1).normal(scale=0.3, size=len(v))) for k, v in hidden.items()}
    good = scorer.score_seed(1, informative)
    assert good["deltas"][f"{HYBRID_GBM_VAL}_vs_{GBM_VAL_CONTROL}"]["auc"] > 0.03
    # The shuffled-error control stays near zero.
    assert abs(good["deltas"][f"{NULL_HYBRID}_vs_{GBM_VAL_CONTROL}"]["auc"]) < 0.02

    rng = np.random.default_rng(2)
    useless = {k: np.exp(rng.normal(size=len(v))) for k, v in hidden.items()}
    bad = scorer.score_seed(2, useless)
    assert bad["deltas"][f"{HYBRID_GBM_VAL}_vs_{GBM_VAL_CONTROL}"]["auc"] < 0.02


def test_bootstrap_reports_a_ci_that_excludes_zero_for_a_real_gain() -> None:
    inputs, hidden = _inputs(seed=3)
    scorer = HybridScorer(inputs, HybridScoringConfig(n_bootstrap=60))
    errors = {k: np.exp(v) for k, v in hidden.items()}
    result = scorer.score_seed(3, errors)
    ci = result["bootstrap"][f"{HYBRID_GBM_VAL}_vs_{GBM_VAL_CONTROL}"]
    assert ci["d_auc_ci_low"] > 0.0 and ci["p_d_auc_le_0"] < 0.05
