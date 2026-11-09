"""Figure helpers: the pooled-AUC decomposition must equal a brute-force pair count."""

from __future__ import annotations

import numpy as np
import pytest

from lstm_autoencoder.config import DatasetConfig
from lstm_autoencoder.evaluation.metrics import RankingMetrics
from lstm_autoencoder.reporting.figures import BUCKETS, FigureBuilder


def _per_bucket(scores, labels, buckets):
    out = {}
    for bucket in BUCKETS:
        mask = buckets == bucket
        out[bucket] = {
            "n": int(mask.sum()),
            "n_fraud": int(labels[mask].sum()),
            "auc_mean": RankingMetrics.roc_auc(scores[mask], labels[mask]),
        }
    out["pooled"] = {"auc_mean": RankingMetrics.roc_auc(scores, labels)}
    return out


def test_decomposition_matches_brute_force_pair_counts():
    rng = np.random.default_rng(0)
    n = 700
    buckets = rng.choice(BUCKETS, size=n, p=[0.3, 0.3, 0.2, 0.2])
    labels = (rng.random(n) < np.select([buckets == "1", buckets == "2-4"], [0.2, 0.08], 0.03)).astype(int)
    # error rises with bucket and carries a little signal inside buckets: the confounded case
    scores = np.round(rng.normal(size=n) + 0.8 * labels + np.searchsorted(BUCKETS, buckets), 1)

    parts = FigureBuilder.decompose_pooled_auc(_per_bucket(scores, labels, buckets))

    pos, neg = np.flatnonzero(labels == 1), np.flatnonzero(labels == 0)
    same = buckets[pos][:, None] == buckets[neg][None, :]
    greater = (scores[pos][:, None] > scores[neg][None, :]) + 0.5 * (scores[pos][:, None] == scores[neg][None, :])
    assert parts["cross"] == pytest.approx(greater[~same].mean(), abs=1e-9)
    assert parts["within"] == pytest.approx(greater[same].mean(), abs=1e-9)
    assert parts["within_pair_share"] == pytest.approx(same.mean(), abs=1e-9)
    assert parts["pooled"] == pytest.approx(greater.mean(), abs=1e-9)


def test_diagrams_are_written_and_missing_reports_are_skipped(tmp_path):
    pytest.importorskip("matplotlib")
    builder = FigureBuilder(
        DatasetConfig(project_root=tmp_path, dataset_name="paysim_synthetic_financial"), tmp_path / "out"
    )
    written = {path.name for path in builder.build_all()}
    assert written == {"fig1_pipeline.png", "fig2_architectures.png"}  # no reports -> chart figures skipped
    assert (tmp_path / "out" / "fig1_pipeline.svg").exists()
