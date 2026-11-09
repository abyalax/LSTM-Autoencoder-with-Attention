"""Fast paired bootstrap for ranking metrics, via integer row weights instead of re-sorting every draw."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class _RankedScores:
    """One score vector, sorted once and reused by every bootstrap draw."""

    desc_order: np.ndarray  # row indices of the finite rows, score descending (stable)
    desc_labels: np.ndarray  # float 0/1 labels in that order
    group_ends: np.ndarray  # last position of every tie group in the descending order
    neg_index: np.ndarray  # finite negatives, score ascending
    neg_values: np.ndarray
    pos_index: np.ndarray  # finite positives, score ascending
    pos_values: np.ndarray


class WeightedRankingBootstrap:
    """AUC, average precision and recall@FPR of many score vectors under integer row weights.

    A bootstrap draw is a multiset of rows. Instead of materialising it (a fresh
    gather and sort of every score vector per draw), the draw is expressed as an
    integer weight per row: a row drawn k times has weight k. Each score vector is
    sorted once in the constructor, and every draw then only needs weighted
    cumulative sums in that fixed order. For integer weights the results equal
    `RankingMetrics.roc_auc`, `average_precision` and `recall_at_fpr` computed on
    the materialised resample (see `tests/test_bootstrap.py`), but a draw costs
    O(n) per score vector instead of O(n log n) several times over.

    All score vectors share the labels and the weights of a draw, so differences
    between them are paired by construction.
    """

    def __init__(
        self,
        labels: np.ndarray,
        scores: Mapping[str, np.ndarray],
        fpr_targets: Sequence[float] = (0.01, 0.001),
    ) -> None:
        """Sort every score vector once; non-finite scores are ignored, as in `RankingMetrics`."""
        self._labels = np.asarray(labels).astype("int64")
        self._n = len(self._labels)
        self._fpr_targets = tuple(fpr_targets)
        self._ranked: dict[str, _RankedScores] = {}
        for name, values in scores.items():
            values = np.asarray(values, dtype="float64")
            if len(values) != self._n:
                raise ValueError(f"Scores '{name}' have {len(values)} rows, labels have {self._n}.")
            self._ranked[name] = self._rank(values, self._labels)

    @property
    def names(self) -> tuple[str, ...]:
        """Names of the score vectors, in insertion order."""
        return tuple(self._ranked)

    def metrics(self, weights: np.ndarray) -> dict[str, dict[str, float]]:
        """Return {name: {"auc", "ap", "recall_at_fpr_<fpr>"}} for one draw.

        `weights` must hold non-negative INTEGERS (draw multiplicities) per row.
        Metrics are NaN when a draw contains no positive or no negative.
        """
        weights = np.asarray(weights, dtype="float64")
        if len(weights) != self._n:
            raise ValueError(f"weights have {len(weights)} rows, labels have {self._n}.")
        out: dict[str, dict[str, float]] = {}
        for name, ranked in self._ranked.items():
            auc, ap = self._auc_and_ap(ranked, weights)
            row = {"auc": auc, "ap": ap}
            if self._fpr_targets:
                cum_neg = np.cumsum(weights[ranked.neg_index])  # shared by every FPR target
                cum_pos = np.cumsum(weights[ranked.pos_index])
                for fpr in self._fpr_targets:
                    row[f"recall_at_fpr_{fpr:g}"] = self._recall_at_fpr(ranked, cum_neg, cum_pos, fpr)
            out[name] = row
        return out

    @staticmethod
    def cluster_weights(inverse: np.ndarray, n_clusters: int, rng: np.random.Generator) -> np.ndarray:
        """Row weights of one cluster bootstrap draw: `n_clusters` clusters drawn with replacement.

        `inverse[i]` is the cluster (e.g. receiver) of row i, in 0..n_clusters-1. Every
        row of a drawn cluster travels with it, so rows of one receiver stay dependent.
        """
        counts = np.bincount(rng.integers(0, n_clusters, size=n_clusters), minlength=n_clusters)
        return counts[inverse].astype("float64")

    @staticmethod
    def stratified_row_weights(labels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Row weights of one class-stratified row bootstrap draw (every draw keeps the class counts)."""
        labels = np.asarray(labels)
        positives = np.flatnonzero(labels == 1)
        negatives = np.flatnonzero(labels == 0)
        index = np.concatenate([rng.choice(positives, size=len(positives)), rng.choice(negatives, size=len(negatives))])
        return np.bincount(index, minlength=len(labels)).astype("float64")

    @staticmethod
    def _rank(values: np.ndarray, labels: np.ndarray) -> _RankedScores:
        """Sort one score vector once and precompute everything the draws reuse."""
        finite = np.flatnonzero(np.isfinite(values))
        finite_values = values[finite]
        desc = finite[np.argsort(-finite_values, kind="mergesort")]
        desc_values = values[desc]
        boundaries = np.flatnonzero(np.diff(desc_values))
        group_ends = np.append(boundaries, len(desc) - 1)

        asc = finite[np.argsort(finite_values, kind="mergesort")]
        asc_labels = labels[asc]
        neg_index = asc[asc_labels == 0]
        pos_index = asc[asc_labels == 1]
        return _RankedScores(
            desc_order=desc,
            desc_labels=labels[desc].astype("float64"),
            group_ends=group_ends,
            neg_index=neg_index,
            neg_values=values[neg_index],
            pos_index=pos_index,
            pos_values=values[pos_index],
        )

    @staticmethod
    def _auc_and_ap(ranked: _RankedScores, weights: np.ndarray) -> tuple[float, float]:
        """Weighted ROC AUC (trapezoid over tie groups) and average precision (step-wise, tie groups)."""
        if len(ranked.desc_order) == 0:
            return float("nan"), float("nan")
        weighted = weights[ranked.desc_order]
        positive = weighted * ranked.desc_labels
        cum_pos = np.cumsum(positive)[ranked.group_ends]
        cum_neg = np.cumsum(weighted - positive)[ranked.group_ends]
        n_pos, n_neg = cum_pos[-1], cum_neg[-1]
        if n_pos <= 0 or n_neg <= 0:
            return float("nan"), float("nan")
        tpr = cum_pos / n_pos
        fpr = cum_neg / n_neg
        prev_tpr = np.concatenate(([0.0], tpr[:-1]))
        prev_fpr = np.concatenate(([0.0], fpr[:-1]))
        auc = float(np.sum((fpr - prev_fpr) * (tpr + prev_tpr) / 2.0))
        predicted = cum_pos + cum_neg
        precision = np.divide(cum_pos, predicted, out=np.zeros_like(predicted), where=predicted > 0)
        ap = float(np.sum((tpr - prev_tpr) * precision))
        return auc, ap

    @staticmethod
    def _recall_at_fpr(ranked: _RankedScores, cum_neg: np.ndarray, cum_pos: np.ndarray, fpr: float) -> float:
        """Weighted `RankingMetrics.recall_at_fpr`: strict '>' against the (1 - fpr) quantile of negatives.

        `cum_neg` / `cum_pos` are the cumulative draw weights of the negatives / positives in ascending score order.
        """
        if len(cum_neg) == 0 or len(cum_pos) == 0:
            return float("nan")
        n_neg, n_pos = cum_neg[-1], cum_pos[-1]
        if n_neg <= 0 or n_pos <= 0:
            return float("nan")
        # np.quantile(method="linear") on the expanded sample: virtual index (n - 1) * q.
        virtual = (n_neg - 1.0) * (1.0 - fpr)
        lower = int(np.floor(virtual))
        upper = min(lower + 1, int(n_neg) - 1)
        fraction = virtual - lower
        low_value = ranked.neg_values[np.searchsorted(cum_neg, lower, side="right")]
        high_value = ranked.neg_values[np.searchsorted(cum_neg, upper, side="right")]
        gap = high_value - low_value
        # numpy's own lerp, so the threshold is bit-identical to np.quantile.
        threshold = high_value - gap * (1.0 - fraction) if fraction >= 0.5 else low_value + gap * fraction
        below_or_equal = int(np.searchsorted(ranked.pos_values, threshold, side="right"))
        weight_at_or_below = cum_pos[below_or_equal - 1] if below_or_equal > 0 else 0.0
        return float((n_pos - weight_at_or_below) / n_pos)
