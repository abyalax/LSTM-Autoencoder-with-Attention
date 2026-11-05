"""Reusable metrics for evaluation across domains."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SignalCheckResult:
    """Univariate AUC-like signal strength for one feature."""

    feature: str
    auc: float
    is_leakage_candidate: bool


class RankingMetrics:
    """Threshold-free metrics computed from scores and binary labels."""

    @staticmethod
    def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
        """Return the area under the ROC curve, where 0.5 means no signal.

        Non-finite scores are ignored. Returns NaN when a class is missing.
        """
        values, truth = RankingMetrics._finite_pairs(scores, labels)
        n_positive = int((truth == 1).sum())
        n_negative = int((truth == 0).sum())
        if n_positive == 0 or n_negative == 0:
            return float("nan")
        ranks = pd.Series(values).rank(method="average").to_numpy()
        rank_sum = ranks[truth == 1].sum()
        u_statistic = rank_sum - n_positive * (n_positive + 1) / 2
        return float(u_statistic / (n_positive * n_negative))

    @staticmethod
    def average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
        """Return average precision, whose random baseline is the prevalence.

        Tied scores are grouped into one threshold. Returns NaN without positives.
        """
        values, truth = RankingMetrics._finite_pairs(scores, labels)
        n_positive = int((truth == 1).sum())
        if n_positive == 0:
            return float("nan")
        order = np.argsort(-values, kind="mergesort")
        sorted_values = values[order]
        sorted_truth = truth[order]
        boundaries = np.flatnonzero(np.diff(sorted_values))
        last_index = np.append(boundaries, len(sorted_values) - 1)
        true_positives = np.cumsum(sorted_truth)[last_index]
        predicted = last_index + 1
        precision = true_positives / predicted
        recall = true_positives / n_positive
        recall_steps = np.diff(np.concatenate(([0.0], recall)))
        return float(np.sum(recall_steps * precision))

    @staticmethod
    def recall_at_fpr(scores: np.ndarray, labels: np.ndarray, fpr: float) -> float:
        """Return recall at the threshold that lets through `fpr` of the negatives.

        The threshold is the (1 - fpr) quantile of negative scores and positives
        must score strictly above it, so ties never push the real FPR above
        `fpr`. Returns NaN when a class is missing.
        """
        values, truth = RankingMetrics._finite_pairs(scores, labels)
        negatives = values[truth == 0]
        positives = values[truth == 1]
        if len(negatives) == 0 or len(positives) == 0:
            return float("nan")
        threshold = np.quantile(negatives, 1.0 - fpr)
        return float((positives > threshold).mean())

    @staticmethod
    def _finite_pairs(
        scores: np.ndarray, labels: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        values = np.asarray(scores, dtype="float64")
        truth = np.asarray(labels)
        mask = np.isfinite(values)
        return values[mask], truth[mask]


class UnivariateSignalChecker:
    """Compute univariate AUC for numeric features using Mann-Whitney U statistic."""

    def __init__(self, signal_threshold: float = 0.02) -> None:
        """Initialize with configurable signal threshold.

        Args:
            signal_threshold: Below this, a feature's AUC is considered
                indistinguishable from random.
        """
        self.signal_threshold = signal_threshold

    @staticmethod
    def rank_auc(values: np.ndarray, target: np.ndarray) -> float:
        """Mann-Whitney U based AUC; avoids training a model just to check signal."""
        ranks = pd.Series(values).rank().to_numpy()
        n_pos = int(target.sum())
        n_neg = len(target) - n_pos
        if n_pos == 0 or n_neg == 0:
            return float("nan")
        sum_ranks_pos = ranks[target == 1].sum()
        u_statistic = sum_ranks_pos - n_pos * (n_pos + 1) / 2
        return float(u_statistic / (n_pos * n_neg))

    def check_univariate_signal(
        self,
        df: pd.DataFrame,
        target_column: str = "isFraud",
        exclude_columns: tuple[str, ...] = (),
        leakage_candidate_columns: tuple[str, ...] = (),
    ) -> list[SignalCheckResult]:
        """Compute univariate AUC per numeric feature, with optional exclusions.

        Args:
            df: Input dataframe
            target_column: Name of the target column
            exclude_columns: Columns to exclude from analysis
            leakage_candidate_columns: Columns marked as potential leakage

        Returns:
            List of SignalCheckResult sorted by signal strength
        """
        numeric_columns = [
            column
            for column in df.select_dtypes(include="number").columns
            if column not in (target_column, *exclude_columns)
        ]

        target = df[target_column].to_numpy()
        results = [
            SignalCheckResult(
                feature=column,
                auc=self.rank_auc(df[column].to_numpy(), target),
                is_leakage_candidate=column in leakage_candidate_columns,
            )
            for column in numeric_columns
        ]
        return sorted(results, key=lambda r: abs(r.auc - 0.5), reverse=True)

    def has_usable_signal(self, results: list[SignalCheckResult]) -> bool:
        """Check if any feature shows usable signal above the threshold."""
        max_gap = max((abs(r.auc - 0.5) for r in results), default=0.0)
        return max_gap >= self.signal_threshold