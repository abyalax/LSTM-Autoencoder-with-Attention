"""Univariate signal metrics for PaySim synthetic financial datasets.

This module provides PaySim-specific configuration for the common UnivariateSignalChecker.
"""

from __future__ import annotations

import pandas as pd

from lstm_autoencoder.evaluation.metrics import (
    SignalCheckResult,
)
from lstm_autoencoder.evaluation.metrics import (
    UnivariateSignalChecker as BaseUnivariateSignalChecker,
)

# Columns that encode post-transaction account state; not causally available
# before the outcome, so they are excluded as leakage candidates.
LEAKAGE_CANDIDATE_COLUMNS: tuple[str, ...] = (
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
)

# Below this, a feature's AUC is considered indistinguishable from random.
SIGNAL_THRESHOLD = 0.02


class UnivariateSignalChecker(BaseUnivariateSignalChecker):
    """PaySim-specific univariate signal checker with predefined configuration."""

    def __init__(self) -> None:
        """Initialize with PaySim-specific signal threshold."""
        super().__init__(signal_threshold=SIGNAL_THRESHOLD)

    def check_univariate_signal(
        self,
        df: pd.DataFrame,
        target_column: str = "isFraud",
        exclude_leakage: bool = False,
    ) -> list[SignalCheckResult]:
        """Compute univariate AUC per numeric feature, with or without leakage columns.

        Args:
            df: Input dataframe
            target_column: Name of the target column
            exclude_leakage: Whether to exclude leakage candidate columns

        Returns:
            List of SignalCheckResult sorted by signal strength
        """
        exclude_columns = ("isFlaggedFraud",)
        if exclude_leakage:
            exclude_columns = exclude_columns + LEAKAGE_CANDIDATE_COLUMNS

        return super().check_univariate_signal(
            df=df,
            target_column=target_column,
            exclude_columns=exclude_columns,
            leakage_candidate_columns=LEAKAGE_CANDIDATE_COLUMNS,
        )

