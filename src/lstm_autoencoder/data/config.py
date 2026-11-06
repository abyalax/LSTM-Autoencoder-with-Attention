"""Configuration objects for data exploration and preprocessing."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExploratoryConfig:
    """Settings for exploratory data analysis."""

    expected_fraud_rate: float = 0.04
    window_sizes: tuple[int, ...] = (1, 3, 5, 10)
    feature_columns: tuple[str, ...] = (
        "amount",
        "step_diff",
        "log_amount_ratio",
        "history_length",
        "is_new_receiver",
    )
    min_history_length: int = 1
