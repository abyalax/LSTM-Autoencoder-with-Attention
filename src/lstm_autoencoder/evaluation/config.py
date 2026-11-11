"""Configuration objects for evaluation and signal probing."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SplitConfig:
    """Settings for train/validation/test split."""

    train_fraction: float = 0.70
    validation_fraction: float = 0.15
    # Receivers are assigned to train/val/test without ever looking at labels.
    # "receiver_size" balances the number of transactions per receiver across splits
    # (label-free); "none" is a plain random receiver split.
    stratify_by: str = "receiver_size"

    def __post_init__(self) -> None:
        """Validate the stratification mode."""
        if self.stratify_by not in ("receiver_size", "none"):
            raise ValueError(f"stratify_by must be 'receiver_size' or 'none', got {self.stratify_by!r}.")


@dataclass(frozen=True)
class ProbeConfig:
    """Settings for the supervised check of whether labels are learnable."""

    max_train_rows: int = 1_500_000
    max_iterations: int = 300
    learning_rate: float = 0.08
    max_leaf_nodes: int = 63
    max_categories: int = 50
    random_seed: int = 42
    no_signal_auc: float = 0.51
    weak_signal_auc: float = 0.60
    no_signal_lift: float = 1.15
