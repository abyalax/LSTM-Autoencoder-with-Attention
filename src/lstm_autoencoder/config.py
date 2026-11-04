"""Configuration objects for the LSTM Autoencoder research project."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetConfig:
    """Location and high-level settings of the transaction dataset."""

    project_root: Path
    dataset_name: str
    row_limit: int | None = None
    expected_fraud_rate: float = 0.04
    window_sizes: tuple[int, ...] = (1, 3, 5, 10)
    train_fraction: float = 0.70
    validation_fraction: float = 0.15
    dtype: dict[str, str] | None = None

    @property
    def raw_dir(self) -> Path:
        """Directory that holds the untouched dataset files."""
        return self.project_root / "data" / "raw" / self.dataset_name



    @property
    def artifacts_dir(self) -> Path:
        """Directory that receives generated reports."""
        return self.project_root / "artifacts"


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
