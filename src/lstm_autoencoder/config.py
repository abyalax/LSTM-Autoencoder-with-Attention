"""Configuration objects for the LSTM Autoencoder research project."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetConfig:
    """High-level settings of the transaction dataset."""

    project_root: Path
    dataset_name: str
    row_limit: int | None = None
    dtype: dict[str, str] | None = None

    @property
    def raw_dir(self) -> Path:
        """Directory that holds the untouched dataset files."""
        return self.project_root / "data" / "raw" / self.dataset_name

    @property
    def artifacts_dir(self) -> Path:
        """Directory that receives generated reports."""
        return self.project_root / "artifacts"
