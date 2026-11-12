"""Loading and basic cleaning of the raw transaction dataset."""

from __future__ import annotations

import zipfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from lstm_autoencoder.config import DatasetConfig
from lstm_autoencoder.data.dataset_files import DATASET_FILES


class Schema:
    """Base schema class for column name mapping."""

    target_column: str = "target"
    timestamp_column: str = "timestamp"
    sender_column: str = "sender"
    receiver_column: str = "receiver"
    amount_column: str = "amount"
    id_column: str = "id"

    @property
    def required_columns(self) -> tuple[str, ...]:
        """Columns without which no analysis can run."""
        return (
            self.target_column,
            self.timestamp_column,
            self.sender_column,
            self.amount_column,
        )


TARGET_LABELS: dict[str, float] = {
    "true": 1.0,
    "false": 0.0,
    "1": 1.0,
    "0": 0.0,
    "1.0": 1.0,
    "0.0": 0.0,
    "yes": 1.0,
    "no": 0.0,
}


@dataclass(frozen=True)
class LoadedDataset:
    """A cleaned dataset together with what the cleaning removed."""

    frame: pd.DataFrame
    source: Path
    n_raw_rows: int
    n_unreadable_target: int
    n_unreadable_timestamp: int
    n_filtered_by_scope: int = 0


class DatasetLoader:
    """Read the raw dataset and normalize the target and timestamp columns."""

    def __init__(self, config: DatasetConfig, schema: Schema | None = None) -> None:
        """Bind the loader to a dataset configuration and schema."""
        self._config = config
        self._schema = schema or Schema()

    def load(self) -> LoadedDataset:
        """Load the dataset from the CSV file, or from the zip if no CSV exists.

        Rows whose target or timestamp cannot be read are dropped and counted.
        Raises FileNotFoundError or ValueError when the data is unusable.
        """
        raw, source = self._read_raw()
        self._validate_columns(raw)

        target = self._normalize_target(raw[self._schema.target_column])
        timestamps = self._parse_timestamps(raw[self._schema.timestamp_column])
        valid = target.notna() & timestamps.notna()

        frame = raw if bool(valid.all()) else raw.loc[valid].copy()
        frame[self._schema.target_column] = target.loc[frame.index].astype("int8")
        frame[self._schema.timestamp_column] = timestamps.loc[frame.index]

        # Apply dataset-specific scope filter if applicable
        n_filtered_by_scope = 0
        if hasattr(self._schema, "valid_transaction_types") and hasattr(self._schema, "transaction_type_column"):
            if self._schema.transaction_type_column in frame.columns and self._schema.valid_transaction_types:
                n_before_filter = len(frame)
                frame = frame[
                    frame[self._schema.transaction_type_column].isin(self._schema.valid_transaction_types)
                ].copy()
                n_filtered_by_scope = n_before_filter - len(frame)

        return LoadedDataset(
            frame=frame,
            source=source,
            n_raw_rows=len(raw),
            n_unreadable_target=int(target.isna().sum()),
            n_unreadable_timestamp=int(timestamps.isna().sum()),
            n_filtered_by_scope=n_filtered_by_scope,
        )

    def _read_raw(self) -> tuple[pd.DataFrame, Path]:
        config = self._config
        try:
            spec = DATASET_FILES[config.dataset_name]
        except KeyError as exc:
            supported = sorted(DATASET_FILES)
            raise ValueError(
                f"Unsupported dataset: {config.dataset_name!r}. Supported: {supported}"
            ) from exc
        dtype = config.dtype if config.dtype is not None else spec.dtype
        csv_path = config.raw_dir / spec.csv_name
        zip_path = config.raw_dir / spec.archive_name

        if csv_path.exists():
            frame = pd.read_csv(csv_path, nrows=config.row_limit, low_memory=False, dtype=dtype)
            return frame, csv_path
        if zip_path.exists():
            return self._read_zip(zip_path, spec.archive_member, dtype), zip_path
        raise FileNotFoundError(f"Dataset not found. Expected {csv_path} or {zip_path}.")

    def _read_zip(
        self, zip_path: Path, archive_member: str, dtype: dict[str, str] | None = None
    ) -> pd.DataFrame:
        with zipfile.ZipFile(zip_path) as archive:
            names = archive.namelist()
            matches = [name for name in names if name == archive_member or name.endswith("/" + archive_member)]
            if len(matches) != 1:
                raise ValueError(
                    f"Expected one {archive_member!r} member in {zip_path}, found {len(matches)}."
                )
            with archive.open(matches[0]) as handle:
                return pd.read_csv(handle, nrows=self._config.row_limit, low_memory=False, dtype=dtype)

    def _validate_columns(self, frame: pd.DataFrame) -> None:
        missing = [c for c in self._schema.required_columns if c not in frame.columns]
        if missing:
            raise ValueError(f"Required columns are missing from dataset: {missing}")

    @staticmethod
    def _normalize_target(series: pd.Series) -> pd.Series:
        if pd.api.types.is_bool_dtype(series) or pd.api.types.is_numeric_dtype(series):
            numeric = series.astype("float64")
            return numeric.where(numeric.isin([0.0, 1.0]))
        text = series.astype(str).str.strip().str.lower()
        return text.map(TARGET_LABELS)

    @staticmethod
    def _parse_timestamps(series: pd.Series) -> pd.Series:
        if pd.api.types.is_numeric_dtype(series):
            return series
        parsed = pd.to_datetime(series, format="ISO8601", errors="coerce", utc=True)
        return parsed.dt.tz_localize(None)