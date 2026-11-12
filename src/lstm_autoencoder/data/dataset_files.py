"""Raw-file metadata for supported datasets."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DatasetFileSpec:
    """Names and CSV parsing options for one raw dataset."""

    csv_name: str
    archive_name: str
    archive_member: str
    dtype: dict[str, str] = field(default_factory=dict)


DATASET_FILES: dict[str, DatasetFileSpec] = {
    "banksim": DatasetFileSpec(
        csv_name="bs140513_032310.csv",
        archive_name="banksim.zip",
        archive_member="bs140513_032310.csv",
        dtype={"customer": "string", "merchant": "string", "category": "string"},
    ),
    "financial_fraud_detection": DatasetFileSpec(
        csv_name="financial_fraud_detection.csv",
        archive_name="financial_fraud_detection.zip",
        archive_member="financial_fraud_detection.csv",
    ),
    "ieee_fraud_detection": DatasetFileSpec(
        csv_name="train_transaction.csv",
        archive_name="ieee_fraud_detection.zip",
        archive_member="train_transaction.csv",
        dtype={"card1": "string", "ProductCD": "string"},
    ),
    "paysim_synthetic_financial": DatasetFileSpec(
        csv_name="paysim_synthetic_financial.csv",
        archive_name="paysim_synthetic_financial.zip",
        archive_member="paysim_synthetic_financial.csv",
        dtype={"nameOrig": "string", "nameDest": "string", "type": "string"},
    ),
}
