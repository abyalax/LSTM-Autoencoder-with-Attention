"""Schema configuration for financial fraud detection dataset."""

from dataclasses import dataclass


@dataclass(frozen=True)
class FinancialFraudDetectionSchema:
    """Column names and schema-specific settings for financial fraud detection dataset."""

    target_column: str = "is_fraud"
    timestamp_column: str = "timestamp"
    sender_column: str = "sender_account"
    receiver_column: str = "receiver_account"
    amount_column: str = "amount"
    id_column: str = "transaction_id"
    expected_columns: tuple[str, ...] = (
        "transaction_id",
        "timestamp",
        "sender_account",
        "receiver_account",
        "amount",
        "transaction_type",
        "merchant_category",
        "location",
        "device_used",
        "is_fraud",
    )
    leakage_columns: tuple[str, ...] = ("fraud_type",)
    reuse_columns: tuple[str, ...] = ("device_hash", "ip_address")

    @property
    def required_columns(self) -> tuple[str, ...]:
        """Columns without which no analysis can run."""
        return (
            self.target_column,
            self.timestamp_column,
            self.sender_column,
            self.amount_column,
        )
