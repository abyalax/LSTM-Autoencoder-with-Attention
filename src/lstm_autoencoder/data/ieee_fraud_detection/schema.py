"""Schema configuration for IEEE-CIS Fraud Detection datasets."""

from dataclasses import dataclass


@dataclass(frozen=True)
class IEEEFraudDetectionSchema:
    """Column names and schema-specific settings for IEEE-CIS Fraud Detection datasets."""

    target_column: str = "isFraud"
    timestamp_column: str = "TransactionDT"
    sender_column: str = "card1"
    receiver_column: str = "card1"
    amount_column: str = "TransactionAmt"
    transaction_type_column: str = "ProductCD"
    valid_transaction_types: tuple[str, ...] = ()  # IEEE uses all product codes
    expected_columns: tuple[str, ...] = (
        "TransactionID",
        "isFraud",
        "TransactionDT",
        "TransactionAmt",
        "ProductCD",
        "card1",
        "card2",
        "card3",
        "card4",
        "card5",
        "card6",
    )
    leakage_columns: tuple[str, ...] = ()
    reuse_columns: tuple[str, ...] = ()

    @property
    def required_columns(self) -> tuple[str, ...]:
        """Columns without which no analysis can run."""
        return (
            self.target_column,
            self.timestamp_column,
            self.sender_column,
            self.amount_column,
        )
