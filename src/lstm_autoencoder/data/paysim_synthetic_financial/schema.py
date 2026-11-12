"""Schema configuration for PaySim synthetic financial datasets."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PaySimSchema:
    """Column names and schema-specific settings for PaySim synthetic financial datasets."""

    target_column: str = "isFraud"
    timestamp_column: str = "step"
    timestamp_unit: str = "h"  # PaySim step is in hours
    sender_column: str = "nameOrig"
    receiver_column: str = "nameDest"
    amount_column: str = "amount"
    transaction_type_column: str = "type"
    valid_transaction_types: tuple[str, ...] = ("CASH_OUT", "TRANSFER")
    expected_columns: tuple[str, ...] = (
        "step",
        "type",
        "amount",
        "nameOrig",
        "nameDest",
        "isFraud",
        "isFlaggedFraud",
    )
    leakage_columns: tuple[str, ...] = (
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
    )
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
