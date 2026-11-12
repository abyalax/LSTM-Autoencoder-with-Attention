"""Schema configuration for BankSim datasets."""

from dataclasses import dataclass


@dataclass(frozen=True)
class BankSimSchema:
    """Column names and schema-specific settings for BankSim datasets."""

    target_column: str = "fraud"
    timestamp_column: str = "step"
    sender_column: str = "customer"
    receiver_column: str = "merchant"
    amount_column: str = "amount"
    transaction_type_column: str = "category"
    valid_transaction_types: tuple[str, ...] = ()  # BankSim uses all categories
    expected_columns: tuple[str, ...] = (
        "step",
        "customer",
        "age",
        "gender",
        "zipcodeOri",
        "merchant",
        "zipMerchant",
        "category",
        "amount",
        "fraud",
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
