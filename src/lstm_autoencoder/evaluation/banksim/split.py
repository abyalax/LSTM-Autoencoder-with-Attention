"""Single customer-level split shared by every BankSim workflow.

The ablation baselines, the LSTM-AE training pipeline and the hybrid evaluation must
score the SAME rows, so they all take their split from here instead of building it
independently (an earlier version split the AE pipeline by merchant and the baselines
by customer, which made their numbers incomparable).
"""

from __future__ import annotations

import pandas as pd

from lstm_autoencoder.evaluation.config import SplitConfig
from lstm_autoencoder.evaluation.receiver_split import ReceiverSplitter

# Plain random customer split: BankSim customers all have >= 5 transactions, so size
# stratification buys nothing, and a plain split never needs a fallback for tiny strata.
BANKSIM_SPLIT_CONFIG = SplitConfig(stratify_by="none")
BANKSIM_SPLIT_SEED = 42


def split_customers(
    customers: pd.Series,
    split_config: SplitConfig | None = None,
    seed: int = BANKSIM_SPLIT_SEED,
) -> dict[str, pd.Index]:
    """Assign whole customers to train/val/test without reading any label.

    `customers` holds one customer id per transaction row. Every row of a customer
    lands in the same split, so a customer's history never crosses the boundary.
    """
    return ReceiverSplitter(split_config or BANKSIM_SPLIT_CONFIG, random_seed=seed).split(customers)
