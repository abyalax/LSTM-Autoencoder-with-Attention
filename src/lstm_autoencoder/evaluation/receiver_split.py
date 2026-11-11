"""Label-free receiver-level train/validation/test split, shared by every workflow."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from lstm_autoencoder.evaluation.config import SplitConfig

SIZE_BINS: tuple[float, ...] = (0, 1, 4, 9, np.inf)
SIZE_LABELS: tuple[str, ...] = ("1", "2-4", "5-9", ">=10")


class ReceiverSplitter:
    """Assign whole receivers to train/val/test using NO label information.

    Earlier versions stratified receivers by their own fraud rate, i.e. the split
    itself read `isFraud`. This splitter only looks at the receiver identifier and,
    optionally (`SplitConfig.stratify_by="receiver_size"`), at how many transactions
    the receiver has -- a label-free quantity that keeps the history-length mix
    similar across splits. It is deterministic for a fixed seed and its result does
    not change if every label is altered (asserted in the tests and the pipeline audit).
    """

    def __init__(self, split_config: SplitConfig | None = None, random_seed: int = 42) -> None:
        """Bind the splitter to split fractions and a seed."""
        self._config = split_config or SplitConfig()
        self._seed = random_seed

    def split(self, receivers: pd.Series) -> dict[str, pd.Index]:
        """Return `{"train","val","test"}` receiver ids, given one receiver id per transaction row."""
        counts = receivers.value_counts().sort_index()
        ids = counts.index
        strata = self._strata(counts)

        train_ids, rest_ids = train_test_split(
            ids,
            test_size=1 - self._config.train_fraction,
            random_state=self._seed,
            stratify=strata,
        )
        val_ratio = self._config.validation_fraction / (1 - self._config.train_fraction)
        val_ids, test_ids = train_test_split(
            rest_ids,
            test_size=1 - val_ratio,
            random_state=self._seed,
            stratify=None if strata is None else strata.loc[rest_ids],
        )
        return {"train": pd.Index(train_ids), "val": pd.Index(val_ids), "test": pd.Index(test_ids)}

    def split_rows(self, receivers: pd.Series) -> dict[str, np.ndarray]:
        """Row positions (into `receivers`) of each split."""
        groups = self.split(receivers)
        return {name: np.flatnonzero(receivers.isin(ids).to_numpy()) for name, ids in groups.items()}

    def _strata(self, counts: pd.Series) -> pd.Series | None:
        """Label-free strata (receiver transaction-count bucket), or None for a plain random split."""
        if self._config.stratify_by == "none":
            return None
        return pd.cut(counts, bins=SIZE_BINS, labels=SIZE_LABELS, right=True).astype(str)
