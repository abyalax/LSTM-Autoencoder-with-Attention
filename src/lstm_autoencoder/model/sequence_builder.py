"""Turn causal-feature-built transactions into fixed-length per-receiver sequences."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from lstm_autoencoder.data.paysim_synthetic_financial.causal_features import ACCOUNT_CODE
from lstm_autoencoder.data.paysim_synthetic_financial.schema import PaySimSchema
from lstm_autoencoder.model.config import SequenceConfig


@dataclass(frozen=True)
class SequenceBatch:
    """Padded sequences with their masks, labels, receivers, and history buckets."""

    sequences: np.ndarray  # (n, window_length, n_features), float32
    mask: np.ndarray  # (n, window_length), 1 for real steps, 0 for padding
    labels: np.ndarray  # (n,), last-step isFraud
    history_bucket: np.ndarray  # (n,), last-step history_bucket label
    receiver: np.ndarray  # (n,), last-step receiver id, for receiver-level splitting


class SequenceBuilder:
    """Build the last `window_length` transactions per receiver as one training example.

    One example is emitted per transaction that has at least `min_history_length`
    prior transactions on the same receiver; when a receiver's history is shorter
    than the window, the earlier steps are left-padded with zeros.
    """

    def __init__(self, config: SequenceConfig | None = None, schema: PaySimSchema | None = None) -> None:
        """Bind the builder to a sequence configuration and PaySim schema."""
        self._config = config or SequenceConfig()
        self._schema = schema or PaySimSchema()

    def build(self, built: pd.DataFrame) -> SequenceBatch:
        """Build sequences from a frame already processed by CausalFeatureBuilder.

        `built` must be sorted by (receiver, step), which CausalFeatureBuilder guarantees.
        """
        window = self._config.window_length
        feature_matrix = built[list(self._config.feature_columns)].to_numpy(dtype="float32")
        labels_all = built[self._schema.target_column].to_numpy()
        buckets_all = built["history_bucket"].to_numpy()
        receivers_all = built[self._schema.receiver_column].to_numpy()
        codes = built[ACCOUNT_CODE].to_numpy()
        history = built.groupby(ACCOUNT_CODE).cumcount().to_numpy()

        eligible = np.flatnonzero(history >= self._config.min_history_length - 1)

        # codes is non-decreasing because CausalFeatureBuilder sorts by [ACCOUNT_CODE, step],
        # so the first occurrence of each code gives that account's group-start row.
        unique_codes, first_index = np.unique(codes, return_index=True)
        group_start = np.zeros(int(unique_codes.max()) + 1, dtype="int64")
        group_start[unique_codes] = first_index

        end = eligible
        starts = np.maximum(group_start[codes[end]], end - window + 1)
        length = end - starts + 1
        pad_len = window - length

        col = np.arange(window)[None, :]
        real = col >= pad_len[:, None]
        source_row = np.where(real, starts[:, None] + (col - pad_len[:, None]), 0)

        sequences = np.where(real[:, :, None], feature_matrix[source_row], 0.0).astype("float32")
        mask = real.astype("float32")

        return SequenceBatch(
            sequences=sequences,
            mask=mask,
            labels=labels_all[eligible],
            history_bucket=buckets_all[eligible],
            receiver=receivers_all[eligible],
        )
