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
    mask: np.ndarray  # (n, window_length), 1 for real steps (first `length` columns), 0 for padding
    labels: np.ndarray  # (n,), isFraud of the current transaction
    history_bucket: np.ndarray  # (n,), history_bucket label of the current transaction
    receiver: np.ndarray  # (n,), receiver id of the current transaction, for receiver-level splitting


class SequenceBuilder:
    """Build one example per transaction: the transaction itself plus its receiver's recent past.

    PaySim's `step` is hourly, so the order of same-step transactions of a receiver is
    not observable and is never used as chronology. A window therefore holds

        [up to W-1 most recent transactions at STRICTLY EARLIER steps, oldest first,
         then the current transaction]

    Same-step peers of the current transaction are excluded, so the window never
    contains a transaction whose chronological relation to the current one is unknown,
    and it is identical for every ordering of the same-step rows.

    Layout is RIGHT-padded: real steps occupy columns `0 .. length-1`, the current
    transaction sits at column `length-1`, and the remaining columns are zero with
    mask 0. The encoder packs sequences by `length` (see `Encoder`), so padding never
    enters the recurrent state.

    An example is emitted for every transaction that has at least
    `min_history_length - 1` transactions at earlier steps (`min_history_length=1`
    keeps all of them).
    """

    def __init__(self, config: SequenceConfig | None = None, schema: PaySimSchema | None = None) -> None:
        """Bind the builder to a sequence configuration and PaySim schema."""
        self._config = config or SequenceConfig()
        self._schema = schema or PaySimSchema()

    def build(self, built: pd.DataFrame) -> SequenceBatch:
        """Build sequences from a frame already processed by CausalFeatureBuilder.

        `built` must be sorted by (account, step), which CausalFeatureBuilder guarantees.
        """
        window = self._config.window_length
        feature_matrix = built[list(self._config.feature_columns)].to_numpy(dtype="float32")
        labels_all = built[self._schema.target_column].to_numpy()
        buckets_all = built["history_bucket"].to_numpy()
        receivers_all = built[self._schema.receiver_column].to_numpy()
        codes = built[ACCOUNT_CODE].to_numpy()
        step = built[self._schema.timestamp_column].to_numpy()

        n_rows = len(built)
        position = np.arange(n_rows)
        new_account = np.ones(n_rows, dtype=bool)
        new_step = np.ones(n_rows, dtype=bool)
        if n_rows > 1:
            new_account[1:] = codes[1:] != codes[:-1]
            new_step[1:] = new_account[1:] | (step[1:] != step[:-1])
        account_start = np.maximum.accumulate(np.where(new_account, position, 0))
        step_start = np.maximum.accumulate(np.where(new_step, position, 0))
        earlier = step_start - account_start  # transactions of this receiver at strictly earlier steps

        eligible = np.flatnonzero(earlier >= self._config.min_history_length - 1)
        prior_count = np.minimum(earlier[eligible], window - 1)
        length = prior_count + 1
        first_prior = step_start[eligible] - prior_count

        col = np.arange(window)[None, :]
        real = col < length[:, None]
        source_row = np.where(col < prior_count[:, None], first_prior[:, None] + col, eligible[:, None])
        source_row = np.where(real, source_row, 0)

        sequences = np.where(real[:, :, None], feature_matrix[source_row], 0.0).astype("float32")
        mask = real.astype("float32")

        return SequenceBatch(
            sequences=sequences,
            mask=mask,
            labels=labels_all[eligible],
            history_bucket=buckets_all[eligible],
            receiver=receivers_all[eligible],
        )
