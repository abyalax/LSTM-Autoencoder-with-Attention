"""PaySim-specific causal feature builder for per-account sequence features."""

from __future__ import annotations

import numpy as np
import pandas as pd

from lstm_autoencoder.config import DatasetConfig
from lstm_autoencoder.data.paysim_synthetic_financial.schema import PaySimSchema

ACCOUNT_CODE = "account_code"
HISTORY_LENGTH = "history_length"
STEP_DIFF = "step_diff"
LOG_AMOUNT_RATIO = "log_amount_ratio"
IS_NEW_RECEIVER = "is_new_receiver"
HISTORY_BUCKET = "history_bucket"
AMOUNT_EPSILON = 1e-9

DERIVED_FEATURES: tuple[str, ...] = (
    HISTORY_LENGTH,
    STEP_DIFF,
    LOG_AMOUNT_RATIO,
    IS_NEW_RECEIVER,
    HISTORY_BUCKET,
)


class CausalFeatureBuilder:
    """Derive features per receiver account without looking at future transactions.

    PaySim-specific implementation that handles step-based timestamps rather than
    datetime-based timestamps. The step column represents discrete time steps in the
    simulation, not actual wall-clock time.
    """

    def __init__(self, config: DatasetConfig, schema: PaySimSchema | None = None) -> None:
        """Bind the builder to a dataset configuration and PaySim schema."""
        self._config = config
        self._schema = schema or PaySimSchema()

    def build(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return a copy ordered by account and step with causal features added.

        Every derived value of a transaction depends only on that transaction
        and on earlier transactions of the same account, so the same features
        can be computed in a streaming setting.
        """
        codes = pd.factorize(frame[self._schema.receiver_column])[0]
        ordered = (
            frame.assign(**{ACCOUNT_CODE: codes})
            .sort_values([ACCOUNT_CODE, self._schema.timestamp_column], kind="stable")
            .reset_index(drop=True)
        )
        groups = ordered.groupby(ACCOUNT_CODE, sort=False)
        amount = ordered[self._schema.amount_column]

        history = groups.cumcount()
        ordered[HISTORY_LENGTH] = history
        step_diff = groups[self._schema.timestamp_column].diff()
        ordered[STEP_DIFF] = step_diff.fillna(0).astype("int32")

        prior_total = groups[self._schema.amount_column].cumsum() - amount
        prior_mean = prior_total / history.replace(0, np.nan)
        ratio = (amount + AMOUNT_EPSILON) / (prior_mean + AMOUNT_EPSILON)
        ordered[LOG_AMOUNT_RATIO] = np.log(ratio).fillna(0)

        # is_new_receiver: receiver has no prior transactions (history_len == 0)
        ordered[IS_NEW_RECEIVER] = (history == 0).astype("int8")

        # history_bucket: categorical grouping for stratified reporting
        history_bucket_bins = [0, 1, 4, 9, np.inf]
        history_bucket_labels = ["1", "2-4", "5-9", ">=10"]
        ordered[HISTORY_BUCKET] = pd.cut(
            history,
            bins=history_bucket_bins,
            labels=history_bucket_labels,
            right=False,
        ).astype(str)

        return ordered

    def feature_columns(self, built: pd.DataFrame) -> list[str]:
        """List the derived feature columns present in a built frame."""
        return [name for name in DERIVED_FEATURES if name in built.columns]
