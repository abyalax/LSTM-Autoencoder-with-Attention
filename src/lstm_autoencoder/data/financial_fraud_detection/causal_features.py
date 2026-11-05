"""Per-account features that only use information available in the past."""

from __future__ import annotations

import numpy as np
import pandas as pd

from lstm_autoencoder.config import DatasetConfig
from lstm_autoencoder.data.loader import Schema

ACCOUNT_CODE = "account_code"
HISTORY_LENGTH = "history_length"
LOG_SECONDS_SINCE_PREVIOUS = "log_seconds_since_previous"
LOG_AMOUNT_RATIO = "log_amount_ratio"
IS_NEW_RECEIVER = "is_new_receiver"
HOUR_OF_DAY = "hour_of_day"
DAY_OF_WEEK = "day_of_week"
AMOUNT_EPSILON = 1e-9

DERIVED_FEATURES: tuple[str, ...] = (
    HISTORY_LENGTH, 
    LOG_SECONDS_SINCE_PREVIOUS,
    LOG_AMOUNT_RATIO,
    IS_NEW_RECEIVER,
    HOUR_OF_DAY,
    DAY_OF_WEEK,
)


class CausalFeatureBuilder:
    """Derive features per sender account without looking at future transactions."""

    def __init__(self, config: DatasetConfig, schema: Schema | None = None) -> None:
        """Bind the builder to a dataset configuration and schema."""
        self._config = config
        self._schema = schema or Schema()

    def build(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return a copy ordered by account and time with causal features added.

        Every derived value of a transaction depends only on that transaction
        and on earlier transactions of the same account, so the same features
        can be computed in a streaming setting.
        """
        codes = pd.factorize(frame[self._schema.sender_column])[0]
        ordered = (
            frame.assign(**{ACCOUNT_CODE: codes})
            .sort_values([ACCOUNT_CODE, self._schema.timestamp_column], kind="stable")
            .reset_index(drop=True)
        )
        groups = ordered.groupby(ACCOUNT_CODE, sort=False)
        amount = ordered[self._schema.amount_column]

        history = groups.cumcount()
        ordered[HISTORY_LENGTH] = history
        seconds = groups[self._schema.timestamp_column].diff().dt.total_seconds()
        ordered[LOG_SECONDS_SINCE_PREVIOUS] = np.log1p(seconds)

        prior_total = groups[self._schema.amount_column].cumsum() - amount
        prior_mean = prior_total / history.replace(0, np.nan)
        ratio = (amount + AMOUNT_EPSILON) / (prior_mean + AMOUNT_EPSILON)
        ordered[LOG_AMOUNT_RATIO] = np.log(ratio)

        if self._schema.receiver_column in ordered.columns:
            ordered[IS_NEW_RECEIVER] = self._first_time_receiver(ordered)

        timestamps = ordered[self._schema.timestamp_column]
        ordered[HOUR_OF_DAY] = timestamps.dt.hour
        ordered[DAY_OF_WEEK] = timestamps.dt.day_of_week
        return ordered

    def feature_columns(self, built: pd.DataFrame) -> list[str]:
        """List the derived feature columns present in a built frame."""
        return [name for name in DERIVED_FEATURES if name in built.columns]

    def _first_time_receiver(self, ordered: pd.DataFrame) -> pd.Series:
        receiver_codes = pd.Series(
            pd.factorize(ordered[self._schema.receiver_column])[0],
            index=ordered.index,
        )
        keys = [ordered[ACCOUNT_CODE], receiver_codes]
        seen_before = ordered.groupby(keys, sort=False).cumcount()
        return (seen_before == 0).astype("int8")
