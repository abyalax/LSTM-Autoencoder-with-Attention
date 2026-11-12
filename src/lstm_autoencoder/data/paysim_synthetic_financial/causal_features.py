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
# Role-aware account identity: the account under analysis is the DESTINATION of a TRANSFER
# (funds arriving) and the ORIGIN of a CASH_OUT (funds leaving), so a mule's TRANSFER-in and
# later CASH_OUT-out belong to one account history.
CASH_OUT_TYPE = "CASH_OUT"

DERIVED_FEATURES: tuple[str, ...] = (
    HISTORY_LENGTH,
    STEP_DIFF,
    LOG_AMOUNT_RATIO,
    IS_NEW_RECEIVER,
    HISTORY_BUCKET,
)


def account_identity(frame: pd.DataFrame, schema: PaySimSchema | None = None) -> pd.Series:
    """Account under analysis per row: `nameDest` for TRANSFER, `nameOrig` for CASH_OUT."""
    schema = schema or PaySimSchema()
    is_cash_out = frame[schema.transaction_type_column] == CASH_OUT_TYPE
    return frame[schema.receiver_column].where(~is_cash_out, frame[schema.sender_column])


class CausalFeatureBuilder:
    """Derive features per receiver account without looking at future transactions.

    Account identity is role-aware (see `account_identity`): the identity is written into the
    receiver column and the original `nameDest` is kept as `nameDest_raw` for audits.

    PaySim's `step` is an hourly bucket, so the order of two transactions of the same
    receiver inside one step is NOT observable. This builder therefore never treats
    it as chronology:

    * `history_length`, `step_diff`, `log_amount_ratio`, `is_new_receiver` and
      `history_bucket` of a transaction are computed ONLY from the same receiver's
      transactions at STRICTLY EARLIER steps. Every transaction of one receiver in
      the same step gets identical history features, whatever the file order.
    * Rows are still sorted deterministically (account, step, then the label-free
      tie-break columns type, amount, sender), so frames built from any row
      permutation are identical and window construction is reproducible. That tie
      order carries no chronological meaning; `SequenceBuilder` never lets it
      decide which transactions appear in a window.
    """

    def __init__(self, config: DatasetConfig, schema: PaySimSchema | None = None) -> None:
        """Bind the builder to a dataset configuration and PaySim schema."""
        self._config = config
        self._schema = schema or PaySimSchema()

    def account_identity(self, frame: pd.DataFrame) -> pd.Series:
        """Account under analysis per row: `nameDest` for TRANSFER, `nameOrig` for CASH_OUT."""
        return account_identity(frame, self._schema)

    def tie_break_columns(self, frame: pd.DataFrame) -> list[str]:
        """Label-free columns that order same-(account, step) rows deterministically."""
        schema = self._schema
        candidates = (schema.transaction_type_column, schema.amount_column, schema.sender_column)
        return [column for column in candidates if column in frame.columns]

    def build(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return a copy ordered by (account, step, tie-break) with causal features added.

        The features of a transaction depend only on that transaction and on
        earlier-STEP transactions of the same account, so they can be computed in
        a streaming setting where same-hour order is unknown.
        """
        schema = self._schema
        identity = self.account_identity(frame)
        raw_receiver = f"{schema.receiver_column}_raw"
        codes = pd.factorize(identity)[0]
        ordered = (
            frame.assign(**{raw_receiver: frame[schema.receiver_column], schema.receiver_column: identity})
            .assign(**{ACCOUNT_CODE: codes})
            .sort_values(
                [ACCOUNT_CODE, schema.timestamp_column, *self.tie_break_columns(frame)],
                kind="stable",
            )
            .reset_index(drop=True)
        )
        # Rename amount column to standard "amount" for consistency across datasets
        ordered["amount"] = ordered[schema.amount_column]

        n_rows = len(ordered)
        position = np.arange(n_rows)
        code = ordered[ACCOUNT_CODE].to_numpy()
        step = ordered[schema.timestamp_column].to_numpy().astype("int64")
        amount = ordered["amount"].to_numpy(dtype="float64")

        new_account = np.ones(n_rows, dtype=bool)
        new_step = np.ones(n_rows, dtype=bool)
        if n_rows > 1:
            new_account[1:] = code[1:] != code[:-1]
            new_step[1:] = new_account[1:] | (step[1:] != step[:-1])
        account_start = np.maximum.accumulate(np.where(new_account, position, 0))
        step_start = np.maximum.accumulate(np.where(new_step, position, 0))

        # Rows of the same account at strictly earlier steps = rows before this step group.
        history = step_start - account_start
        has_history = history > 0
        previous = np.maximum(step_start - 1, 0)

        ordered[HISTORY_LENGTH] = history
        ordered[STEP_DIFF] = np.where(has_history, step - step[previous], 0).astype("int32")

        # Per-account running sum keeps the numbers small (float precision); the
        # prior total is the running sum at the last row of the previous step group.
        running = ordered.groupby(ACCOUNT_CODE, sort=False)["amount"].cumsum().to_numpy(dtype="float64")
        prior_total = np.where(has_history, running[previous], 0.0)
        prior_mean = np.where(has_history, prior_total / np.maximum(history, 1), np.nan)
        ratio = (amount + AMOUNT_EPSILON) / (prior_mean + AMOUNT_EPSILON)
        ordered[LOG_AMOUNT_RATIO] = np.nan_to_num(np.log(ratio), nan=0.0)

        # is_new_receiver: no transaction of this receiver at an earlier step
        ordered[IS_NEW_RECEIVER] = (~has_history).astype("int8")

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
