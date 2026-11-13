"""BankSim-specific causal feature builder for per-account sequence features."""

from __future__ import annotations

import numpy as np
import pandas as pd

from lstm_autoencoder.config import DatasetConfig
from lstm_autoencoder.data.banksim.schema import BankSimSchema

ACCOUNT_CODE = "account_code"
HISTORY_LENGTH = "history_length"
STEP_DIFF = "step_diff"
LOG_AMOUNT_RATIO = "log_amount_ratio"
IS_NEW_RECEIVER = "is_new_receiver"
HISTORY_BUCKET = "history_bucket"
ROLLING_LOG_RATIO_MEAN_3 = "rolling_log_ratio_mean_3"
ROLLING_LOG_RATIO_MAX_3 = "rolling_log_ratio_max_3"
ROLLING_LOG_RATIO_MEAN_5 = "rolling_log_ratio_mean_5"
ROLLING_LOG_RATIO_MAX_5 = "rolling_log_ratio_max_5"
IS_NEW_MERCHANT = "is_new_merchant"
LOG_MERCHANT_VISITS = "log_merchant_visits"
LOG_AMOUNT_RATIO_CAT = "log_amount_ratio_cat"
AMOUNT_EPSILON = 1e-9

DERIVED_FEATURES: tuple[str, ...] = (
    HISTORY_LENGTH,
    STEP_DIFF,
    LOG_AMOUNT_RATIO,
    IS_NEW_RECEIVER,
    HISTORY_BUCKET,
    ROLLING_LOG_RATIO_MEAN_3,
    ROLLING_LOG_RATIO_MAX_3,
    ROLLING_LOG_RATIO_MEAN_5,
    ROLLING_LOG_RATIO_MAX_5,
    IS_NEW_MERCHANT,
    LOG_MERCHANT_VISITS,
    LOG_AMOUNT_RATIO_CAT,
)

# Columns that describe how a transaction deviates from the customer's own past
# (used by the `rolling_only` ablation); the three newest are merchant/category aware.
DEVIATION_FEATURES: tuple[str, ...] = (
    STEP_DIFF,
    LOG_AMOUNT_RATIO,
    ROLLING_LOG_RATIO_MEAN_3,
    ROLLING_LOG_RATIO_MAX_3,
    ROLLING_LOG_RATIO_MEAN_5,
    ROLLING_LOG_RATIO_MAX_5,
    IS_NEW_MERCHANT,
    LOG_MERCHANT_VISITS,
    LOG_AMOUNT_RATIO_CAT,
)


def earlier_step_count(keys: np.ndarray, step: np.ndarray) -> np.ndarray:
    """Per row: how many rows with the same key sit at a STRICTLY earlier step.

    Requires the rows to be ordered by step within every key (the builder orders by
    (account, step), which implies it for any key nested in the account). Rows that
    share a step never count for each other, so the result does not depend on how the
    same-step rows happen to be ordered.
    """
    frame = pd.DataFrame({"key": keys, "step": step})
    position = frame.groupby("key", sort=False).cumcount().to_numpy()
    position_in_step = frame.groupby(["key", "step"], sort=False).cumcount().to_numpy()
    return position - position_in_step


def earlier_step_sum(keys: np.ndarray, step: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Per row: the sum of `values` over the rows with the same key at a STRICTLY earlier step."""
    frame = pd.DataFrame({"key": keys, "step": step, "value": values})
    running = frame.groupby("key", sort=False)["value"].cumsum().to_numpy()
    running_in_step = frame.groupby(["key", "step"], sort=False)["value"].cumsum().to_numpy()
    return running - running_in_step


def account_identity(frame: pd.DataFrame, schema: BankSimSchema | None = None) -> pd.Series:
    """Account under analysis per row: `customer` for BankSim."""
    schema = schema or BankSimSchema()
    return frame[schema.sender_column]


class CausalFeatureBuilder:
    """Derive features per customer account without looking at future transactions.

    BankSim uses `step` as timestamp and `customer` as the account identity.
    """

    def __init__(self, config: DatasetConfig, schema: BankSimSchema | None = None) -> None:
        """Bind the builder to a dataset configuration and BankSim schema."""
        self._config = config
        self._schema = schema or BankSimSchema()

    def account_identity(self, frame: pd.DataFrame) -> pd.Series:
        """Account under analysis per row: `customer` for BankSim."""
        return account_identity(frame, self._schema)

    def tie_break_columns(self, frame: pd.DataFrame) -> list[str]:
        """Label-free columns that order same-(account, step) rows deterministically."""
        schema = self._schema
        candidates = (schema.transaction_type_column, schema.amount_column, schema.receiver_column)
        return [column for column in candidates if column in frame.columns]

    def build(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return a copy ordered by (account, step, tie-break) with causal features added.

        The features of a transaction depend only on that transaction and on
        earlier-STEP transactions of the same account, so they can be computed in
        a streaming setting where same-hour order is unknown.
        """
        schema = self._schema
        identity = self.account_identity(frame)
        raw_merchant = f"{schema.receiver_column}_raw"
        codes = pd.factorize(identity)[0]
        ordered = frame.copy()
        ordered[raw_merchant] = frame[schema.receiver_column]
        ordered[ACCOUNT_CODE] = codes
        # Rename amount column to standard "amount" for consistency across datasets
        ordered["amount"] = ordered[schema.amount_column]
        ordered = ordered.sort_values(
            [ACCOUNT_CODE, schema.timestamp_column, *self.tie_break_columns(frame)],
            kind="stable",
        ).reset_index(drop=True)

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

        # NOTE: despite the name (kept so reports stay comparable with PaySim), this flags a
        # customer with no transaction at an earlier step, NOT a first visit to a merchant.
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

        # Rolling features: mean/max of log_amount_ratio from last 3 and 5 transactions
        # Only using transactions from strictly earlier steps (causal)
        # First, compute per-step per-account mean log_amount_ratio
        df_for_rolling = ordered[[ACCOUNT_CODE, schema.timestamp_column, LOG_AMOUNT_RATIO]].copy()
        step_stats = df_for_rolling.groupby([ACCOUNT_CODE, schema.timestamp_column], sort=False)[LOG_AMOUNT_RATIO].mean().reset_index()
        step_stats.columns = [ACCOUNT_CODE, schema.timestamp_column, "step_log_ratio_mean"]

        # Shift to get stats from previous steps
        step_stats[f"prev_{LOG_AMOUNT_RATIO}"] = step_stats.groupby(ACCOUNT_CODE, sort=False)["step_log_ratio_mean"].shift(1)

        # Rolling windows on the shifted series (per account)
        rolling3 = step_stats.groupby(ACCOUNT_CODE, sort=False)[f"prev_{LOG_AMOUNT_RATIO}"].rolling(window=3, min_periods=1)
        rolling5 = step_stats.groupby(ACCOUNT_CODE, sort=False)[f"prev_{LOG_AMOUNT_RATIO}"].rolling(window=5, min_periods=1)

        step_stats[ROLLING_LOG_RATIO_MEAN_3] = rolling3.mean().reset_index(level=0, drop=True)
        step_stats[ROLLING_LOG_RATIO_MAX_3] = rolling3.max().reset_index(level=0, drop=True)
        step_stats[ROLLING_LOG_RATIO_MEAN_5] = rolling5.mean().reset_index(level=0, drop=True)
        step_stats[ROLLING_LOG_RATIO_MAX_5] = rolling5.max().reset_index(level=0, drop=True)

        # Merge back to original dataframe
        rolling_cols = [ROLLING_LOG_RATIO_MEAN_3, ROLLING_LOG_RATIO_MAX_3, ROLLING_LOG_RATIO_MEAN_5, ROLLING_LOG_RATIO_MAX_5]
        ordered = ordered.merge(
            step_stats[[ACCOUNT_CODE, schema.timestamp_column] + rolling_cols],
            on=[ACCOUNT_CODE, schema.timestamp_column],
            how="left"
        )

        # Fill NaN with 0 for transactions with no history
        for col in rolling_cols:
            ordered[col] = ordered[col].fillna(0.0)

        # Re-read the arrays from the merged frame so the features below never depend on
        # the row order the merge happened to return.
        code = ordered[ACCOUNT_CODE].to_numpy().astype("int64")
        step = ordered[schema.timestamp_column].to_numpy().astype("int64")
        amount = ordered["amount"].to_numpy(dtype="float64")
        self._add_merchant_and_category_features(ordered, code, step, amount)
        return ordered

    def _add_merchant_and_category_features(
        self, ordered: pd.DataFrame, code: np.ndarray, step: np.ndarray, amount: np.ndarray
    ) -> None:
        """Add merchant-novelty and category-aware deviation features (in place), causally.

        Everything is computed from rows of the same customer at STRICTLY earlier steps,
        like the other derived features, so it is available in a streaming setting where
        same-day order is unknown. No label is read.
        """
        schema = self._schema
        merchant_code = pd.factorize(ordered[schema.receiver_column])[0].astype("int64")
        pair = code * (int(merchant_code.max()) + 1) + merchant_code
        visits = earlier_step_count(pair, step)
        ordered[IS_NEW_MERCHANT] = (visits == 0).astype("int8")
        ordered[LOG_MERCHANT_VISITS] = np.log1p(visits)

        category_code = pd.factorize(ordered[schema.transaction_type_column])[0].astype("int64")
        category_pair = code * (int(category_code.max()) + 1) + category_code
        prior_count = earlier_step_count(category_pair, step)
        prior_total = earlier_step_sum(category_pair, step, amount)
        has_prior = prior_count > 0
        prior_mean = np.where(has_prior, prior_total / np.maximum(prior_count, 1), np.nan)
        ratio = (amount + AMOUNT_EPSILON) / (prior_mean + AMOUNT_EPSILON)
        ordered[LOG_AMOUNT_RATIO_CAT] = np.where(has_prior, np.log(ratio), 0.0)

    def feature_columns(self, built: pd.DataFrame) -> list[str]:
        """List the derived feature columns present in a built frame."""
        return [name for name in DERIVED_FEATURES if name in built.columns]
