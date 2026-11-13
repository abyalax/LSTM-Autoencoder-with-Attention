"""Input features of the BankSim LSTM-Autoencoder.

BankSim fraud is mostly "a large amount in a category where such amounts are rare", so
the autoencoder needs to see the category and how unusual the amount is FOR THAT
CATEGORY. Neither exists in the PaySim feature set (`amount`, `step_diff`, ...), and
without them the model cannot even express the main fraud pattern of this dataset.

Everything here is label-free; the one statistic that must be learned from data
(`CategoryAmountNormalizer`) is fit on train-normal rows only.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from lstm_autoencoder.data.banksim.causal_features import (
    IS_NEW_MERCHANT,
    LOG_AMOUNT_RATIO,
    LOG_AMOUNT_RATIO_CAT,
)
from lstm_autoencoder.data.banksim.schema import BankSimSchema

CATEGORY_PREFIX = "cat_"
AMOUNT_CATEGORY_Z = "amount_cat_z"
STEP_DIFF = "step_diff"
AMOUNT = "amount"

# Continuous inputs, in a fixed order. One-hot category columns are appended after them.
CONTINUOUS_FEATURES: tuple[str, ...] = (
    AMOUNT,
    STEP_DIFF,
    LOG_AMOUNT_RATIO,
    LOG_AMOUNT_RATIO_CAT,
    AMOUNT_CATEGORY_Z,
)
# Heavy-tailed, non-negative: log1p before standardizing.
LOG1P_FEATURES: tuple[str, ...] = (AMOUNT, STEP_DIFF)
# Binary indicators stay 0/1 (see `FeatureScaler.passthrough_columns`).
BINARY_FEATURES: tuple[str, ...] = (IS_NEW_MERCHANT,)

MIN_CATEGORY_ROWS = 20
MIN_STD = 1e-6


def sanitize_category(value: object) -> str:
    """Column-safe category name: BankSim values arrive quoted, e.g. `'es_travel'`."""
    return re.sub(r"[^0-9A-Za-z_]+", "", str(value))


def category_columns(built: pd.DataFrame, schema: BankSimSchema | None = None) -> tuple[tuple[str, ...], dict[str, str]]:
    """Return (one-hot column names, {column name: raw category value}) for the categories present."""
    schema = schema or BankSimSchema()
    raw_values = sorted(built[schema.transaction_type_column].dropna().unique().tolist())
    mapping: dict[str, str] = {}
    for value in raw_values:
        name = CATEGORY_PREFIX + sanitize_category(value)
        if name in mapping:
            raise ValueError(f"Two categories map to the same column name {name!r}.")
        mapping[name] = value
    return tuple(mapping), mapping


def add_category_one_hot(built: pd.DataFrame, schema: BankSimSchema | None = None) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Return `built` plus one 0/1 float32 column per category, and the new column names."""
    schema = schema or BankSimSchema()
    names, mapping = category_columns(built, schema)
    category = built[schema.transaction_type_column]
    one_hot = pd.DataFrame(
        {name: (category == value).to_numpy(dtype="float32") for name, value in mapping.items()},
        index=built.index,
    )
    return pd.concat([built, one_hot], axis=1), names


class CategoryAmountNormalizer:
    """z-score of log1p(amount) within its category, fit on the rows selected by `fit_mask`.

    A $600 purchase is ordinary for travel and extreme for food; one global amount
    scale cannot tell them apart, a per-category one can. Categories with too few fit
    rows (or no spread) fall back to the global mean/std instead of dividing by ~0.
    """

    def __init__(self, schema: BankSimSchema | None = None) -> None:
        """Bind to the schema that names the category and amount columns."""
        self._schema = schema or BankSimSchema()
        self._stats: dict[object, tuple[float, float]] = {}
        self._fallback: tuple[float, float] = (0.0, 1.0)

    def fit(self, built: pd.DataFrame, fit_mask: pd.Series) -> CategoryAmountNormalizer:
        """Learn per-category mean/std of log1p(amount) from the masked rows."""
        rows = built.loc[fit_mask]
        log_amount = np.log1p(rows[self._schema.amount_column].to_numpy(dtype="float64"))
        self._fallback = (float(log_amount.mean()), max(float(log_amount.std()), MIN_STD))
        grouped = pd.Series(log_amount, index=rows.index).groupby(rows[self._schema.transaction_type_column])
        self._stats = {}
        for category, values in grouped:
            std = float(values.std()) if len(values) > 1 else 0.0
            if len(values) >= MIN_CATEGORY_ROWS and std > MIN_STD:
                self._stats[category] = (float(values.mean()), std)
        return self

    def transform(self, built: pd.DataFrame) -> np.ndarray:
        """Per-row z-score of log1p(amount) in its category (float64)."""
        category = built[self._schema.transaction_type_column]
        mean = category.map(lambda value: self._stats.get(value, self._fallback)[0]).to_numpy(dtype="float64")
        std = category.map(lambda value: self._stats.get(value, self._fallback)[1]).to_numpy(dtype="float64")
        log_amount = np.log1p(built[self._schema.amount_column].to_numpy(dtype="float64"))
        return (log_amount - mean) / std


def ae_feature_columns(category_names: tuple[str, ...]) -> tuple[str, ...]:
    """The ordered input columns of the BankSim LSTM-AE: continuous, binary, then one-hot categories."""
    return (*CONTINUOUS_FEATURES, *BINARY_FEATURES, *category_names)


def add_ae_features(
    built: pd.DataFrame, fit_mask: pd.Series, schema: BankSimSchema | None = None
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Add the one-hot categories and `amount_cat_z` to a causal-feature frame.

    `fit_mask` must select train-normal rows only: it is the only place a statistic
    is learned from data. Returns the new frame and the ordered AE input columns.
    """
    schema = schema or BankSimSchema()
    enriched, names = add_category_one_hot(built, schema)
    normalizer = CategoryAmountNormalizer(schema).fit(enriched, fit_mask)
    enriched[AMOUNT_CATEGORY_Z] = normalizer.transform(enriched)
    return enriched, ae_feature_columns(names)
