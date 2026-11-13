"""BankSim derived features: causal, order-independent, label-free; scaler leaves indicators alone."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from banksim_fixtures import synthetic_banksim

from lstm_autoencoder.config import DatasetConfig
from lstm_autoencoder.data.banksim.ae_features import (
    AMOUNT_CATEGORY_Z,
    BINARY_FEATURES,
    CONTINUOUS_FEATURES,
    LOG1P_FEATURES,
    CategoryAmountNormalizer,
    add_ae_features,
    sanitize_category,
)
from lstm_autoencoder.data.banksim.causal_features import (
    IS_NEW_MERCHANT,
    LOG_AMOUNT_RATIO_CAT,
    LOG_MERCHANT_VISITS,
    CausalFeatureBuilder,
)
from lstm_autoencoder.data.banksim.schema import BankSimSchema
from lstm_autoencoder.model.feature_scaler import FeatureScaler

NEW_FEATURES = [IS_NEW_MERCHANT, LOG_MERCHANT_VISITS, LOG_AMOUNT_RATIO_CAT]


@pytest.fixture
def builder(tmp_path: Path) -> CausalFeatureBuilder:
    return CausalFeatureBuilder(DatasetConfig(project_root=tmp_path, dataset_name="banksim"), BankSimSchema())


def _tagged(seed: int = 1) -> pd.DataFrame:
    frame = synthetic_banksim(n_customers=40, seed=seed)
    frame["row_id"] = np.arange(len(frame))
    return frame


def test_new_features_do_not_depend_on_row_order(builder: CausalFeatureBuilder) -> None:
    frame = _tagged()
    base = builder.build(frame).sort_values("row_id").reset_index(drop=True)
    shuffled = builder.build(frame.sample(frac=1.0, random_state=9)).sort_values("row_id").reset_index(drop=True)
    for column in NEW_FEATURES:
        assert np.allclose(base[column].to_numpy(), shuffled[column].to_numpy()), column


def test_new_features_ignore_the_future(builder: CausalFeatureBuilder) -> None:
    frame = _tagged()
    base = builder.build(frame).set_index("row_id").sort_index()

    # Rewrite every transaction of one customer AFTER a cut step.
    customer = frame.groupby("customer")["step"].nunique().idxmax()
    steps = np.sort(frame.loc[frame["customer"] == customer, "step"].unique())
    cut_step = int(steps[len(steps) // 2])
    changed = frame.copy()
    later = (changed["customer"] == customer) & (changed["step"] > cut_step)
    changed.loc[later, "amount"] = changed.loc[later, "amount"] * 50 + 7
    changed.loc[later, "merchant"] = "'M9999'"
    rebuilt = builder.build(changed).set_index("row_id").sort_index()

    ids = frame.loc[(frame["customer"] == customer) & (frame["step"] <= cut_step), "row_id"].to_numpy()
    for column in NEW_FEATURES:
        assert np.allclose(base.loc[ids, column], rebuilt.loc[ids, column]), column


def test_new_features_ignore_same_step_peers(builder: CausalFeatureBuilder) -> None:
    frame = _tagged()
    base = builder.build(frame).set_index("row_id").sort_index()

    sizes = frame.groupby(["customer", "step"]).size()
    customer, step = sizes[sizes >= 2].index[0]
    peers = frame[(frame["customer"] == customer) & (frame["step"] == step)]
    changed = frame.copy()
    first = peers["row_id"].iloc[0]
    changed.loc[changed["row_id"] == first, ["amount", "merchant"]] = [99999.0, "'M9999'"]
    rebuilt = builder.build(changed).set_index("row_id").sort_index()

    others = peers["row_id"].iloc[1:].to_numpy()
    for column in NEW_FEATURES:
        assert np.allclose(base.loc[others, column], rebuilt.loc[others, column]), column


def test_merchant_features_match_a_slow_reference(builder: CausalFeatureBuilder) -> None:
    frame = _tagged(seed=4)
    built = builder.build(frame).set_index("row_id").sort_index()
    for _, rows in frame.groupby("customer"):
        for _, row in rows.iterrows():
            earlier = rows[(rows["step"] < row["step"])]
            same_merchant = earlier[earlier["merchant"] == row["merchant"]]
            same_category = earlier[earlier["category"] == row["category"]]
            got = built.loc[row["row_id"]]
            assert got[IS_NEW_MERCHANT] == int(len(same_merchant) == 0)
            assert np.isclose(got[LOG_MERCHANT_VISITS], np.log1p(len(same_merchant)))
            if len(same_category) == 0:
                assert got[LOG_AMOUNT_RATIO_CAT] == 0.0
            else:
                expected = np.log((row["amount"] + 1e-9) / (same_category["amount"].mean() + 1e-9))
                assert np.isclose(got[LOG_AMOUNT_RATIO_CAT], expected)
        break  # one customer is enough for the O(n^2) reference; the shuffle test covers the rest


def test_category_one_hot_and_normalizer_are_fit_on_the_mask_only(builder: CausalFeatureBuilder) -> None:
    built = builder.build(_tagged())
    schema = BankSimSchema()
    train_normal = (built[schema.target_column] == 0) & built[schema.sender_column].isin(
        built[schema.sender_column].unique()[:25]
    )
    enriched, columns = add_ae_features(built, train_normal)

    one_hot = [c for c in columns if c.startswith("cat_")]
    assert len(one_hot) == 15 and all("'" not in c for c in one_hot)
    assert np.array_equal(enriched[one_hot].sum(axis=1).to_numpy(), np.ones(len(enriched)))
    assert columns[: len(CONTINUOUS_FEATURES)] == CONTINUOUS_FEATURES
    assert set(BINARY_FEATURES) <= set(columns)

    # On the rows it was fit on, the per-category z-score has mean ~0 and std ~1.
    fitted = enriched.loc[train_normal]
    by_category = fitted.groupby(schema.transaction_type_column)[AMOUNT_CATEGORY_Z].agg(["mean", "std"])
    assert np.allclose(by_category["mean"], 0.0, atol=1e-6)
    assert np.allclose(by_category["std"], 1.0, atol=1e-6)

    # Changing labels or non-fit rows must not change the learned statistics.
    other = built.copy()
    other.loc[~train_normal, "amount"] *= 100
    again = CategoryAmountNormalizer(schema).fit(other, train_normal).transform(built)
    assert np.allclose(again, enriched[AMOUNT_CATEGORY_Z].to_numpy())


def test_normalizer_falls_back_for_rare_categories() -> None:
    schema = BankSimSchema()
    frame = pd.DataFrame(
        {
            schema.transaction_type_column: ["a"] * 30 + ["rare"] * 2,
            schema.amount_column: list(np.linspace(1, 50, 30)) + [5.0, 5.0],
        }
    )
    z = CategoryAmountNormalizer(schema).fit(frame, pd.Series(True, index=frame.index)).transform(frame)
    assert np.isfinite(z).all()


def test_scaler_leaves_passthrough_columns_untouched() -> None:
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(
        {
            "amount": rng.lognormal(3, 1, 500),
            "step_diff": rng.integers(0, 30, 500).astype(float),
            "flag": rng.integers(0, 2, 500).astype(float),
        }
    )
    columns = ("amount", "step_diff", "flag")
    scaler = FeatureScaler(columns, ("amount", "step_diff"), passthrough_columns=("flag",))
    scaled = scaler.fit(frame, pd.Series(True, index=frame.index)).transform(frame)

    assert np.array_equal(scaled["flag"].to_numpy(), frame["flag"].to_numpy())
    assert abs(scaled["amount"].mean()) < 1e-9 and abs(scaled["amount"].std(ddof=0) - 1) < 1e-9
    assert set(LOG1P_FEATURES) == {"amount", "step_diff"}


def test_sanitize_category_strips_the_csv_quotes() -> None:
    assert sanitize_category("'es_travel'") == "es_travel"
