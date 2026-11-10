"""P0 pre-submit methodology audits (TODO.md P0).

Each test below is the automated, runnable version of one P0 checklist item:
no future leak in `CausalFeatureBuilder`, deterministic/causal windowing in
`SequenceBuilder`, receiver-disjoint splits, and a scaler fit only on
train-normal rows. All fixtures are tiny synthetic frames -- no PaySim CSV is
required to run this file.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lstm_autoencoder.config import DatasetConfig
from lstm_autoencoder.data.paysim_synthetic_financial.causal_features import (
    ACCOUNT_CODE,
    HISTORY_LENGTH,
    IS_NEW_RECEIVER,
    LOG_AMOUNT_RATIO,
    STEP_DIFF,
    CausalFeatureBuilder,
)
from lstm_autoencoder.data.paysim_synthetic_financial.schema import PaySimSchema
from lstm_autoencoder.evaluation.config import SplitConfig
from lstm_autoencoder.evaluation.paysim_synthetic_financial.baseline import BaselineEvaluator
from lstm_autoencoder.model.config import SequenceConfig
from lstm_autoencoder.model.feature_scaler import FeatureScaler
from lstm_autoencoder.model.sequence_builder import SequenceBuilder


@pytest.fixture
def dataset_config(tmp_path: Path) -> DatasetConfig:
    """A DatasetConfig that is never actually read from disk in these tests."""
    return DatasetConfig(project_root=tmp_path, dataset_name="audit_fixture")


@pytest.fixture
def schema() -> PaySimSchema:
    return PaySimSchema()


def _raw_frame() -> pd.DataFrame:
    """Two receivers (R1: 4 txns, R2: 2 txns) with one shuffled, out-of-order row.

    Row order is deliberately NOT sorted by (receiver, step) so a test can
    confirm CausalFeatureBuilder's own sort -- not incoming row order -- is
    what determines history, not whatever order the CSV happened to arrive in.
    """
    return pd.DataFrame(
        {
            "step": [1, 10, 5, 20, 2, 30],
            "nameDest": ["R1", "R1", "R1", "R2", "R1", "R2"],
            "nameOrig": ["A", "B", "C", "D", "E", "F"],
            "amount": [100.0, 300.0, 200.0, 50.0, 150.0, 70.0],
            "type": ["TRANSFER"] * 6,
            "isFraud": [0, 0, 0, 0, 0, 0],
            "isFlaggedFraud": [0] * 6,
        }
    )


class TestCausalFeatureBuilder:
    """P0 'Audit fitur historis': history_length/history_bucket/derived features
    must depend only on same-receiver transactions at or before the current step.
    """

    def test_sorts_by_receiver_then_step_regardless_of_input_order(self, dataset_config, schema):
        built = CausalFeatureBuilder(dataset_config, schema).build(_raw_frame())
        for _, group in built.groupby(ACCOUNT_CODE):
            assert group["step"].is_monotonic_increasing

    def test_history_length_counts_only_strictly_earlier_same_receiver_rows(self, dataset_config, schema):
        built = CausalFeatureBuilder(dataset_config, schema).build(_raw_frame())
        r1 = built[built["nameDest"] == "R1"].sort_values("step")
        # Steps for R1 after the builder's own sort: 1, 2, 5, 10 -> history 0,1,2,3
        assert r1[HISTORY_LENGTH].tolist() == [0, 1, 2, 3]
        r2 = built[built["nameDest"] == "R2"].sort_values("step")
        assert r2[HISTORY_LENGTH].tolist() == [0, 1]

    def test_is_new_receiver_matches_history_length_zero(self, dataset_config, schema):
        built = CausalFeatureBuilder(dataset_config, schema).build(_raw_frame())
        assert (built[IS_NEW_RECEIVER] == (built[HISTORY_LENGTH] == 0)).all()

    def test_log_amount_ratio_uses_only_prior_amounts(self, dataset_config, schema):
        """Row i's ratio must be computable from amounts of rows strictly before it;
        perturbing a LATER same-receiver amount must not change an EARLIER row's feature.
        """
        base = _raw_frame()
        built_a = CausalFeatureBuilder(dataset_config, schema).build(base)

        perturbed = base.copy()
        # R1's step=10 row (index 1) is NOT the first R1 row chronologically;
        # changing its amount must not affect the earlier R1 rows' features.
        perturbed.loc[1, "amount"] = 999_999.0
        built_b = CausalFeatureBuilder(dataset_config, schema).build(perturbed)

        earlier_mask = (built_a["nameDest"] == "R1") & (built_a["step"] < 10)
        pd.testing.assert_series_equal(
            built_a.loc[earlier_mask, LOG_AMOUNT_RATIO].reset_index(drop=True),
            built_b.loc[earlier_mask, LOG_AMOUNT_RATIO].reset_index(drop=True),
        )

    def test_step_diff_is_zero_for_first_transaction_of_each_receiver(self, dataset_config, schema):
        built = CausalFeatureBuilder(dataset_config, schema).build(_raw_frame())
        first_rows = built.groupby(ACCOUNT_CODE, as_index=False).nth(0)
        assert (first_rows[STEP_DIFF] == 0).all()

    def test_deterministic_across_repeated_builds(self, dataset_config, schema):
        """Same input frame -> byte-identical output frame, every time (stable sort)."""
        frame = _raw_frame()
        first = CausalFeatureBuilder(dataset_config, schema).build(frame)
        second = CausalFeatureBuilder(dataset_config, schema).build(frame)
        pd.testing.assert_frame_equal(first, second)

    def test_same_step_ties_break_by_input_row_order_deterministically(self, dataset_config, schema):
        """Three R1 rows share step=5; tie order must follow input row order, every rebuild."""
        frame = pd.DataFrame(
            {
                "step": [5, 5, 5, 1],
                "nameDest": ["R1", "R1", "R1", "R1"],
                "nameOrig": ["A", "B", "C", "D"],
                "amount": [10.0, 20.0, 30.0, 1.0],
                "type": ["TRANSFER"] * 4,
                "isFraud": [0, 0, 0, 0],
                "isFlaggedFraud": [0] * 4,
            }
        )
        built = CausalFeatureBuilder(dataset_config, schema).build(frame)
        # step=1 (nameOrig D) is first; among the step=5 tie, input order A,B,C is kept.
        assert built["nameOrig"].tolist() == ["D", "A", "B", "C"]
        rebuilt = CausalFeatureBuilder(dataset_config, schema).build(frame)
        assert rebuilt["nameOrig"].tolist() == built["nameOrig"].tolist()

    def test_account_identity_is_role_aware_transfer_vs_cash_out(self, dataset_config, schema):
        """TRANSFER rows key on nameDest (funds arriving); CASH_OUT rows key on
        nameOrig (funds leaving) -- so a mule account M1 that receives a TRANSFER
        and later performs a CASH_OUT is windowed as ONE account history, not two
        unrelated PaySim columns."""
        frame = pd.DataFrame(
            {
                "step": [1, 2, 3],
                "type": ["TRANSFER", "CASH_OUT", "CASH_OUT"],
                "amount": [100.0, 40.0, 60.0],
                "nameOrig": ["Sender1", "M1", "M1"],
                "nameDest": ["M1", "ATM_X", "ATM_Y"],
                "isFraud": [0, 0, 0],
                "isFlaggedFraud": [0, 0, 0],
            }
        )
        built = CausalFeatureBuilder(dataset_config, schema).build(frame)
        assert built[ACCOUNT_CODE].nunique() == 1  # all three rows belong to M1
        assert built["nameDest"].tolist() == ["M1", "M1", "M1"]  # role-aware identity column
        assert built["nameDest_raw"].tolist() == ["M1", "ATM_X", "ATM_Y"]  # original value preserved
        assert built[HISTORY_LENGTH].tolist() == [0, 1, 2]  # one continuous history for M1


class TestSequenceBuilderCausality:
    """P0 'Audit sequence construction': windowing/padding/masking must be
    deterministic and every window step must come from the SAME receiver at
    or before the current (last) row -- never a future row, never another receiver.
    """

    def _built_frame(self, dataset_config, schema, n_receivers: int = 2, steps_per_receiver: int = 7) -> pd.DataFrame:
        rows = []
        for r in range(n_receivers):
            for s in range(steps_per_receiver):
                rows.append(
                    {
                        "step": s,
                        "nameDest": f"R{r}",
                        "nameOrig": f"S{r}_{s}",
                        "amount": 10.0 * (s + 1) + r,
                        "type": "TRANSFER",
                        "isFraud": 0,
                        "isFlaggedFraud": 0,
                    }
                )
        return CausalFeatureBuilder(dataset_config, schema).build(pd.DataFrame(rows))

    def test_every_window_step_belongs_to_the_same_receiver_as_the_last_step(self, dataset_config, schema):
        built = self._built_frame(dataset_config, schema)
        config = SequenceConfig(window_length=4, min_history_length=1)
        batch = SequenceBuilder(config, schema).build(built)
        # amount column (feature index 0) uniquely identifies the source row here
        # (10*(s+1)+r), so a mismatch in the implied receiver offset would show up
        # as a non-integer / out-of-range value among the real (non-padded) steps.
        amounts = batch.sequences[:, :, 0]
        for i in range(len(batch.receiver)):
            real_steps = np.flatnonzero(batch.mask[i] == 1.0)
            last_amount = amounts[i, real_steps[-1]]
            r = 0.0 if batch.receiver[i] == "R0" else 1.0
            # amounts strictly increase with step for a fixed receiver, and the
            # last real step must be the example's own last step.
            for pos in real_steps[:-1]:
                assert amounts[i, pos] < last_amount
                assert (amounts[i, pos] - r) % 10.0 == 0.0  # belongs to the same receiver's series

    def test_padding_is_left_padding_and_mask_matches(self, dataset_config, schema):
        built = self._built_frame(dataset_config, schema, n_receivers=1, steps_per_receiver=2)
        config = SequenceConfig(window_length=5, min_history_length=1)
        batch = SequenceBuilder(config, schema).build(built)
        # history_length=0 (first txn) -> 1 real step, 4 padded; padding must be
        # on the LEFT (mask = [0,0,0,0,1]) so the real step lands at the window's end.
        first_example_mask = batch.mask[0]
        assert first_example_mask.tolist() == [0.0, 0.0, 0.0, 0.0, 1.0]
        assert np.all(batch.sequences[0, :4, :] == 0.0)

    def test_padding_mask_exhaustive_for_every_valid_length_1_through_window(self, dataset_config, schema):
        """TODO P1.3: mask must contain EXACTLY the real (non-padded) positions,
        left-padded, for every valid history length from 1 up to window_length --
        not just the single length=1 case covered above."""
        window = 5
        built = self._built_frame(dataset_config, schema, n_receivers=1, steps_per_receiver=window + 3)
        config = SequenceConfig(window_length=window, min_history_length=1)
        batch = SequenceBuilder(config, schema).build(built)
        for i, mask in enumerate(batch.mask):
            expected_len = min(i + 1, window)
            expected_mask = [0.0] * (window - expected_len) + [1.0] * expected_len
            assert mask.tolist() == expected_mask, f"example {i}: expected {expected_mask}, got {mask.tolist()}"
            assert np.all(batch.sequences[i, : window - expected_len, :] == 0.0)

    def test_min_history_length_filters_examples_without_enough_history(self, dataset_config, schema):
        built = self._built_frame(dataset_config, schema, n_receivers=1, steps_per_receiver=3)
        config = SequenceConfig(window_length=3, min_history_length=3)
        batch = SequenceBuilder(config, schema).build(built)
        # Only the 3rd transaction (history_length=2, i.e. 3rd txn) has >= 2 prior
        # txns, i.e. satisfies min_history_length=3 (>=2 prior + itself).
        assert len(batch.labels) == 1

    def test_deterministic_across_repeated_builds(self, dataset_config, schema):
        built = self._built_frame(dataset_config, schema)
        config = SequenceConfig(window_length=4, min_history_length=1)
        first = SequenceBuilder(config, schema).build(built)
        second = SequenceBuilder(config, schema).build(built)
        np.testing.assert_array_equal(first.sequences, second.sequences)
        np.testing.assert_array_equal(first.mask, second.mask)
        np.testing.assert_array_equal(first.receiver, second.receiver)


class TestReceiverSplitLeakage:
    """P0 'Audit split dan leakage': train/val/test must partition receivers
    (not rows) disjointly -- no receiver's rows may appear in two splits.
    """

    def _built_frame(self, dataset_config, schema, n_receivers: int = 60) -> pd.DataFrame:
        rng = np.random.default_rng(0)
        rows = []
        for r in range(n_receivers):
            n_txn = rng.integers(1, 5)
            is_fraud_receiver = r % 7 == 0
            for s in range(n_txn):
                rows.append(
                    {
                        "step": s,
                        "nameDest": f"R{r}",
                        "nameOrig": f"S{r}_{s}",
                        "amount": float(rng.uniform(10, 1000)),
                        "type": "TRANSFER",
                        "isFraud": int(is_fraud_receiver and s == n_txn - 1),
                        "isFlaggedFraud": 0,
                    }
                )
        return CausalFeatureBuilder(dataset_config, schema).build(pd.DataFrame(rows))

    def _evaluator(self, dataset_config) -> BaselineEvaluator:
        return BaselineEvaluator(dataset_config=dataset_config, split_config=SplitConfig())

    def test_splits_partition_rows_disjointly(self, dataset_config, schema):
        built = self._built_frame(dataset_config, schema)
        splits = self._evaluator(dataset_config)._stratified_receiver_split(built)
        train, val, test = set(splits["train"]), set(splits["val"]), set(splits["test"])
        assert train.isdisjoint(val)
        assert train.isdisjoint(test)
        assert val.isdisjoint(test)
        assert train | val | test == set(range(len(built)))

    def test_no_receiver_has_rows_in_more_than_one_split(self, dataset_config, schema):
        built = self._built_frame(dataset_config, schema)
        splits = self._evaluator(dataset_config)._stratified_receiver_split(built)
        receivers_in = {name: set(built.iloc[idx]["nameDest"].unique()) for name, idx in splits.items()}
        assert receivers_in["train"].isdisjoint(receivers_in["val"])
        assert receivers_in["train"].isdisjoint(receivers_in["test"])
        assert receivers_in["val"].isdisjoint(receivers_in["test"])

    def test_split_is_deterministic_for_a_fixed_seed(self, dataset_config, schema):
        built = self._built_frame(dataset_config, schema)
        splits_a = self._evaluator(dataset_config)._stratified_receiver_split(built)
        splits_b = self._evaluator(dataset_config)._stratified_receiver_split(built)
        for key in ("train", "val", "test"):
            np.testing.assert_array_equal(splits_a[key], splits_b[key])

    def test_split_does_not_use_receiver_fraud_rate(self, dataset_config, schema):
        """TODO P1.4: relabeling WHICH receivers are fraud (keeping each receiver's
        transaction COUNT fixed) must not change the split, since stratification
        is on transaction count, never on isFraud."""
        built = self._built_frame(dataset_config, schema)
        splits_a = self._evaluator(dataset_config)._stratified_receiver_split(built)

        relabeled = built.copy()
        receivers = relabeled["nameDest"].unique()
        rng = np.random.default_rng(99)
        fraud_receivers = set(rng.choice(receivers, size=max(1, len(receivers) // 5), replace=False))
        relabeled["isFraud"] = relabeled["nameDest"].isin(fraud_receivers).astype(int)
        splits_b = self._evaluator(dataset_config)._stratified_receiver_split(relabeled)

        for key in ("train", "val", "test"):
            np.testing.assert_array_equal(splits_a[key], splits_b[key])

    def test_split_does_not_use_receiver_fraud_rate(self, dataset_config, schema):
        """TODO P1.4: relabeling WHICH receivers are fraud (keeping each receiver's
        transaction COUNT fixed) must not change the split, since stratification
        is on transaction count, never on isFraud."""
        built = self._built_frame(dataset_config, schema)
        splits_a = self._evaluator(dataset_config)._stratified_receiver_split(built)

        relabeled = built.copy()
        receivers = relabeled["nameDest"].unique()
        rng = np.random.default_rng(99)
        fraud_receivers = set(rng.choice(receivers, size=max(1, len(receivers) // 5), replace=False))
        relabeled["isFraud"] = relabeled["nameDest"].isin(fraud_receivers).astype(int)
        splits_b = self._evaluator(dataset_config)._stratified_receiver_split(relabeled)

        for key in ("train", "val", "test"):
            np.testing.assert_array_equal(splits_a[key], splits_b[key])


class TestScalerFitOnTrainNormalOnly:
    """P0 'Audit split dan leakage' (preprocessing clause): the feature scaler's
    mean/std must come ONLY from train-split, label==0 rows -- never val, test,
    or fraud rows, even when they are present in the frame passed to `fit`.
    """

    def test_fit_statistics_ignore_rows_outside_the_mask(self):
        rng = np.random.default_rng(1)
        n = 200
        frame = pd.DataFrame({"amount": rng.uniform(1, 100, n), "history_length": rng.integers(0, 10, n)})
        # First half = "train normal" (mask True), second half = everything else,
        # deliberately drawn from a very different distribution.
        mask = pd.Series([True] * (n // 2) + [False] * (n // 2))
        frame.loc[~mask, "amount"] = rng.uniform(10_000, 100_000, n // 2)

        scaler = FeatureScaler(feature_columns=("amount", "history_length")).fit(frame, mask)
        expected = StandardScalerFit(frame.loc[mask, ["amount", "history_length"]])
        np.testing.assert_allclose(scaler._scaler.mean_, expected.mean_, rtol=1e-10)
        np.testing.assert_allclose(scaler._scaler.scale_, expected.scale_, rtol=1e-10)


def StandardScalerFit(frame: pd.DataFrame):
    """Small local helper so the test above doesn't depend on FeatureScaler internals
    to compute its own expected value (that would make the test tautological)."""
    from sklearn.preprocessing import StandardScaler

    return StandardScaler().fit(frame)
