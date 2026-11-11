"""The audit's checks must pass on a sound pipeline and FAIL on deliberately broken ones."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lstm_autoencoder.config import DatasetConfig
from lstm_autoencoder.data.paysim_synthetic_financial.causal_features import (
    ACCOUNT_CODE,
    LOG_AMOUNT_RATIO,
    CausalFeatureBuilder,
)
from lstm_autoencoder.evaluation.paysim_synthetic_financial import pipeline_audit
from lstm_autoencoder.evaluation.paysim_synthetic_financial.pipeline_audit import (
    FAIL,
    INFO,
    PASS,
    PipelineAudit,
    PipelineAuditConfig,
)


def _frame(seed: int = 0, n_receivers: int = 80, steps: int = 30) -> pd.DataFrame:
    """PaySim-shaped transactions sorted by step, with ties inside a step and receivers of mixed history length."""
    rng = np.random.default_rng(seed)
    counts = np.minimum(rng.geometric(0.15, size=n_receivers), 40)
    receiver = np.repeat([f"C{i}" for i in range(n_receivers)], counts)
    n = len(receiver)
    frame = pd.DataFrame(
        {
            "step": rng.integers(1, steps, size=n),
            "type": rng.choice(["CASH_OUT", "TRANSFER"], size=n),
            "amount": np.round(rng.lognormal(10, 1.2, size=n), 2),
            "nameOrig": [f"O{i}" for i in range(n)],
            "nameDest": receiver,
            "isFraud": (rng.random(n) < 0.08).astype(int),
        }
    )
    return frame.sort_values("step", kind="stable").reset_index(drop=True)


def _audit(tmp_path, **config) -> PipelineAudit:
    return PipelineAudit(
        DatasetConfig(project_root=tmp_path, dataset_name="paysim_synthetic_financial"),
        PipelineAuditConfig(sample_receivers=1000, **config),
    )


def test_sound_pipeline_passes_every_hard_check(tmp_path):
    frame, rng = _frame(), np.random.default_rng(0)
    audit = _audit(tmp_path)
    built = audit._build(frame)
    findings = [
        audit.check_feature_reference(frame),
        audit.check_future_independence(frame, rng),
        audit.check_receiver_isolation(frame, frame[frame["nameDest"].isin([f"C{i}" for i in range(20)])]),
        *[audit.check_sequences(built, window, rng) for window in (1, 3, 5, 10)],
    ]
    assert [(f.name, f.status) for f in findings if f.status != PASS] == []


class _LeakyBuilder(CausalFeatureBuilder):
    """Adds the NEXT transaction's amount into log_amount_ratio: a textbook look-ahead leak."""

    def build(self, frame: pd.DataFrame) -> pd.DataFrame:
        built = super().build(frame)
        next_amount = built.groupby(ACCOUNT_CODE)["amount"].shift(-1).fillna(0.0)
        built[LOG_AMOUNT_RATIO] = built[LOG_AMOUNT_RATIO] + np.log1p(next_amount)
        return built


def test_look_ahead_leak_is_detected(tmp_path):
    frame, rng = _frame(), np.random.default_rng(0)
    audit = _audit(tmp_path)
    audit._builder = _LeakyBuilder(audit._dataset_config, audit._schema)
    assert audit.check_future_independence(frame, rng).status == FAIL
    assert audit.check_feature_reference(frame).status == FAIL


def test_broken_sequence_builder_is_detected(tmp_path, monkeypatch):
    class Broken(pipeline_audit.SequenceBuilder):
        def build(self, built):
            batch = super().build(built)
            batch.sequences[:, 0, 0] += 1.0  # corrupts every window, including padded steps
            return batch

    monkeypatch.setattr(pipeline_audit, "SequenceBuilder", Broken)
    audit = _audit(tmp_path)
    finding = audit.check_sequences(audit._build(_frame()), 3, np.random.default_rng(0))
    assert finding.status == FAIL


class _FileOrderBuilder(CausalFeatureBuilder):
    """The OLD behaviour: same-step rows are ordered by file order and counted as earlier history."""

    def build(self, frame: pd.DataFrame) -> pd.DataFrame:
        codes = pd.factorize(frame["nameDest"])[0]
        ordered = frame.assign(**{ACCOUNT_CODE: codes}).sort_values([ACCOUNT_CODE, "step"], kind="stable")
        ordered = ordered.reset_index(drop=True)
        groups = ordered.groupby(ACCOUNT_CODE, sort=False)
        history = groups.cumcount()
        ordered["history_length"] = history
        ordered["step_diff"] = groups["step"].diff().fillna(0).astype("int32")
        prior = (groups["amount"].cumsum() - ordered["amount"]) / history.replace(0, np.nan)
        ordered["log_amount_ratio"] = np.log((ordered["amount"] + 1e-9) / (prior + 1e-9)).fillna(0)
        ordered["is_new_receiver"] = (history == 0).astype("int8")
        ordered["history_bucket"] = np.select([history == 0, history < 4, history < 9], ["1", "2-4", "5-9"], ">=10")
        return ordered


def test_features_are_invariant_to_same_step_order(tmp_path):
    audit = _audit(tmp_path)
    finding = audit.check_tie_sensitivity(_frame(), np.random.default_rng(0))
    assert finding.status == PASS
    assert finding.detail["share_rows_with_same_step_peer"] > 0  # the fixture really has ties
    assert all(v == 0 for v in finding.detail["share_rows_changed_when_all_rows_are_shuffled"].values())


def test_file_order_dependence_is_detected(tmp_path):
    audit = _audit(tmp_path)
    audit._builder = _FileOrderBuilder(audit._dataset_config, audit._schema)
    assert audit.check_tie_sensitivity(_frame(), np.random.default_rng(0)).status == FAIL
    assert audit.check_feature_reference(_frame()).status == FAIL


def test_receiver_semantics_reports_the_mule_path_coverage(tmp_path):
    frame = pd.DataFrame(
        {
            "step": [1, 5, 6, 7],
            "type": ["TRANSFER", "CASH_OUT", "CASH_OUT", "CASH_OUT"],
            "amount": [100.0, 90.0, 50.0, 20.0],
            "nameOrig": ["S", "M", "X", "Y"],
            "nameDest": ["M", "Z", "Z", "Z"],
            "isFraud": [1, 1, 0, 0],
        }
    )
    finding = _audit(tmp_path).check_receiver_semantics(frame)
    assert finding.status == INFO
    assert finding.detail["share_cash_out_origin_was_earlier_transfer_receiver"] == pytest.approx(1 / 3)
    assert finding.detail["share_fraud_cash_out_origin_was_earlier_transfer_receiver"] == 1.0


def _random_split(built: pd.DataFrame, leak: bool = False) -> dict[str, pd.Index]:
    receivers = pd.Index(built["nameDest"].unique())
    third = len(receivers) // 3
    ids = {"train": receivers[:third], "val": receivers[third : 2 * third], "test": receivers[2 * third :]}
    if leak:  # one receiver shows up in both train and test
        ids["test"] = ids["test"].append(ids["train"][:1])
    return ids


def test_split_checks_pass_on_disjoint_split_and_fail_on_overlap(tmp_path):
    audit = _audit(tmp_path)
    built = CausalFeatureBuilder(audit._dataset_config, audit._schema).build(_frame())
    good = {f.name: f for f in audit.check_split(built, _random_split(built), (1, 5))}
    assert good["split_disjoint"].status == PASS and good["scaler_fit_on_train_normal_only"].status == PASS
    bad = {f.name: f for f in audit.check_split(built, _random_split(built, leak=True), (1, 5))}
    assert bad["split_disjoint"].status == FAIL


def test_context_contamination_matches_brute_force(tmp_path):
    audit = _audit(tmp_path)
    built = CausalFeatureBuilder(audit._dataset_config, audit._schema).build(_frame(3))
    normal_rows = np.flatnonzero(built["isFraud"].to_numpy() == 0)
    for window in (2, 3, 5, 10):
        measured = audit._context_contamination(built, normal_rows, window)["share_with_fraud_in_context"]
        label = built["isFraud"].to_numpy()
        code = built[ACCOUNT_CODE].to_numpy()
        hits = 0
        step = built["step"].to_numpy()
        for row in normal_rows:
            same = np.flatnonzero((code == code[row]) & (step < step[row]))  # strictly earlier steps only
            context = same[-(window - 1) :]
            hits += int(label[context].sum() > 0)
        assert measured == pytest.approx(hits / len(normal_rows))


def test_metric_implementation_matches_scikit_learn_even_with_ties(tmp_path):
    rng = np.random.default_rng(0)
    labels = (rng.random(5000) < 0.05).astype(int)
    scores = np.round(rng.normal(size=5000) + labels, 1)
    assert _audit(tmp_path).check_metric_implementation(scores, labels, "synthetic").status == PASS
