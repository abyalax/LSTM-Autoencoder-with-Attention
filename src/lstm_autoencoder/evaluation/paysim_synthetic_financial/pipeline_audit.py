"""P0 pipeline audit: causality of history features, sequence construction, split leakage, metric consistency."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from lstm_autoencoder.config import DatasetConfig
from lstm_autoencoder.data.loader import DatasetLoader
from lstm_autoencoder.data.paysim_synthetic_financial.causal_features import (
    ACCOUNT_CODE,
    AMOUNT_EPSILON,
    CASH_OUT_TYPE,
    HISTORY_BUCKET,
    HISTORY_LENGTH,
    IS_NEW_RECEIVER,
    LOG_AMOUNT_RATIO,
    STEP_DIFF,
    CausalFeatureBuilder,
    account_identity,
)
from lstm_autoencoder.data.paysim_synthetic_financial.schema import PaySimSchema
from lstm_autoencoder.evaluation.metrics import RankingMetrics
from lstm_autoencoder.model.config import SequenceConfig
from lstm_autoencoder.model.feature_scaler import FeatureScaler
from lstm_autoencoder.model.sequence_builder import SequenceBuilder
from lstm_autoencoder.reporting.reporter import ReportWritter

ROW_ID = "_row_id"
DERIVED = (HISTORY_LENGTH, STEP_DIFF, LOG_AMOUNT_RATIO, IS_NEW_RECEIVER, HISTORY_BUCKET)
BUCKET_LABELS = ("1", "2-4", "5-9", ">=10")
Splitter = Callable[[pd.DataFrame], dict[str, pd.Index]]

PASS, FAIL, INFO = "pass", "fail", "info"


@dataclass(frozen=True)
class AuditFinding:
    """Outcome of one audit check: pass/fail for hard invariants, info for measurements."""

    name: str
    status: str
    detail: dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        """Plain-dict form for the JSON report."""
        return {"name": self.name, "status": self.status, **self.detail}


@dataclass(frozen=True)
class PipelineAuditConfig:
    """Settings of the audit. `sample_receivers` bounds the per-row reference checks."""

    windows: tuple[int, ...] = (1, 3, 5, 10)
    sample_receivers: int = 3000
    seed: int = 42
    atol: float = 1e-7


class PipelineAudit:
    """Verify, by independent reimplementation and perturbation, that the data pipeline is sound.

    Every hard invariant is a pass/fail check on the real data:

      1. *reference*    the history features equal a slow, loop-based reimplementation;
      2. *future-free*  changing every transaction AFTER a cut point of a receiver leaves
                        the features of all earlier transactions of that receiver unchanged
                        (the operational definition of "uses only past information");
      3. *isolation*    a receiver's features do not depend on other receivers' rows;
      4. *sequences*    windows, right padding and masks equal a naive reimplementation, windows
                        never contain a same-step peer, the build is deterministic, and masks
                        are structurally valid;
      5. *ties*         transactions share an hourly `step`, so their relative order is not
                        observable: features must not change when same-step rows are reordered;
      6. *split*        receiver sets are disjoint, the split does not depend on any label,
                        scaler statistics come from normal train rows only.

    and a few measurements that bound what the pipeline can claim:

      - receiver semantics: how much of the TRANSFER -> CASH_OUT mule path the receiver-keyed
        sequences actually cover (the sequence entity is `nameDest` for every row);
      - context contamination: normal training windows may contain earlier FRAUD steps;
      - metric implementation vs scikit-learn; feature-set mismatch between baselines and the AE;
      - hour-of-day structure of fraud (a simulator artefact tree models can exploit).
    """

    def __init__(
        self,
        dataset_config: DatasetConfig,
        config: PipelineAuditConfig | None = None,
        reporter: ReportWritter | None = None,
    ) -> None:
        """Bind the audit to a dataset configuration."""
        self._dataset_config = dataset_config
        self._config = config or PipelineAuditConfig()
        self._schema = PaySimSchema()
        self._reporter = reporter or ReportWritter(dataset_config)
        self._reporter.set_path_info_from_module()
        self._builder = CausalFeatureBuilder(dataset_config, self._schema)

    # ------------------------------------------------------------------ orchestration

    def run(self) -> dict[str, object]:
        """Run every check on the real dataset, write the report, and return it."""
        loader = DatasetLoader(self._dataset_config, self._schema)
        loader._parse_timestamps = lambda series: series
        loader._normalize_target = lambda series: series
        frame = loader.load().frame.reset_index(drop=True)
        rng = np.random.default_rng(self._config.seed)
        sample = self._sample_receivers(frame, rng)

        self._reporter.display_section("Pipeline Audit (P0)")
        self._reporter.display_text(
            f"rows {len(frame):,} | sampled rows {len(sample):,} "
            f"({account_identity(sample, self._schema).nunique():,} receivers) | seed {self._config.seed}"
        )

        findings: list[AuditFinding] = [
            self.check_feature_reference(sample),
            self.check_future_independence(sample, rng),
            self.check_receiver_isolation(frame, sample),
            self.check_tie_sensitivity(sample, rng),
            self.check_time_structure(frame),
            self.check_receiver_semantics(frame),
        ]
        built = self._builder.build(sample.assign(**{ROW_ID: np.arange(len(sample))}))
        for window in self._config.windows:
            findings.append(self.check_sequences(built, window, rng))

        findings.extend(self._split_findings(frame))
        findings.extend(self._metric_findings(frame))

        failed = [finding.name for finding in findings if finding.status == FAIL]
        self._report(findings, failed)
        return {"passed": not failed, "failed": failed, "findings": [finding.as_dict() for finding in findings]}

    # ------------------------------------------------------------------ feature checks

    def check_feature_reference(self, sample: pd.DataFrame) -> AuditFinding:
        """Compare the vectorised features with a per-receiver Python loop (independent reimplementation)."""
        built = self._build(sample)
        expected = self._reference_features(sample)
        merged = built.set_index(ROW_ID)[list(DERIVED)].join(expected.add_suffix("_ref"))
        mismatches: dict[str, int] = {}
        for column in (HISTORY_LENGTH, STEP_DIFF, IS_NEW_RECEIVER, HISTORY_BUCKET):
            mismatches[column] = int((merged[column].astype(str) != merged[f"{column}_ref"].astype(str)).sum())
        ratio_gap = (merged[LOG_AMOUNT_RATIO] - merged[f"{LOG_AMOUNT_RATIO}_ref"]).abs()
        mismatches[LOG_AMOUNT_RATIO] = int((ratio_gap > self._config.atol).sum())
        ok = sum(mismatches.values()) == 0
        return AuditFinding(
            "feature_reference",
            PASS if ok else FAIL,
            {"rows": len(merged), "mismatches": mismatches, "max_abs_log_ratio_gap": float(ratio_gap.max())},
        )

    def check_future_independence(self, sample: pd.DataFrame, rng: np.random.Generator) -> AuditFinding:
        """Perturb every transaction after a random per-receiver cut; earlier features must not move."""
        schema = self._schema
        base = sample.assign(**{ROW_ID: np.arange(len(sample))})
        built = self._builder.build(base)
        position = built.groupby(ACCOUNT_CODE).cumcount()
        sizes = built.groupby(ACCOUNT_CODE).size()
        # One cut per RECEIVER (the last kept position); a per-row cut would scatter the perturbation
        # through the middle of a history and reorder it, which is not a "future" perturbation.
        cut = built[ACCOUNT_CODE].map(pd.Series(rng.integers(0, sizes.to_numpy()), index=sizes.index))
        future_ids = built.loc[position > cut, ROW_ID].to_numpy()

        perturbed = base.copy()
        future = perturbed[ROW_ID].isin(future_ids)
        perturbed.loc[future, schema.amount_column] = perturbed.loc[future, schema.amount_column] * 7.0 + 13.0
        perturbed.loc[future, schema.timestamp_column] = perturbed.loc[future, schema.timestamp_column] + 5
        perturbed.loc[future, schema.target_column] = 1 - perturbed.loc[future, schema.target_column]
        rebuilt = self._builder.build(perturbed)

        past_ids = built.loc[position <= cut, ROW_ID].to_numpy()
        before = built.set_index(ROW_ID).loc[past_ids, list(DERIVED)]
        after = rebuilt.set_index(ROW_ID).loc[past_ids, list(DERIVED)]
        moved = {
            column: int((before[column].astype(str) != after[column].astype(str)).sum())
            if column in (HISTORY_BUCKET,)
            else int(((before[column] - after[column]).abs() > self._config.atol).sum())
            for column in DERIVED
        }
        return AuditFinding(
            "future_independence",
            PASS if sum(moved.values()) == 0 else FAIL,
            {"rows_checked": len(past_ids), "future_rows_perturbed": int(len(future_ids)), "features_moved": moved},
        )

    def check_receiver_isolation(self, frame: pd.DataFrame, sample: pd.DataFrame) -> AuditFinding:
        """Features of the sampled receivers must be the same whether built alone or inside the full dataset."""
        full = self._builder.build(frame.assign(**{ROW_ID: np.arange(len(frame))}))
        original_ids = sample.index.to_numpy()
        full_part = full[full[ROW_ID].isin(original_ids)].set_index(ROW_ID)[list(DERIVED)].sort_index()
        alone = self._builder.build(sample.assign(**{ROW_ID: sample.index.to_numpy()}))
        alone = alone.set_index(ROW_ID)[list(DERIVED)].sort_index()
        same_index = full_part.index.equals(alone.index)
        moved = (
            {
                column: int((full_part[column].astype(str) != alone[column].astype(str)).sum())
                if column == HISTORY_BUCKET
                else int(((full_part[column] - alone[column]).abs() > self._config.atol).sum())
                for column in DERIVED
            }
            if same_index
            else {"index": -1}
        )
        return AuditFinding(
            "receiver_isolation",
            PASS if same_index and sum(moved.values()) == 0 else FAIL,
            {"receivers": int(account_identity(sample, self._schema).nunique()), "features_moved": moved},
        )

    def check_tie_sensitivity(self, sample: pd.DataFrame, rng: np.random.Generator) -> AuditFinding:
        """Hard check: features must NOT depend on the unobservable order of same-`step` transactions."""
        base = sample.assign(**{ROW_ID: np.arange(len(sample))})
        built = self._builder.build(base)
        shuffled = base.iloc[rng.permutation(len(base))]
        rebuilt = self._builder.build(shuffled)

        step = self._schema.timestamp_column
        account = account_identity(base, self._schema)
        peers = base.groupby([account, base[step]])[step].transform("size") > 1
        a = built.set_index(ROW_ID).sort_index()
        b = rebuilt.set_index(ROW_ID).sort_index()
        changed = {
            HISTORY_LENGTH: float((a[HISTORY_LENGTH] != b[HISTORY_LENGTH]).mean()),
            STEP_DIFF: float((a[STEP_DIFF] != b[STEP_DIFF]).mean()),
            LOG_AMOUNT_RATIO: float(((a[LOG_AMOUNT_RATIO] - b[LOG_AMOUNT_RATIO]).abs() > self._config.atol).mean()),
            IS_NEW_RECEIVER: float((a[IS_NEW_RECEIVER] != b[IS_NEW_RECEIVER]).mean()),
            HISTORY_BUCKET: float((a[HISTORY_BUCKET] != b[HISTORY_BUCKET]).mean()),
        }
        target = self._schema.target_column
        return AuditFinding(
            "tie_invariance",
            PASS if all(value == 0.0 for value in changed.values()) else FAIL,
            {
                "rows": len(built),
                "share_rows_with_same_step_peer": float(peers.mean()),
                "fraud_rate_rows_with_peer": float(base.loc[peers, target].mean()) if peers.any() else float("nan"),
                "fraud_rate_rows_without_peer": float(base.loc[~peers, target].mean()),
                "share_rows_changed_when_all_rows_are_shuffled": changed,
                "reading": (
                    "`step` is hourly, so the order of same-step transactions of one receiver is not observable. "
                    "History features use only strictly earlier steps and windows exclude same-step peers, so "
                    "shuffling the input rows must change nothing."
                ),
            },
        )

    def check_receiver_semantics(self, frame: pd.DataFrame) -> AuditFinding:
        """How often the TRANSFER -> CASH_OUT mule path is actually visible to the role-aware sequences.

        The sequence entity is role-aware (nameDest for TRANSFER, nameOrig for CASH_OUT), so a CASH_OUT
        row can only inherit history from an earlier TRANSFER-in when its origin account earlier received
        a TRANSFER. The shares below bound how much of the mule narrative the data can support; the paper
        must not claim more than that coverage.
        """
        schema = self._schema
        type_col = schema.transaction_type_column
        if type_col not in frame.columns:
            return AuditFinding("receiver_semantics", INFO, {"skipped": f"no {type_col} column"})
        step, label = schema.timestamp_column, schema.target_column
        transfers = frame.loc[frame[type_col] == "TRANSFER"]
        first_received = transfers.groupby(schema.receiver_column)[step].min()
        cash_out = frame.loc[frame[type_col] == "CASH_OUT"]
        seen_at = cash_out[schema.sender_column].map(first_received)
        earlier_receiver = (seen_at < cash_out[step]).fillna(False)
        fraud = cash_out[label] == 1
        return AuditFinding(
            "receiver_semantics",
            INFO,
            {
                "sequence_entity": "role-aware account: nameDest for TRANSFER, nameOrig for CASH_OUT",
                "cash_out_rows": int(len(cash_out)),
                "share_cash_out_origin_was_earlier_transfer_receiver": float(earlier_receiver.mean()),
                "fraud_cash_out_rows": int(fraud.sum()),
                "share_fraud_cash_out_origin_was_earlier_transfer_receiver": (
                    float(earlier_receiver[fraud].mean()) if fraud.any() else float("nan")
                ),
                "reading": (
                    "A low share means most CASH_OUT rows have no earlier TRANSFER-in on the same account, so "
                    "their windows hold no mule-path context; state this coverage in Methods/Limitations."
                ),
            },
        )

    def check_time_structure(self, frame: pd.DataFrame) -> AuditFinding:
        """
        Hour-of-day profile of fraud vs normal;
        a flat fraud profile against a diurnal normal one is a simulator artefact.
        """
        step = frame[self._schema.timestamp_column].to_numpy()
        label = frame[self._schema.target_column].to_numpy()
        hour = (step % 24).astype(int)
        rate = pd.Series(label).groupby(hour).mean()
        share_normal = pd.Series(1 - label).groupby(hour).sum() / (1 - label).sum()
        share_fraud = pd.Series(label).groupby(hour).sum() / max(int(label.sum()), 1)
        return AuditFinding(
            "time_structure",
            INFO,
            {
                "fraud_rate_by_hour_min": float(rate.min()),
                "fraud_rate_by_hour_max": float(rate.max()),
                "fraud_rate_max_over_min": float(rate.max() / rate.min()) if rate.min() > 0 else float("inf"),
                "normal_share_by_hour_max_over_min": float(share_normal.max() / share_normal.min()),
                "fraud_share_by_hour_max_over_min": float(share_fraud.max() / max(share_fraud.min(), 1e-12)),
                "reading": (
                    "Baselines receive the raw `step` column; tree models can turn it into hour-of-day. "
                    "A large fraud-rate spread across hours means part of their advantage is a simulator "
                    "artefact the five AE features cannot use."
                ),
            },
        )

    # ------------------------------------------------------------------ sequence checks

    @staticmethod
    def _step_positions(built: pd.DataFrame, step_column: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(account_start, step_start, earlier) row positions of a frame sorted by (account, step).

        `earlier` = number of rows of the same account at STRICTLY earlier steps.
        """
        codes = built[ACCOUNT_CODE].to_numpy()
        step = built[step_column].to_numpy()
        n = len(built)
        position = np.arange(n)
        new_account = np.ones(n, dtype=bool)
        new_step = np.ones(n, dtype=bool)
        if n > 1:
            new_account[1:] = codes[1:] != codes[:-1]
            new_step[1:] = new_account[1:] | (step[1:] != step[:-1])
        account_start = np.maximum.accumulate(np.where(new_account, position, 0))
        step_start = np.maximum.accumulate(np.where(new_step, position, 0))
        return account_start, step_start, step_start - account_start

    def check_sequences(self, built: pd.DataFrame, window: int, rng: np.random.Generator) -> AuditFinding:
        """Windows, padding and masks vs a naive reimplementation; determinism; structural validity."""
        config = SequenceConfig(window_length=window)
        columns = list(config.feature_columns)
        builder = SequenceBuilder(config, self._schema)
        batch = builder.build(built)
        again = builder.build(built)
        deterministic = all(
            np.array_equal(getattr(batch, name), getattr(again, name))
            for name in ("sequences", "mask", "labels", "history_bucket", "receiver")
        )

        step_column = self._schema.timestamp_column
        _, step_start, earlier = self._step_positions(built, step_column)
        eligible = np.flatnonzero(earlier >= config.min_history_length - 1)
        count_ok = len(batch.labels) == len(eligible)

        mask = batch.mask
        structure = {
            "mask_is_binary": bool(np.isin(mask, (0.0, 1.0)).all()),
            "padding_is_right_only": bool((np.diff(mask, axis=1) <= 0).all()),
            "first_step_is_always_real": bool((mask[:, 0] == 1.0).all()),
            "padded_features_are_zero": bool((batch.sequences[mask == 0.0] == 0.0).all()),
            "real_steps_equal_earlier_steps_plus_current": bool(
                (mask.sum(axis=1) == np.minimum(earlier[eligible] + 1, window)).all()
            ),
        }

        features = built[columns].to_numpy(dtype="float32")
        codes = built[ACCOUNT_CODE].to_numpy()
        step = built[step_column].to_numpy()
        picks = rng.choice(len(eligible), size=min(2000, len(eligible)), replace=False)
        wrong = 0
        for k in picks:
            end = eligible[k]
            n_prior = int(min(earlier[end], window - 1))
            prior_rows = np.arange(step_start[end] - n_prior, step_start[end])
            rows = np.concatenate([prior_rows, [end]]).astype(int)
            expected = np.zeros((window, len(columns)), dtype="float32")
            expected[: len(rows)] = features[rows]
            ok = np.array_equal(batch.sequences[k], expected)
            ok &= bool((step[prior_rows] < step[end]).all())  # no same-step peer, no later row
            ok &= bool((codes[rows] == codes[end]).all())  # one receiver only
            ok &= batch.labels[k] == built[self._schema.target_column].iloc[end]
            ok &= batch.history_bucket[k] == built[HISTORY_BUCKET].iloc[end]
            ok &= batch.receiver[k] == built[self._schema.receiver_column].iloc[end]
            wrong += int(not ok)

        passed = deterministic and count_ok and all(structure.values()) and wrong == 0
        return AuditFinding(
            f"sequences_w{window}",
            PASS if passed else FAIL,
            {
                "examples": int(len(batch.labels)),
                "deterministic_rebuild": deterministic,
                "example_count_matches_eligible_rows": count_ok,
                "structure": structure,
                "sampled_examples_checked": int(len(picks)),
                "sampled_examples_wrong": wrong,
            },
        )

    # ------------------------------------------------------------------ split / leakage checks

    def check_split(
        self,
        built: pd.DataFrame,
        receiver_ids: dict[str, pd.Index],
        windows: tuple[int, ...],
        tolerance: float = 1e-6,
    ) -> list[AuditFinding]:
        """Disjointness, scaler provenance, and training-context contamination for one receiver split."""
        schema = self._schema
        receivers = built[schema.receiver_column]
        names = list(receiver_ids)
        overlaps = {
            f"{a}&{b}": len(set(receiver_ids[a]) & set(receiver_ids[b]))
            for i, a in enumerate(names)
            for b in names[i + 1 :]
        }
        assigned = pd.concat([pd.Series(1, index=pd.Index(ids)) for ids in receiver_ids.values()])
        covers_all = set(receivers.unique()) == set(assigned.index)
        rows = {name: int(receivers.isin(ids).sum()) for name, ids in receiver_ids.items()}
        fraud = {
            name: int(built.loc[receivers.isin(ids), schema.target_column].sum()) for name, ids in receiver_ids.items()
        }
        split_ok = AuditFinding(
            "split_disjoint",
            PASS
            if all(v == 0 for v in overlaps.values()) and covers_all and sum(rows.values()) == len(built)
            else FAIL,
            {
                "receiver_overlaps": overlaps,
                "every_receiver_assigned": covers_all,
                "rows": rows,
                "fraud_rows": fraud,
                "fraud_rate": {k: fraud[k] / rows[k] if rows[k] else float("nan") for k in rows},
            },
        )

        train_normal = receivers.isin(receiver_ids["train"]) & (built[schema.target_column] == 0)
        config = SequenceConfig()
        log1p_columns = (schema.amount_column, HISTORY_LENGTH)
        scaler = FeatureScaler(config.feature_columns, log1p_columns).fit(built, train_normal)
        scaled = scaler.transform(built)
        columns = list(config.feature_columns)
        on_fit = scaled.loc[train_normal, columns]
        provenance = {
            "train_normal_mean_abs_max": float(on_fit.mean().abs().max()),
            "train_normal_std_minus_1_abs_max": float((on_fit.std(ddof=0) - 1.0).abs().max()),
        }
        drift = {}
        for name in names:
            part = scaled.loc[receivers.isin(receiver_ids[name]) & (built[schema.target_column] == 0), columns]
            drift[name] = {
                "mean_abs_max": float(part.mean().abs().max()),
                "std_ratio_range": [float(part.std(ddof=0).min()), float(part.std(ddof=0).max())],
            }
        scaler_ok = AuditFinding(
            "scaler_fit_on_train_normal_only",
            PASS
            if provenance["train_normal_mean_abs_max"] < tolerance
            and provenance["train_normal_std_minus_1_abs_max"] < tolerance
            else FAIL,
            {**provenance, "scaled_normal_rows_per_split": drift},
        )

        normal_rows = np.flatnonzero(train_normal.to_numpy())
        findings = [split_ok, scaler_ok]
        for window_length in windows:
            findings.append(
                AuditFinding(
                    f"training_context_contamination_w{window_length}",
                    INFO,
                    self._context_contamination(built, normal_rows, window_length),
                )
            )
        return findings

    def _context_contamination(self, built: pd.DataFrame, normal_rows: np.ndarray, window: int) -> dict[str, object]:
        """Share of normal training windows whose EARLIER steps include a fraud transaction.

        The autoencoder is trained on windows whose LAST transaction is normal; with
        window > 1 the earlier steps of such a window may still be fraud, so the training
        set is "normal at the label step", not "free of fraud context".
        """
        if window == 1 or len(normal_rows) == 0:
            return {"normal_train_windows": int(len(normal_rows)), "share_with_fraud_in_context": 0.0}
        label = built[self._schema.target_column].to_numpy().astype("int64")
        codes = built[ACCOUNT_CODE].to_numpy()
        _, step_start, earlier = self._step_positions(built, self._schema.timestamp_column)
        cumulative = pd.Series(label).groupby(codes).cumsum().to_numpy()  # inclusive, per receiver
        in_window_before = np.minimum(earlier, window - 1)  # earlier-step rows inside the window
        last_prior = step_start - 1  # last row of the previous step group
        has_prior = in_window_before > 0
        before_window = np.where(
            earlier - in_window_before > 0, cumulative[np.maximum(last_prior - in_window_before, 0)], 0
        )
        prior_fraud = np.where(has_prior, cumulative[np.maximum(last_prior, 0)] - before_window, 0)
        fraud_in_context = prior_fraud
        return {
            "normal_train_windows": int(len(normal_rows)),
            "share_with_fraud_in_context": float((fraud_in_context[normal_rows] > 0).mean()),
        }

    def _split_findings(self, frame: pd.DataFrame) -> list[AuditFinding]:
        """Run the split checks with the pipeline's own splitter, and compare it with the baseline's."""
        from lstm_autoencoder.evaluation.config import SplitConfig
        from lstm_autoencoder.evaluation.paysim_synthetic_financial.baseline import BaselineEvaluator
        from lstm_autoencoder.pipeline.paysim.paysim_pipeline import PaySimModelTrainingPipeline

        built = self._builder.build(frame)
        pipeline = PaySimModelTrainingPipeline(
            dataset_config=self._dataset_config, sequence_config=SequenceConfig(), split_config=SplitConfig()
        )
        receiver_ids = pipeline.split_receivers(built)
        findings = self.check_split(built, receiver_ids, self._config.windows)

        findings.append(self._label_independence(built, pipeline, receiver_ids))

        baseline = BaselineEvaluator(self._dataset_config)  # own reporter: sharing ours would redirect the report path
        rows = baseline.split_rows(built)
        receiver = built[self._schema.receiver_column]
        same = {name: set(receiver.iloc[rows[name]].unique()) == set(receiver_ids[name]) for name in receiver_ids}
        findings.append(
            AuditFinding(
                "split_identical_in_baseline_and_ae",
                PASS if all(same.values()) else FAIL,
                {"receiver_sets_equal": same},
            )
        )
        return findings

    def _label_independence(
        self, built: pd.DataFrame, pipeline: object, receiver_ids: dict[str, pd.Index]
    ) -> AuditFinding:
        """The receiver split must be identical when every label is replaced by random labels."""
        target = self._schema.target_column
        rng = np.random.default_rng(self._config.seed + 11)
        scrambled = built.assign(**{target: rng.integers(0, 2, size=len(built))})
        again = pipeline.split_receivers(scrambled)
        same = {name: set(again[name]) == set(receiver_ids[name]) for name in receiver_ids}
        return AuditFinding(
            "split_is_label_free",
            PASS if all(same.values()) else FAIL,
            {"receiver_sets_equal_under_random_labels": same},
        )

    # ------------------------------------------------------------------ metric / comparability checks

    def check_metric_implementation(self, scores: np.ndarray, labels: np.ndarray, name: str) -> AuditFinding:
        """`RankingMetrics` AUC/AP must equal scikit-learn on the same scores and labels."""
        from sklearn.metrics import average_precision_score, roc_auc_score

        auc_gap = abs(RankingMetrics.roc_auc(scores, labels) - roc_auc_score(labels, scores))
        ap_gap = abs(RankingMetrics.average_precision(scores, labels) - average_precision_score(labels, scores))
        return AuditFinding(
            f"metric_implementation_{name}",
            PASS if auc_gap < 1e-9 and ap_gap < 1e-9 else FAIL,
            {
                "abs_gap_auc": float(auc_gap),
                "abs_gap_ap": float(ap_gap),
                "n": int(len(labels)),
                "prevalence": float(np.mean(labels)),
            },
        )

    def _metric_findings(self, frame: pd.DataFrame) -> list[AuditFinding]:
        """Metric implementation on real LR scores, plus the feature-set comparison across models."""
        from lstm_autoencoder.evaluation.paysim_synthetic_financial.baseline import BaselineEvaluator

        baseline = BaselineEvaluator(self._dataset_config)  # own reporter, see `_split_findings`
        _, predicted, labels = baseline.fit_full_variant()
        data = baseline.prepare_full_features()
        ae_features = set(SequenceConfig().feature_columns)
        baseline_features = set(data.features.columns)
        return [
            self.check_metric_implementation(predicted, labels, "lr_full_test"),
            AuditFinding(
                "feature_set_comparability",
                INFO,
                {
                    "ae_features": sorted(ae_features),
                    "baseline_features": sorted(baseline_features),
                    "seen_only_by_baselines": sorted(baseline_features - ae_features),
                    "seen_only_by_ae": sorted(ae_features - baseline_features),
                    "reading": (
                        "LR, gradient boosting and Isolation Forest see columns the LSTM-AE does not "
                        "(raw `step`, transaction type). Compare the AE with the `ae_features` baseline "
                        "variants (same five columns) for a feature-matched comparison."
                    ),
                },
            ),
        ]

    # ------------------------------------------------------------------ helpers

    def _sample_receivers(self, frame: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
        """All rows of a receiver sample: half uniformly random, half the receivers with the longest history.

        Samples based on the role-aware account identity (nameDest for TRANSFER, nameOrig for CASH_OUT)
        to match the receiver transformation done by CausalFeatureBuilder.
        """
        account_code = account_identity(frame, self._schema)
        counts = account_code.value_counts()
        n = min(self._config.sample_receivers, len(counts))
        longest = counts.index[: n // 2]
        rest = counts.index.difference(longest)
        random_part = rng.choice(rest.to_numpy(), size=min(n - len(longest), len(rest)), replace=False)
        chosen = set(longest) | set(random_part)
        return frame[account_code.isin(chosen)]

    def _build(self, sample: pd.DataFrame) -> pd.DataFrame:
        """Build features with a stable row id carried through the builder's re-sorting."""
        return self._builder.build(sample.assign(**{ROW_ID: np.arange(len(sample))}))

    def _reference_features(self, sample: pd.DataFrame) -> pd.DataFrame:
        """Loop-based reimplementation of the history features, one receiver at a time.

        A transaction's history is the set of the same ACCOUNT's rows (role-aware identity: nameDest for
        TRANSFER, nameOrig for CASH_OUT) at STRICTLY earlier steps.
        """
        schema = self._schema
        rows = sample.assign(**{ROW_ID: np.arange(len(sample))})
        is_cash_out = rows[schema.transaction_type_column] == CASH_OUT_TYPE
        rows = rows.assign(_account=rows[schema.receiver_column].where(~is_cash_out, rows[schema.sender_column]))
        out: list[pd.DataFrame] = []
        for _, group in rows.groupby("_account", sort=False):
            group = group.sort_values(schema.timestamp_column, kind="stable")
            amounts = group[schema.amount_column].to_numpy(dtype="float64")
            steps = group[schema.timestamp_column].to_numpy()
            history = np.zeros(len(group), dtype="int64")
            diff = np.zeros(len(group), dtype="int64")
            ratio = np.zeros(len(group))
            for i in range(len(group)):
                earlier = np.flatnonzero(steps < steps[i])
                history[i] = len(earlier)
                if len(earlier):
                    diff[i] = steps[i] - steps[earlier].max()
                    ratio[i] = np.log((amounts[i] + AMOUNT_EPSILON) / (amounts[earlier].mean() + AMOUNT_EPSILON))
            bucket = np.select([history == 0, history < 4, history < 9], ["1", "2-4", "5-9"], default=">=10")
            out.append(
                pd.DataFrame(
                    {
                        ROW_ID: group[ROW_ID].to_numpy(),
                        HISTORY_LENGTH: history,
                        STEP_DIFF: diff,
                        LOG_AMOUNT_RATIO: ratio,
                        IS_NEW_RECEIVER: (history == 0).astype("int8"),
                        HISTORY_BUCKET: bucket,
                    }
                )
            )
        return pd.concat(out).set_index(ROW_ID)

    def _report(self, findings: list[AuditFinding], failed: list[str]) -> None:
        """Print a one-line-per-check table and write the text/JSON reports."""
        table = pd.DataFrame(
            [
                {"check": finding.name, "status": finding.status.upper(), "detail": self._one_line(finding)}
                for finding in findings
            ]
        )
        self._reporter.display_table(table)
        self._reporter.display_text("ALL HARD CHECKS PASSED" if not failed else f"FAILED CHECKS: {', '.join(failed)}")
        self._reporter.update_summary(
            {
                "pipeline_audit": {
                    "seed": self._config.seed,
                    "sample_receivers": self._config.sample_receivers,
                    "windows": list(self._config.windows),
                    "passed": not failed,
                    "failed": failed,
                    "findings": [finding.as_dict() for finding in findings],
                }
            }
        )
        self._reporter.save_reports(
            text_filename="pipeline_audit_report.txt", json_filename="pipeline_audit_report.json"
        )

    @staticmethod
    def _one_line(finding: AuditFinding) -> str:
        """Compact detail string for the console table (full detail lives in the JSON)."""
        keys = [k for k in finding.detail if k != "reading"][:3]
        return "; ".join(f"{k}={finding.detail[k]}" for k in keys)[:140]
