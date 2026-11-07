# Roadmap

> Last updated: 2026-11-07 · Current phase: **experiment gaps (P6) + IMRaD rewrite** (all workflows rerun with `run_config` at commit `f6bfe93`; grid workflows are CLI commands since `b0224a6`)
>
> Note (2026-11-07): the numbers in P2 to P3.5 below are historical (bottleneck-only runs and the pre-fix baseline). Current numbers live in `artifacts/summary/results_summary.md` (`summarize-results`); open work is in P6. Convention: $W$ = window length, grid $W \\in {1,3,5,10}$. AUC/AP are on the test split unless stated.

## P0. Data Redesign (nameDest windowing + scope filter) — DONE

### P0.0 Schema & scope update

- [x] Update `paysim_synthetic_financial/schema.py`: restrict to `transaction_type in {CASH_OUT, TRANSFER}`
- [x] Drop leakage columns at schema level (`oldbalanceOrg`, `newbalanceOrig`, `oldbalanceDest`, `newbalanceDest`)

### P0.1 Windowing unit change

- [x] In `causal_features.py`: switch windowing group key from `nameOrig` to `nameDest`
- [x] Change window grid from `{5,10,20}` to `{1,3,5,10}` — W=5 viability 39.63% (vs \~10% unscoped estimate)

### P0.2 New explicit features

- [x] `history_length` (count of prior receiver transactions at time t) — already in `CausalFeatureBuilder` (named `history_length`, not `history_len`)
- [x] `is_new_receiver` — already in `CausalFeatureBuilder`
- [x] `history_bucket` categorical (`1`, `2-4`, `5-9`, `>=10`) for stratified reporting

### P0.3 Split fraud-rate gap — RESOLVED

- [x] Receiver-stratified split: train 0.2960% / val 0.2981% / test 0.2969%, overall prevalence 0.2965%

## P1. Exploratory Re-validation — DONE

- [x] Re-ran `inspect-data` on CASH_OUT+TRANSFER-only, receiver-keyed dataset
- [x] Window viability: W=5 39.63%, W=10 17.03%; fraud coverage W=5 38.4%, W=10 20.41%
- [x] Receiver stats: min=1, p25=1, median=3.0, p75=7, p95=17.0, max=75

## P2. Baseline & Ablation Setup — DONE

- [x] `full` LR: test AUC 0.8625 (HistGradientBoosting probe: 0.8825)
- [x] `no_history` LR: test AUC 0.797 — history is worth \~0.065 AUC
- [x] `history_only` LR: test AUC 0.765 — \~entirely between-bucket separation (bucket "1" AUC = 0.50 exactly)
- [x] Stratified reporting via `BaselineEvaluator` (AUC/AP per `history_bucket`, all 3 variants)

## P3. Model Architecture (Hybrid LSTM-AE + Attention)

### P3.0 Model implementation — DONE

- [x] `model/lstm_autoencoder.py`: Encoder, masked SelfAttention, Decoder, `reconstruction_error()`
- [x] `history_length`/`is_new_receiver` wired as input features (not auxiliary target)

### P3.1 Training pipeline + scaler fix — DONE

- [x] `config.py`, `pipeline.py`, `trainer.py`, `sequence_builder.py`, `dataset.py`
- [x] Bug 1 (unnormalized `amount` dominating MSE) fixed with `FeatureScaler` (fit on train-normal only)

### P3.2 Architecture fixes — DONE

- [x] Bug 2: SelfAttention could copy itself → diagonal mask (fallback to self only when 1 real step)
- [x] Bug 3: no `torch.manual_seed` → seeded `Trainer`, `--seeds 42,123,2026`, mean ± std
- [x] True bottleneck mode (`--bottleneck`): decoder only sees the final hidden state

### P3.3 Standalone LSTM-AE results (3 seeds) — DONE, **verdict: does not beat baselines**

| $W$ | attention (diag-mask) AUC | bottleneck AUC  | bottleneck AP   |
| --- | ------------------------- | --------------- | --------------- |
| 1   | 0.7199 ± 0.0224           | 0.7200 ± 0.0229 | 0.0180 ± 0.0071 |
| 3   | 0.6257 ± 0.0691           | 0.6239 ± 0.0151 | 0.0091 ± 0.0008 |
| 5   | 0.5713 ± 0.0870           | 0.4563 ± 0.0747 | 0.0050 ± 0.0021 |
| 10  | 0.4724 ± 0.0669           | 0.4168 ± 0.0200 | 0.0031 ± 0.0003 |

- [x] Decision: standalone LSTM-AE is **not** pursued as a stand-alone detector (best $0.72$ \< `history_only` $0.765$ \< `full` LR $0.8625$). Pivot to hybrid: reconstruction error as an extra feature for `full` LR.

### P3.4 Hybrid signal probe — IN PROGRESS

Probe results (bottleneck, seed 42, `n_test` = 415,074, LR test AUC 0.8625):

| $W$ | AE AUC | AE AP  | Pearson $r$ | Spearman $\\rho$ | err normal | err fraud_lr_low | err fraud_lr_high | low / normal | high / normal |
| --- | ------ | ------ | ----------- | ---------------- | ---------- | ---------------- | ----------------- | ------------ | ------------- |
| 1   | 0.6965 | 0.0138 | 0.0148      | 0.1505           | 0.0038     | 0.3305           | 0.0075            | 87.0×        | 2.0×          |
| 3   | 0.6068 | 0.0101 | 0.0135      | -0.1184          | 0.0050     | 0.1528           | 0.0145            | 30.4×        | 2.9×          |
| 5   | 0.3864 | 0.0027 | 0.0189      | -0.3534          | 0.1502     | 0.2475           | 0.1325            | 1.6×         | 0.9×          |
| 10  | 0.4385 | 0.0029 | 0.0059      | -0.4154          | 0.0208     | 0.0970           | 0.0271            | 4.7×         | 1.3×          |

Reading:

- $W=1$ is clearly best: error is concentrated on fraud that LR misses (87× normal) and barely elevated on fraud LR already catches (2×) → complementary signal. $W=3$ keeps the same pattern, weaker (30×).
- $W \\ge 5$ is not usable as-is: $\\rho$ goes monotonically negative as $W$ grows (0.15 → -0.12 → -0.35 → -0.42), and at $W=5$ fraud_lr_high error is *below* normal. The $W=5$ checkpoint is from epoch 1 (nearly untrained, normal error 0.15 vs 0.005 at $W=3$).
- Seed 42 is the weakest seed at $W=1$ (0.6965 vs 0.7212 / 0.7423), so $W=1$ numbers are conservative.

To do:

- [x] `probe-hybrid-signal` for $W=1,3,5,10$ (seed 42)
- [x] `hybrid_signal_probe.py` and the `probe-hybrid-signal` CLI command are on `origin/main`
- [ ] Re-run probe on seeds 123 and 2026 for $W=1$ and $W=3$, report mean ± std (single-seed is fragile, std up to 0.087 at $W \\ge 3$)
- [ ] Document the `fraud_lr_low` / `fraud_lr_high` threshold and report $n$ per group, plus median / p90 / p99 error per group (means over heavy-tailed errors can be driven by a few rows)
- [ ] (partly confirmed by P3.5, see below) Diagnose $W \\ge 5$ inversion: correlate error with `real_count` (non-padded steps) and compute AUC of error *within* each `history_bucket` (hypothesis, untested: short-history receivers have mostly padded windows that reconstruct trivially, and LR scores those as high risk)
- [ ] Change AE early stopping from val AUC to val reconstruction loss on normal-only rows (val AUC on inverted windows picks arbitrary epochs, e.g. $W=5$ seed 42 stops at epoch 1), then retrain $W=3,5,10$ and re-probe
- [ ] Confirm what drops the 497 rows between the P0 split summary (415,571) and `n_test` (415,074), and that LR and AE are scored on identical rows
- [x] (done: every training and hybrid report now carries a `run_config` block with architecture, window, seeds and checkpoint names; reports before `20261107_` have none and are legacy) Label reports with architecture flags (e.g. `--bottleneck`) in filename/header; runs `20261105_13xxxx`-`15xxxx` are ambiguous

### P3.5 Hybrid classifier — RUN DONE ($W=1,3,5,10$, 3 seeds), verdict: marginal gain, not a clear win

Test results (bottleneck, 3 seeds, 200 paired bootstraps; reference `baseline_train` AUC 0.8625 / AP 0.0415):

| $W$ | `hybrid_val` $\\Delta$AUC vs `lr_val_control` | CI>0 seeds | `stack_val` $\\Delta$AUC vs baseline | CI>0 seeds | `stack_val` $\\Delta$AP | AE-only AUC |
| --- | --------------------------------------------- | ---------- | ------------------------------------ | ---------- | ----------------------- | ----------- |
| 1   | +0.0013 ± 0.0017                              | 1/3        | +0.0026 ± 0.0017                     | 2/3        | -0.0011                 | 0.7200      |
| 3   | +0.0017 ± 0.0020                              | 1/3        | +0.0036 ± 0.0011                     | 3/3        | -0.0016                 | 0.6239      |
| 5   | +0.0005 ± 0.0004                              | 0/3        | +0.0015 ± 0.0005                     | 3/3        | -0.0010                 | 0.4563      |
| 10  | -0.0000 ± 0.0006                              | 0/3        | +0.0013 ± 0.0004                     | 2/3        | -0.0012                 | 0.4168      |

Reading:

- Gains are above the noise floor (null controls at $\\pm 0.0001$) but tiny: the best is +0.0036 AUC ($W=3$, `stack_val`), about 18% of the gap to the supervised HistGradientBoosting probe (0.8825). AP does not improve: `stack_val` loses 0.001-0.0016, and `hybrid_val` vs baseline is 0.0000 ($W=1$), +0.0019 ($W=3$), about 0 for $W \\ge 5$.
- The leak does not matter here: `hybrid_train_naive` ≈ `hybrid_val` ($W=1$: +0.0013 for both).
- Where it helps: mostly history bucket "1" (823 of 1,235 test fraud), AUC 0.768 → 0.774 ($W=1$, `hybrid_val`); the ">=10" bucket (77 fraud) is noisy and sometimes goes down.
- Recall at 0.1% FPR: `hybrid_val` 0.043 ($W=1$) / 0.045 ($W=3$) vs 0.032 for `lr_val_control`, roughly +13 fraud of 1,235; no CI yet.
- The $W \\ge 5$ inversion is a between-bucket confound: within each history bucket the AE-only AUC is 0.62-0.75 (overall 0.42-0.46), and the error coefficient is positive in every hybrid. Error depends on history length / padding, and fraud concentrates in short histories.
- $W=3$ vs $W=1$ differences are within seed std: no evidence that temporal context helps.

To do:

- [x] (implemented in commit 6befccc: `history_bucket_*` dummies in the LR features, `full_gbm` in `ablation`; results and caveats below) Nonlinear no-AE control: LR + one-hot `history_bucket`, and HistGradientBoosting under the same val protocol, to test whether the error only proxies a nonlinear history effect (HGB alone reaches 0.8825, above every hybrid at ≤ 0.866)
- [x] (implemented as `hybrid_val_norm`, but buggy, see the fix below) History-normalized error: rank or z-score the AE error within `history_bucket` (or per `real_count`), then re-run standalone AUC and the hybrid; targets the confound behind the $W \\ge 5$ inversion
- [x] Add bootstrap CIs for recall at 0.1% / 1% FPR (current recall numbers carry no uncertainty) **After the baseline fix (commit 6befccc): `ablation` plus one `eval-hybrid` run ($W=5$, attention, seed 42 only, 200 bootstraps)**

| model                                                                   | test AUC                                       | test AP             |
| ----------------------------------------------------------------------- | ---------------------------------------------- | ------------------- |
| `full` LR (13 features: + `type_*`, `history_bucket_*`, `account_code`) | 0.8693                                         | 0.0426              |
| `no_history` LR                                                         | 0.8685                                         | 0.0422              |
| `history_only` LR                                                       | 0.7648                                         | 0.0088              |
| `full_gbm` (HistGradientBoosting, 200 iterations, no early stopping)    | 0.7779                                         | 0.1452              |
| `hybrid_val` vs `lr_val_control`                                        | $\\Delta$AUC -0.0012 (CI above 0 in 0/1 seeds) | $\\Delta$AP -0.0002 |
| `stack_val` vs baseline                                                 | $\\Delta$AUC +0.0000                           | $\\Delta$AP -0.0018 |

Reading:

- The feature fix lifts LR from 0.8625 to 0.8693 (+0.0068), more than any earlier hybrid gain (best +0.0036). At $W=5$ the hybrid now adds nothing. This is consistent with the idea that AE error mostly proxied a non-linear history effect, but it is tested only at $W=5$ with the attention variant; $W=1,3$ are not re-run.
- `no_history` ≈ `full` (0.8685 vs 0.8693) only because the `history_bucket_*` dummies stay in `no_history`, so that ablation no longer removes history information.
- `full_gbm` has lower AUC than LR (0.778 vs 0.869) but 3.4× the AP (0.145 vs 0.043). Its within-bucket AUC is only 0.59-0.60 for buckets "2-4", "5-9", ">=10" (LR: 0.73-0.77). It trains with `account_code` and no early stopping, so overfitting to an id/time proxy is plausible (not tested).
- `hybrid_val_norm` is not interpretable yet (bug below); it equals `hybrid_val` in AUC (0.8676) and is lower in AP.

To do:

- [x] (done, `ablation` 20261107_000439: `full` 0.8661 / 0.0426, `no_history` 0.8027 / 0.0211, `history_only` 0.7618 / 0.0087; history is worth about 0.063 AUC again; whether `history_only` should include the bucket dummies is still undecided) Fix `no_history`: also drop `history_bucket_*` (and decide whether `history_only` should include them), then rerun `ablation`; restores the history-contribution ablation needed for RQ D3 and the Discussion
- [x] Decide `account_code`: it is a factorize id of the receiver, not a behavioural feature, and `_encode` now includes it explicitly; rerun `ablation` and `full_gbm` without it, keep it only if it can be justified in Methods
- [x] Re-evaluate `full_gbm`: early stopping on val, rerun without `account_code`, and explain why AUC (0.778) is below LR while AP is 3.4× higher (the earlier probe-signal HGB reached 0.8825 AUC)
- [x] Fix `hybrid_val_norm` in `_error_feature`: it z-scores the raw error per bucket, then clips negatives to 0 and takes the log, so every below-mean row becomes the same constant ($\\log 10^{-8}$); mean/std are also computed from each split's own errors. Take the log first, standardize per `history_bucket` with mean/std from train-normal rows, no clipping, then rerun
- [x] (superseded by the full 16-configuration grid on the fixed baseline: `hybrid_val` vs `lr_val_control` $\Delta$AUC +0.0000 to +0.0017, CI above 0 in 0/3 seeds in every configuration; `stack_val` $\Delta$AUC +0.0022 to +0.0034, CI above 0 in 3/3 seeds, but $\Delta$AP about -0.002) Re-run `eval-hybrid` for $W=1,3$ (bottleneck, 3 seeds, `--bootstrap 200`) on the fixed baseline, to see whether the earlier gain (+0.0026 to +0.0036 AUC) survives
- [x] Make `_encode` column order deterministic (`sorted(include_cols)` instead of `list(set)`) so reports are reproducible
- [ ] (the 2026-11-07 IMRaD draft is written with this framing, confirm) Decide framing: with marginal gains the negative-result framing is the honest default (claim: reconstruction error adds at most about +0.004 AUC over LR with receiver-history features)
- [x] Script written: `evaluation/paysim_synthetic_financial/hybrid_classifier.py` + hooks in `pipeline.py`, `baseline.py`, `metrics.py` (`recall_at_fpr`), `cli.py`; delivered as `p3_5_hybrid.patch`
- [x] Apply the patch locally (`git apply p3_5_hybrid.patch`), then run `uv run python -m lstm_autoencoder eval-hybrid --dataset paysim_synthetic_financial --window 1 --bottleneck --seeds 42,123,2026` (needs the 3 trained `lstm_ae_w1_bottleneck_seed*_best.pt` checkpoints)
- [x] Run for `--window 3,5,10` as well (all four windows done)
- [x] Decision gate on the output (RESULT: not met for `hybrid_val` at any window; `stack_val` passes for AUC at $W=3$ only, while losing AP): `hybrid_val` beats `lr_val_control` with the paired CI of $\\Delta$AUC above 0 in every seed and `null_*` deltas near 0 → hybrid framing; otherwise → negative-result framing
- [x] (done in commit 6befccc: `type_*` dummies restored and `history_bucket_*` added; `account_code` is now kept explicitly, see the decision line below) Verify the baseline feature set before trusting any LR number: `BaselineEvaluator._encode` appears to keep `account_code` (factorize id of the receiver, not a real feature) and to drop the one-hot `type_*` columns (pandas returns bool dummies and `select_dtypes(include="number")` excludes bool; checked on a toy frame, not on the real run). Print `list(encoded.columns)`, fix if confirmed, rerun `ablation`, and treat $0.8625$ / $0.797$ / $0.765$ as provisional until then
- [x] (superseded, all windows were run) Fix hybrid window: build on $W=1$ now (optionally $W=3$ as secondary); keep $W=5,10$ only as the RQ D3 negative result unless the diagnosis above fixes them
- [x] Leak-free protocol implemented: AE error on train-normal rows is in-sample, so the second stage is fitted on val and judged on test; the naive train-fit version is kept as a leak diagnostic (`hybrid_train_naive`)
- [ ] Optional, low priority (`hybrid_train_naive` ≈ `hybrid_val`): K-fold cross-fitting of the AE (needs K retrains per window) if val-fit looks unstable
- [ ] Accept or fix the known caveat: val picked the AE checkpoint and also fits the second stage (mild bias on val only, test untouched); bootstrap resamples rows, not receivers (CIs slightly optimistic)
- [x] `eval-hybrid` builds `hybrid_val` (LR full + log error) and `stack_val` (LR on baseline logit + log error); reports AUC, AP, recall at 1% / 0.1% FPR vs `baseline_train` ($0.8625$ / $0.0415$) and `lr_val_control`, with paired bootstrap CIs and AUC/AP per `history_bucket`
- [x] Null controls (`null_hybrid_val`, `null_stack_val`: shuffled error) run in the same command
- [x] (done by the 16-configuration grid: plain, bottleneck, attention, attention pooling; differences between architectures are about 0.001 AUC or less) Ablation: bottleneck vs attention variant as the error source

### P3.6 Baselines for RQ D1 — DONE (`eval-unsupervised`, report 20261107_065750)

- [x] Isolation Forest (all features, and the 5 sequence features) and a dense AE on the 5 sequence features: `evaluation/paysim_synthetic_financial/unsupervised_baselines.py`, `model/dense_autoencoder.py`, `DenseAutoencoderConfig` in `model/config.py`, `cli.py`; delivered as `p4_baselines.patch`
- [x] (done: `isolation_forest_all` 0.8665 ± 0.0018 / AP 0.0495, `isolation_forest_seq5` 0.8286, `dense_ae_seq5` 0.3590 ± 0.0102) Apply the patch and run `uv run python -m lstm_autoencoder eval-unsupervised --dataset paysim_synthetic_financial --seeds 42,123,2026` (drops `account_code` by default; `--keep-account-code` to keep it)
- [x] (done: plain and bottleneck runs exist for all four windows in the 16-configuration grid) LSTM-AE without attention needs no new code: bottleneck runs already exist for $W=1,3,5,10$; for the per-step variant run `train-model --window N --plain --seeds 42,123,2026`
- [ ] Dense AE latent sweep (`--latent-dim 2,3,4`): the LSTM-AE uses hidden size 64 for 5 input features, so at $W=1$ it has no real compression; check whether a narrower latent changes the AE-only AUC
- [x] Attention-pooling variant implemented: `AttentionPooling` class in `lstm_autoencoder.py` pools encoder outputs to a single bottleneck vector using a learnable query, then repeats across the window for decoder. Added `--attention-pooling` flag to CLI. This addresses contribution 2 from the proposal.
- [x] (done, verified `run_config`: attention pooling test AUC 0.6004 / 0.6082 / 0.5934 / 0.5830 for W=1/3/5/10, no consistent advantage over plain) Train attention-pooling variant for all windows (1,3,5,10) and 3 seeds, compare with other architectures
- [x] (generated by `summarize-results` into `artifacts/summary/results_summary.md`; a single side-by-side table with per-bucket AUC is still open, see P6) Build one comparison table (P4.0): Isolation Forest, dense AE, LSTM-AE plain / bottleneck / attention / attention-pooling, `full` LR, `full_gbm`, hybrid; AUC, AP, recall at fixed FPR, per-bucket AUC

### P3.7 Training Infrastructure & Tooling — DONE

- [x] Optimized `TrainConfig`: batch_size increased from 4096 to 16384, num_workers from 4 to 8 for faster GPU utilization
- [x] Created `scripts/train_all_configs.py`: batch training script for all configurations (windows: 1,3,5,10; architectures: attention, bottleneck, plain, attention_pooling; seeds: 42,123,2026)
  - Resume mechanism: skips existing checkpoints
  - Streaming output with real-time progress
  - Error handling with summary reporting
  - Colored console output
- [x] Created `scripts/validate_checkpoints.py`: validates all checkpoints can load and run forward pass
- [x] Updated `ReportWritter` to include architecture_config parameter
- [x] Updated `PaySimModelTrainingPipeline` to pass architecture details (window, architecture flags) to reporter
- [x] Created `scripts/fix_report_commands.py`: fixes old reports by matching checkpoints via modification times
- [x] Created `scripts/verify_report_fix.py`: verifies report commands match checkpoint configurations
- [x] Report commands now explicitly show architecture flags (e.g., `--window 5 --bottleneck --seeds 42,123,2026`) instead of generic script references
- [x] (commit `b0224a6`) Grid workflows moved into the main CLI: `train-all-configs`, `eval-hybrid-all-configs`, `run-pipeline`; `scripts/` is deprecated (see `scripts/README.md`)

## P4. Evaluation

### P4.0 Core metrics

- [x] (tables in `results_summary.md`; the supervised reference is now `full_gbm` AUC 0.8592 / AP 0.2938 in the reruns) Hybrid vs `full` LR vs `history_only` vs standalone AE: AUC, AUPRC (supervised upper bound $0.8825$)

### P4.1 Mandatory stratified evaluation

- [x] (per-bucket results exist in the baseline and hybrid reports; `summarize-results` does not yet build the side-by-side table) AUC/AUPRC per `history_bucket` (not aggregate-only), hybrid vs P2 baselines side by side
- [x] AUC of AE error within each `history_bucket`

## P5. Writing (IMRaD)

- [x] Decision: drop the real-time claim (RQ D4, contribution 5 streaming evaluation, "Real-Time" in the title); no latency/throughput experiments
- [x] Decision: 3 seeds (42, 123, 2026) instead of the planned 5; state it as a limitation and report mean ± std
- [x] First IMRaD draft (Abstract to References, Indonesian, negative-result framing, results from `f6bfe93`) written 2026-11-07 as `imrad-draft-hybrid-lstm-ae.md`; marked `[TODO]` where P6 items are missing
- [ ] Rewrite the title, RQ list and contributions without "real-time", with the negative-result framing once the P3.5 follow-ups are done
- [ ] Decide framing before drafting: at $W=1$ the LSTM sees one step and attention falls back to self, so the best model is effectively an MLP-style autoencoder over per-transaction features; either reframe the title/claims or find a setting where $W>1$ helps
- [ ] Draft Introduction (money-mule / receiver-centric fraud narrative, PaySim 31-day limitation as scope boundary)
- [ ] Draft Methods (dataset, scope filter, windowing redesign, features, model architecture, hybrid design)
- [ ] Draft Results (stratified metrics, ablation table, baseline comparison, hybrid probe)
- [ ] Draft Discussion (history_length lift interpretation, limitations, future work)
- [ ] Target venue check against SINTA 3 formatting requirements

## P6. Experiment gaps vs the research plan (E1-E7) — OPEN

Conventions (see `PRINCIPLE_DESIGN.md`): one main class per module, `cli.py` only parses arguments and dispatches, long runs write a report to file, new evaluation code goes under `src/lstm_autoencoder/evaluation/paysim_synthetic_financial/`, each new CLI command gets a section in `command/paysim_synthetic_financial.md` (currently sections 1-12), and `summarize-results` is extended when a new report type should appear in the Results tables. When proposing a change, list every file it touches, including existing ones.

Current state (2026-11-07, 3 seeds, test split): standalone LSTM-AE AUC 0.49-0.67 (best W=1, bottleneck collapses to chance at W>=3); `full` LR 0.8661 / AP 0.0426; `full_gbm` 0.8592 / AP 0.2938; Isolation Forest 0.8665; hybrid gain is small and not significant in AUC; architecture does not matter.

### P6.1 Needed to support the paper's claims

- [ ] **E7 attention weights (RQ D2 interpretability).** Nothing returns attention weights today. Return weights from `SelfAttention` and `AttentionPooling`, compare fraud vs normal windows for W>=3, save a report and figure. Files: `model/lstm_autoencoder.py`, `model/trainer.py`, new `evaluation/paysim_synthetic_financial/attention_analysis.py`, `cli.py`, `command/paysim_synthetic_financial.md`. If skipped, narrow RQ D2 to the contribution of attention to performance.
- [ ] **E3 threshold strategies and threshold metrics.** Compare p99 of normal train error vs a threshold chosen on the labelled validation set; report precision, recall, F1, MCC and precision@k (currently AUC, AP and recall at fixed FPR only). Files: `evaluation/metrics.py`, `evaluation/paysim_synthetic_financial/metrics.py`, `evaluation/paysim_synthetic_financial/hybrid_classifier.py`, `evaluation/paysim_synthetic_financial/results_summary.py`, `cli.py` if options are added, docs.
- [ ] **Prevalence sensitivity.** Re-evaluate precision/AP with the test set re-sampled to several fraud prevalences. Decide the targets first: the test set is already 0.297%, so the planned "0.5-1%" would raise prevalence. Files: `evaluation/metrics.py`, `hybrid_classifier.py`, `baseline.py`, docs.
- [x] **Fix `hybrid_val_norm` (bug still present at `b0224a6`).** `_error_feature` z-scores the raw error per bucket, clips negatives to 0, then takes the log, so every below-mean row becomes the same constant, and mean/std come from each split's own errors. Take the log first, standardize per `history_bucket` with train-normal statistics, no clipping; rerun `eval-hybrid-all-configs`. Until then do not interpret `hybrid_val_norm`. File: `evaluation/paysim_synthetic_financial/hybrid_classifier.py`.
- [x] **Verify formulas against code** for the IMRaD Methods: the `attn_pool` attention score and `reconstruction_error` (masking over non-padded steps) in `model/lstm_autoencoder.py`.
- [x] **Make `_encode` deterministic.** `BaselineEvaluator._encode` returns `list(include_cols)` over a `set`, so the column order can change between runs. Use `sorted(include_cols)`. File: `evaluation/paysim_synthetic_financial/baseline.py`.

### P6.2 Strengthens the conclusions

- [ ] **Hybrid with a non-linear learner.** Test whether the AE error still adds anything on top of `full_gbm` (AP 0.2938 vs hybrid LR about 0.043), same validation-fit protocol, null control and bootstrap. Files: `hybrid_classifier.py`, `baseline.py` (reuse the GBM fit), `cli.py`, `results_summary.py`, docs.
- [ ] **E5 feature ablation.** Variants without time-difference features (`step_diff`, `log_amount_ratio`), without relational features, and with vs without the leakage-risk balance columns (only `no_history` exists). Also decide `account_code` (still included explicitly in `_encode`). Files: `baseline.py`, `data/paysim_synthetic_financial/causal_features.py` if columns need grouping, `cli.py`, `results_summary.py`, docs.
- [ ] **E6 segment analysis.** Per `transaction_type` (CASH_OUT vs TRANSFER) and per period (bins of `step`); device is not in PaySim. Files: `baseline.py`, `hybrid_classifier.py`, `results_summary.py`, docs.
- [ ] **E2 remaining attention variants.** Mean-pooling is the natural control for attention pooling; multi-head self-attention and cross-attention decoder are in the plan. Decide which to keep. A new architecture touches `model/config.py` (`ModelConfig`, `architecture_tag`), `model/lstm_autoencoder.py`, `model/trainer.py`, `model/pipeline.py`, `cli.py` (grid commands), the hybrid evaluators, and docs.
- [x] **Re-evaluate `full_gbm`.** AUC 0.8592 is below `full` LR (0.8661) while AP is about 6.9x higher; early stopping is still off and `account_code` is still a feature. Confirm the ranking effect is real. File: `baseline.py`.
- [x] **Side-by-side per-bucket table** (baselines, `full_gbm`, hybrid) generated by `summarize-results`, plus AUC of the AE error within each `history_bucket`. Files: `results_summary.py`, docs.

### P6.3 Decide, then implement or document as a limitation

- [x] Baselines in the plan but not implemented: PCA reconstruction error, LightGBM/XGBoost (HistGradientBoosting is used), supervised LSTM + attention classifier.
- [x] Seeds: 3 (42, 123, 2026) instead of the planned 5.
- [x] E4 latency/throughput stays out of scope (real-time claim removed); keep the paper wording consistent.
- [x] Report config documenting: training and hybrid reports carry `run_config`, the `ablation` and `eval-unsupervised` reports do not.
- [x] Minor Bug: Windows exit code `3221226505` at shutdown of `train-model` after the report is saved (suspect DataLoader workers: `persistent_workers`, `shared_memory_()`); only reproduced when need to retrain.
