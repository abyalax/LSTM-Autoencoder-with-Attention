# Research Roadmap

> Last updated: 2026-11-10  
> Active focus: pipeline integrity, reconstruction-score diagnostics, and final evaluation gaps.  
> Convention: `W` is the sequence window length; current grid is `W ∈ {1,3,5,10}`. Unless stated otherwise, AUC/AP refer to the held-out test split.

## 2026-11-10 session (checklist validation pass, see `0003-*.patch`)

Validated the "pre-submit checklist" against the ACTUAL code (not this file's
checkboxes, which had drifted from what was implemented). Two findings:

- **P1.0/entity definition had a real bug, not just a missing audit.** The
  proposal's own decided entity rule ("nameDest for TRANSFER, nameOrig for
  CASH_OUT, so a mule's TRANSFER-in and CASH_OUT-out link into one account
  history") was never implemented -- `PaySimSchema.receiver_column` was used
  uniformly as `nameDest` for every row, including CASH_OUT. Fixed in
  `causal_features.py`: account identity is now role-aware; original value
  kept as `nameDest_raw`. **This changes the dataset CASH_OUT rows are built
  from -- every checkpoint and report trained before this patch is stale and
  must be regenerated.**
- **P2.0 (compare mean/last/max reconstruction score) was already implemented**
  (`ScoreDiagnostics`, `score-diagnostics` CLI command) despite the checkbox
  being unchecked -- ticked below. Still only run at seed=42; P2.1's "across
  seeds 42/123/2026" stays open.

Also fixed this session (code-only, no retraining needed to apply, but
results built on the OLD split/identity must be redone):
- P1.1 tie handling: same-step ties now break on an explicit, tested,
  documented row-order rank (previously an untested side effect of a stable
  sort).
- P1.3 padding: added exhaustive left-padding/mask tests for every valid
  length 1..W (existing vectorized logic was already correct, just untested
  at that coverage).
- P1.4 label-informed stratification: removed. `_split_receiver_ids`
  (`pipeline.py` and `baseline.py`, previously two copies of the same logic)
  stratified train/val/test by each receiver's own `fraud_rate` -- replaced
  with stratification by transaction count only, in both places. Verified
  with a test that relabels which receivers are fraud while holding
  transaction counts fixed: split is unchanged.

**Everything downstream (P2 retraining grid, P3-P5 result tables, IMRAD
numbers) is built on the pre-fix split/identity and is not valid until
re-run** -- see the command list at the end of this file.

## 2026-11-09 session (pre-submit P0/P1 pass, see `0001-*.patch`)

Checked off above are the items this session actually verified with new
automated tests (`tests/test_methodology_audit.py`, 14 tests, run locally
against synthetic fixtures -- not yet re-run on the full PaySim CSV on the
training machine). Also done this session, with no single pre-existing
checkbox to tick, so noted here instead:

- [x] Standalone LSTM-AE reports now include Recall@FPR 1%/0.1% for every
  config (`Trainer.score` returns a dict, not a bare `(auc, ap)` tuple).
- [x] `run_config` (both training and hybrid-classifier reports) now records
  optimizer (Adam), decoder type, and trainable parameter count.
- [x] Per-bucket table now reports `fraud_rate`; `baseline_train`/`hybrid_val`
  per-bucket AUC/AP now carry a bootstrap CI (other variants intentionally
  left without one -- see P5.1 cost note below).
- [x] Extended the hybrid classifier's bootstrap CI (incl. `p_d_ap_le_0`) to
  the null-control comparisons, not only the primary ones.
- [x] Receiver-cluster bootstrap implemented as the new default
  (`bootstrap_mode="receiver"`), `--bootstrap-mode` exposed on the CLI
  (`eval-hybrid`, `eval-hybrid-all-configs`, `run-pipeline`); row-wise kept
  as `bootstrap_mode="row"` for comparison. **Not yet actually compared
  side-by-side on real data** -- P5.1's "Compare row-wise bootstrap with
  receiver-cluster bootstrap" stays unchecked until that run happens.
- [x] Fixed a dead/misleading `val_idx` parameter on the GBM baseline fit;
  confirmed GBM AP/AUC use the identical scoring path as LR (P4.0-adjacent,
  no bug found, just cleanup -- not a checkbox item above).

Still open from this list: exhaustive left-padding-mask coverage for every
length 1..W (P1.3 -- only one length was covered this session); the
row-vs-receiver bootstrap comparison itself (P5.1); GBM+AE-error hybrid
(P4.2); everything in P2-P4 not mentioned above; and the full P6 manuscript
pass (IMRAD narrative + editorial still queued separately).

## P1 Data Pipeline dan Sequence Integrity

### P1.0 Validate receiver-centric sequence semantics

- [ ] - Audit whether `nameDest` is the correct sequence entity for the research question [AC - Research question and entity definition are explicit in Methods] -- `0003-*.patch` fixed a real bug here (code used `nameDest` for CASH_OUT rows too, contradicting the documented role-aware decision); the broader "is this entity choice right at all" comparison below is still open
- [ ] - Compare receiver-only, sender-only, and account-centric sequence construction [AC - All variants have comparable counts, labels, and metrics]
- [ ] - For account-centric sequences, represent incoming and outgoing transactions with an explicit transaction role [AC - One account history contains both directions without duplicating target labels]
- [ ] - Decide whether merchant destinations and account destinations require different handling [AC - Inclusion/exclusion rules are documented and tested]

### P1.1 Validate transaction ordering and causal history

- [x] - Audit sorting by account identifier and PaySim `step` [AC - Every account sequence is monotonically ordered by the documented key]
- [x] - Investigate transactions sharing the same `step`; define deterministic tie handling without claiming unavailable within-hour chronology [AC - Tie policy is deterministic and documented] -- `0003-*.patch`; code-only, applies retroactively once re-run
- [x] - Verify that `history_length`, `step_diff`, and `log_amount_ratio` use only information available at the current transaction [AC - No feature for row t reads a later row or post-transaction value]
- [ ] - Verify that the sequence label is the `isFraud` value of the final/current transaction [AC - Hand-constructed tests confirm exact label alignment]

### P1.2 Validate sequence eligibility and row alignment

- [ ] - Test `min_history_length` behavior for receivers with 1, 2, and fewer than W transactions [AC - Emitted row counts match the documented rule]
- [ ] - Trace the reported 497-row difference between the earlier split summary and model test rows [AC - Every dropped row has a documented reason and LR/AE rows are aligned]
- [ ] - Assert that sequence labels, receiver IDs, history buckets, and row indices remain aligned after filtering [AC - Shape and identity assertions pass on a deterministic fixture]

### P1.3 Audit padding and recurrent-state handling

- [x] - Add tests for left-padding masks at each valid sequence length from 1 through W [AC - Mask contains exactly the real positions] -- `0003-*.patch`; existing vectorized logic found already correct
- [ ] - Measure whether zero-valued left-padding changes LSTM hidden/cell states for otherwise identical valid sequences [AC - A regression test detects padding-induced state changes]
- [ ] - Compare current left-padding with right-padding plus `pack_padded_sequence`, or an equivalent recurrent mask [AC - Both approaches are evaluated on identical data]
- [ ] - Verify that attention and reconstruction loss ignore padding while the encoder state does not encode padding length unintentionally [AC - Padding-invariance tests pass for the selected implementation]
- [ ] - Retrain affected checkpoints if sequence representation changes [AC - Checkpoints are not evaluated with incompatible sequence semantics]

### P1.4 Audit split, scaling, and training eligibility

- [x] - Assert that receiver IDs are disjoint across train, validation, and test [AC - Automated overlap checks return zero]
- [x] - Document label-informed receiver fraud-rate stratification and compare with a split strategy that does not use receiver fraud rates [AC - Split protocol and sensitivity results are reported] -- `0003-*.patch` REMOVES it (stratify by transaction count instead); no longer just documented, the strategy is retired. Pre/post sensitivity numbers still need a real-data re-run.
- [x] - Verify that the scaler is fitted only on normal training rows and reused unchanged for validation/test [AC - Tests confirm fit-row membership and stable scaler parameters]
- [ ] - Verify that the autoencoder optimizer receives only normal training windows [AC - Training dataset labels contain no fraud rows]
- [ ] - Confirm that checkpoint selection uses normal-only validation reconstruction loss [AC - Selection metric and implementation agree]

## P2 Reconstruction Error dan Anomaly Scoring

### P2.0 Compare reconstruction-score definitions

- [ ] - Implement last-step reconstruction error [AC - Error for the current/final transaction is available per sequence]
- [ ] - Preserve mean reconstruction error over valid timesteps [AC - Existing masked mean score remains reproducible]
- [ ] - Implement max reconstruction error over valid timesteps [AC - Maximum is computed over valid timesteps only]
- [x] - Compare last-step, mean, and max scores on the same checkpoint, rows, and labels [AC - AUC, AP, and fixed-FPR recall are reported side by side] -- already implemented in `ScoreDiagnostics`/`score-diagnostics` CLI, checkbox had drifted from the code; see `artifacts/report/.../score_diagnostics/`

### P2.1 Diagnose score behavior by history length

- [ ] - Report count, median, p90, and p99 reconstruction error for normal, fraud, fraud missed by LR, and fraud detected by LR [AC - Every group includes sample count and robust distribution statistics]
- [ ] - Correlate reconstruction error with valid timestep count/history length [AC - Correlation and plots are reported per window]
- [ ] - Compute AE-only AUC/AP within each `history_bucket` [AC - Within-bucket and aggregate metrics use identical scoring rows]
- [ ] - Re-run hybrid signal probe for W=1 and W=3 across seeds 42, 123, and 2026 [AC - Mean, standard deviation, and per-seed results are available]

### P2.2 Add score sanity checks

- [ ] - Verify anomaly-score direction and positive-class convention for every metric [AC - Synthetic high-error fraud ranks above low-error normal]
- [ ] - Add tests for invalid all-padding inputs if such samples can reach the model [AC - Invalid samples are rejected or handled by a documented policy]
- [ ] - Verify score-label alignment after DataLoader batching and concatenation [AC - Sample identifiers match original dataset order]

## P3 Model Architecture dan Training

### P3.0 Correct validation metric reporting

- [ ] - Track validation AUC at the checkpoint selected by minimum normal-only validation loss [AC - `best_val_auc` is stored separately from `best_val_loss`]
- [ ] - Rename aggregate fields so validation loss is never reported as validation AUC [AC - Reports expose distinct `val_loss_mean/std` and `val_auc_mean/std`]
- [ ] - Update console output, JSON reports, and run summaries [AC - No output labels reconstruction loss as AUC]

### P3.1 Validate attention implementations

- [ ] - Expose attention weights from `SelfAttention` and `AttentionPooling` through an opt-in analysis path [AC - Weights can be exported without changing normal inference outputs]
- [ ] - Compare attention distributions for normal and fraud sequences with W ≥ 3 [AC - Reproducible report and figure are generated]
- [ ] - Compare attention pooling with a non-attention mean-pooling control [AC - Same split, seeds, and training budget are used]
- [ ] - Decide whether multi-head self-attention or cross-attention decoder is justified before implementation [AC - Each addition has a stated hypothesis and baseline comparison]

### P3.2 Tune only after pipeline/scoring fixes

- [ ] - Run hidden-dimension sweep (32, 64, 128) [AC - Same seeds, split, and score definition]
- [ ] - Run LSTM-layer sweep (1, 2, 3) [AC - Same seeds, split, and score definition]
- [ ] - Run dropout sweep (0.1, 0.2, 0.3) [AC - Same seeds, split, and score definition]
- [ ] - Run learning-rate sweep (0.0001, 0.0005, 0.001) [AC - Same seeds, split, and score definition]
- [ ] - Compare batch sizes (512, 2048, 4096) with current 16384 baseline [AC - Runtime, convergence, and held-out metrics are reported]
- [ ] - Keep window-length comparisons separate from architecture and feature sweeps [AC - Each experiment changes one factor at a time]

## P4 Feature Engineering dan Baseline Experiments

### P4.0 Audit baseline feature validity

- [ ] - Revisit `account_code` as a receiver identifier rather than a behavioral feature [AC - Main results with and without the identifier are reported]
- [ ] - Re-run `no_history` and `history_only` ablations with explicit definitions for raw history features and `history_bucket_*` indicators [AC - Each ablation adds/removes only the intended feature family]
- [ ] - Verify categorical `type_*` columns are retained and encoded consistently [AC - Encoded feature names and dtypes are asserted]

### P4.1 Run targeted feature ablations

- [ ] - Compare existing five sequence features with and without `is_new_receiver` [AC - Redundancy impact is quantified]
- [ ] - Ablate `step_diff` and `log_amount_ratio` separately and together [AC - Temporal-feature contribution is quantified]
- [ ] - Compare transaction type and transaction role features where supported by the sequence representation [AC - Feature availability and encoding are consistent across splits]
- [ ] - Evaluate balance-derived features only after transaction-time availability/leakage audit [AC - Every retained feature has a documented causal definition]

### P4.2 Test hybrid with a non-linear learner

- [ ] - Evaluate whether reconstruction error adds value to `full_gbm` using the same validation-fit protocol as LR hybrid [AC - GBM-only, GBM+error, and null control use identical test rows]
- [ ] - Add paired bootstrap confidence intervals for AUC, AP, and recall at fixed FPR [AC - Uncertainty is reported for each comparison]

### P4.3 Add segment and prevalence analysis

- [ ] - Report metrics separately for `CASH_OUT` and `TRANSFER` [AC - Each segment reports sample count, fraud prevalence, AUC, and AP]
- [ ] - Report metrics across documented `step` periods [AC - Period boundaries are fixed before test inspection]
- [ ] - Run prevalence-sensitivity analysis with clearly defined resampling targets [AC - Original and simulated prevalence are reported]

### P4.4 Complete unsupervised baseline checks

- [ ] - Run Dense AE latent-dimension sweep (2, 3, 4) [AC - Each dimension uses the same split and seeds]
- [ ] - Compare PCA reconstruction error only if it answers a defined baseline question [AC - Inclusion/exclusion decision is documented]

## P5 Evaluation Reliability dan Reporting

### P5.0 Add threshold-based metrics

- [ ] - Compare p99 normal-training reconstruction-error threshold with a threshold selected on labelled validation data [AC - Threshold selection never uses test labels]
- [ ] - Report precision, recall, F1, MCC, and precision@k in addition to AUC/AP [AC - Metrics are reported for each threshold strategy]
- [ ] - Report recall at fixed FPR (0.1% and 1%) with confidence intervals [AC - Every recall estimate has an uncertainty interval]

### P5.1 Improve uncertainty estimation

- [x] - Increase paired bootstrap replicates from 200 to at least 2000 for final results [AC - Final tables use the larger bootstrap run]
- [ ] - Compare row-wise bootstrap with receiver-cluster bootstrap [AC - Dependence-aware uncertainty is discussed and justified]
- [ ] - Retain per-seed metrics alongside aggregate mean and standard deviation [AC - No aggregate hides a materially divergent seed]

### P5.2 Resolve validation protocol caveats

- [ ] - Document that validation data is used for checkpoint selection and second-stage hybrid fitting [AC - Potential validation-selection bias is stated]
- [ ] - Consider AE cross-fitting only if validation-fit instability materially affects hybrid estimates [AC - Decision is supported by a stability comparison]

## P6 Research Framing dan IMRaD

### P6.0 Finalize research questions and contribution claims

- [ ] - Decide whether the main contribution is a negative-result evaluation of LSTM-AE reconstruction error or a hybrid detector with measurable benefit [AC - Claims follow final multi-seed and confidence-interval results]
- [ ] - Reframe W=1 accurately: a one-step sequence does not demonstrate temporal-context learning [AC - Title, RQs, and contributions do not overclaim temporal modeling]
- [ ] - Keep real-time and latency/throughput claims out of scope unless those experiments are added [AC - No unsupported real-time claim remains]

### P6.1 Complete the manuscript

- [ ] - Rewrite title, RQs, and contributions to match final experimental findings [AC - Abstract, Introduction, and Conclusion use consistent claims]
- [ ] - Draft Introduction with receiver-centric fraud motivation and PaySim scope limitations [AC - Problem, gap, and contribution are supported by literature]
- [ ] - Finalize Methods: dataset, filtering, sequence construction, features, split, scaler, model, training, and metrics [AC - Experiments are reproducible from documented configuration]
- [ ] - Finalize Results with baseline, ablation, architecture, score, and segment comparisons [AC - Tables use validated outputs]
- [ ] - Finalize Discussion with history-length effects, failure cases, threats to validity, and future work [AC - Interpretations are separated from measured findings]
- [ ] - Check target SINTA 3 venue scope, template, citation style, and submission requirements [AC - Target venue and compliance checklist are recorded]

## Completed Milestones

- [x] Restricted PaySim scope to `CASH_OUT` and `TRANSFER`, and removed balance columns designated as leakage-risk inputs.
- [x] Implemented receiver-keyed causal features, history features, history buckets, and receiver-stratified split.
- [x] Implemented LSTM-AE variants: plain, self-attention, bottleneck, and attention pooling.
- [x] Added train-normal-only scaling/training, deterministic seeds, and run configuration metadata.
- [x] Implemented LR, HistGradientBoosting, Isolation Forest, Dense AE, and hybrid evaluation workflows.
- [x] Added paired bootstrap comparisons and null controls for LR hybrid workflow.
- [x] Added side-by-side result summaries and per-history-bucket evaluation.
- [x] Removed unsupported real-time/latency claims from the planned scope.
- [x] Produced an initial Indonesian IMRaD draft; it remains subject to open P6 tasks.

## Current Experimental Reference

The latest reported baseline/results are maintained in `artifacts/summary/results_summary.md`. The 2026-11-07 roadmap snapshot reports standalone LSTM-AE below the strongest baselines, `full` LR around AUC 0.8661 / AP 0.0426, `full_gbm` around AUC 0.8592 / AP 0.2938, and only marginal hybrid AUC gains without consistent evidence of a clear AP improvement. Refresh these references after any P1-P4 pipeline/model change; historical numbers are not results for a modified pipeline.
