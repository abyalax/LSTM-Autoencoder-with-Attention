### 1. Download the Dataset

Download [Synthetic Financial Datasets For Fraud Detection](https://www.kaggle.com/datasets/ealaxi/paysim1/data) from Kaggle and place it under `data/raw/` with a domain-specific folder:

```
data/raw/paysim_synthetic_financial_datasets/paysim_synthetic_financial.csv
data/raw/paysim_synthetic_financial_datasets/paysim_synthetic_financial.zip
```

The loader reads the CSV first and falls back to the zip when no CSV exists. Each dataset should be organized in its own folder under `data/raw/` to support multiple datasets from different domains.

Dataset paths, column names, thresholds, and split fractions are defined in `src/lstm_autoencoder/config.py`.

### 2. Inspect the Dataset

Run the full set of dataset checks:

```Shell
uv run python -m lstm_autoencoder inspect-data --dataset paysim_synthetic_financial
```

The explorer (PaySimSyntheticFinancialDatasetsExplorer) performs the following operations:

- **Structure inspection**: column count, data types, missing values, duplicate rows, and unique value counts
- **Target inspection**: fraud prevalence overall and by transaction type (CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER)
- **isFlaggedFraud inspection**: checks overlap between isFlaggedFraud and isFraud columns (potential artifact)
- **Account viability inspection**: evaluates sender account history lengths and calculates percentage of accounts supporting each window length (5, 10, 20 transactions)
- **Chronological split simulation**: 70/15/15 split ordered by step column, reports fraud rate per split
- **Univariate signal probing**: computes Mann-Whitney U based AUC for each numeric feature, both with and without leakage columns (oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest)

The loader automatically applies explicit string dtypes for `nameOrig`, `nameDest`, and `type` columns to prevent incorrect type inference on large datasets (6.3M rows) during groupby operations.

The command saves the following artifacts under `artifacts/`:

- `dataset_report.txt`
- `dataset_report.json`

Optional arguments:

- `--project-root` : Project root that contains `data/raw/`. Defaults to the current working directory.
- `--dataset` : Dataset folder name in `data/raw/` (e.g., `paysim_synthetic_financial`). Required.
- `--nrows` : Read only the first N rows of the file. Defaults to all rows. Use it for a quick trial run only, because the file is not time-ordered and a partial read distorts the per-account sequence statistics.

### 3. Probe the Label Signal

Train a strong supervised model on a chronological split and compare it with a shuffled-label control. If the best supervised model cannot beat chance, no model can detect fraud in this dataset.

```Shell
uv run python -m lstm_autoencoder probe-signal --dataset paysim_synthetic_financial --max-train-rows 1000000
```

The probe saves the following artifact under `artifacts/`:

- `signal_report.txt`

The report ends with one of three verdicts:

- No signal: test AUC below 0.51 and average precision below 1.15 times the prevalence
- Weak signal: test AUC below 0.60
- Learnable: test AUC of 0.60 or higher

Optional arguments:

- `--project-root` : Project root that contains `data/raw/`. Defaults to the current working directory.
- `--dataset` : Dataset folder name in `data/raw/` (e.g., `paysim_synthetic_financial`). Required.
- `--nrows` : Read only the first N rows of the file. Defaults to all rows. Same caveat as `inspect-data`.
- `--max-train-rows` : Maximum number of training rows sampled for the probe model. Defaults to 1,500,000. Lower it to speed up the run. Validation and test rows are never subsampled.

### 4. Run the Ablation Baselines

Train three logistic-regression variants on the receiver-stratified split — the full feature set, the full set minus the causal history features, and the history features alone — and compare them overall and per receiver history-length bucket (`1`, `2-4`, `5-9`, `>=10`).

```Shell
uv run python -m lstm_autoencoder ablation --dataset paysim_synthetic_financial
```

The evaluator (BaselineEvaluator) performs the following operations:

- **Feature encoding**: one-hot encodes `type`, drops leakage columns (`oldbalanceOrg`, `newbalanceOrig`, `oldbalanceDest`, `newbalanceDest`) and `isFlaggedFraud`
- **Receiver-stratified split**: 70/15/15 split by `nameDest`, stratified by per-receiver fraud rate, same scheme used by `probe-signal`
- **Variant training**: fits a standardized `LogisticRegression` per variant (`full`, `no_history`, `history_only`) on train
- **Stratified scoring**: reports AUC/AP overall (val, test) and per history bucket on test, for every variant

The command saves the following artifacts under `artifacts/`:

- `ablation_report.txt`
- `ablation_report.json`

Optional arguments:

- `--project-root` : Project root that contains `data/raw/`. Defaults to the current working directory.
- `--dataset` : Dataset folder name in `data/raw/` (e.g., `paysim_synthetic_financial`). Required.
- `--nrows` : Read only the first N rows of the file. Defaults to all rows. Same caveat as `inspect-data`.

### 5. Train the LSTM-Autoencoder

Build fixed-length per-receiver transaction windows, train the LSTM-Autoencoder on normal-only windows, and evaluate reconstruction error as an anomaly score on the held-out test set.

```Shell
uv run python -m lstm_autoencoder train-model --dataset paysim_synthetic_financial --window 5
```

Train with attention pooling (learned bottleneck):

```Shell
uv run python -m lstm_autoencoder train-model --dataset paysim_synthetic_financial --window 5 --attention-pooling
```

The pipeline (PaySimModelTrainingPipeline) performs the following operations:

- **Sequence building**: left-pads each receiver's last `--window` transactions into fixed-length windows (`amount`, `step_diff`, `log_amount_ratio`, `history_length`, `is_new_receiver`)
- **Receiver-stratified split**: 70/15/15 split by `nameDest`, stratified by per-receiver fraud rate; the training set is further restricted to normal (non-fraud) windows only
- **Training**: fits the LSTM-Autoencoder via Adam, with early stopping on validation reconstruction-error AUC
- **Evaluation**: scores the held-out test set (fraud included) by reconstruction-error AUC and average precision

The model architecture supports four variants:
- **Default (attention)**: Decoder receives self-attended context over encoder outputs (attends to other real steps, never to itself unless alone). Use this when neither `--bottleneck`, `--plain`, nor `--attention-pooling` is specified.
- **Bottleneck (`--bottleneck`)**: Decoder receives only the encoder's final hidden state, repeated across the window — a true information bottleneck forcing the model to compress sequence information. Mutually exclusive with `--plain` and `--attention-pooling`.
- **Plain (`--plain`)**: Decoder receives raw per-step encoder outputs (can copy directly). Mutually exclusive with `--bottleneck` and `--attention-pooling`.
- **Attention pooling (`--attention-pooling`)**: Decoder receives a single bottleneck vector created by attention pooling over encoder outputs, then repeated across the window. This uses a learnable query vector to attend to all time steps and compress them into a single representation, providing a learned bottleneck. Mutually exclusive with `--bottleneck` and `--plain`.

The command saves the following artifacts under `artifacts/`:

- `training_report.txt` or `multi_seed_training_report.txt` (when `--seeds` is used)
- `training_report.json` or `multi_seed_training_report.json` (when `--seeds` is used)
- `lstm_ae_w{window}_{attn|bottleneck|plain|attn_pool}_seed{seed}_best.pt` (best checkpoint, under `artifacts/model/paysim_synthetic_financial/`)

Optional arguments:

- `--project-root` : Project root that contains `data/raw/`. Defaults to the current working directory.
- `--dataset` : Dataset folder name in `data/raw/` (e.g., `paysim_synthetic_financial`). Required.
- `--nrows` : Read only the first N rows of the file. Defaults to all rows. Same caveat as `inspect-data`.
- `--window` : Window length for per-receiver sequences. Defaults to 5. Matches the grid explored in `inspect-data` (1, 3, 5, 10).
- `--seeds` : Comma-separated random seeds for repeated experiments (e.g., `42,123,2026`). When set, trains once per seed on the same data split and reports mean/std across runs to separate genuine improvement from training noise. Default is single run with seed 42.
- `--bottleneck` : Use a single-vector bottleneck decoder (final encoder hidden state, repeated across the window). Mutually exclusive with `--plain` and `--attention-pooling`. Default is False.
- `--plain` : Use raw per-step encoder outputs (can copy directly). Mutually exclusive with `--bottleneck` and `--attention-pooling`. Default is False.
- `--attention-pooling` : Use attention to pool encoder outputs to a single bottleneck vector, then repeat across the window. Mutually exclusive with `--bottleneck` and `--plain`. Default is False.

### 6. Probe hybrid signal

Correlate a trained LSTM-Autoencoder checkpoint's reconstruction error with the logistic regression baseline's score on the same test rows, before committing to a full hybrid retrain. Low correlation suggests the reconstruction error could add complementary signal to the LR; high correlation suggests it is mostly redundant with what the LR already captures from the same underlying features.

```Shell
uv run python -m lstm_autoencoder probe-hybrid-signal --dataset paysim_synthetic_financial --window 5 --seed 42
```

The probe (HybridSignalProbe) performs the following operations:

- **Autoencoder scoring**: Loads the trained checkpoint (no retraining) and scores the test windows by reconstruction error
- **Baseline scoring**: Fits the LR 'full' baseline fresh and scores the same test rows
- **Alignment check**: Ensures both pipelines score the SAME test rows (same dataset, nrows, and receiver split seed)
- **Correlation analysis**: Computes Pearson and Spearman correlation between the two signals
- **Error stratification**: Reports mean AE reconstruction error split by label and, for fraud rows, by whether LR scored it low (LR's likely blind spot) or high (LR already confident)

The probe saves the following artifacts under `artifacts/`:

- `hybrid_signal_probe_report.txt`
- `hybrid_signal_probe_report.json`

Optional arguments:

- `--project-root` : Project root that contains `data/raw/`. Defaults to the current working directory.
- `--dataset` : Dataset folder name in `data/raw/` (e.g., `paysim_synthetic_financial`). Required.
- `--nrows` : Read only the first N rows of the file. Defaults to all rows. Same caveat as `inspect-data`.
- `--window` : Window length for per-receiver sequences. Defaults to 5. Matches the grid explored in `inspect-data` (1, 3, 5, 10).
- `--seed` : Seed used when the checkpoint was trained (for its filename). Default is 42.
- `--bottleneck` : Match the checkpoint's architecture: bottleneck decoder. Mutually exclusive with `--plain` and `--attention-pooling`. Default is False.
- `--plain` : Match the checkpoint's architecture: raw per-step outputs. Mutually exclusive with `--bottleneck` and `--attention-pooling`. Default is False.
- `--attention-pooling` : Match the checkpoint's architecture: attention pooling. Mutually exclusive with `--bottleneck` and `--plain`. Default is False.
- `--checkpoint` : Path to the trained checkpoint. Defaults to the standard train-model naming convention (`lstm_ae_w{window}_{attn|bottleneck|plain|attn_pool}_seed{seed}_best.pt`).

### 7. Hybrid Classifier

Evaluate whether LSTM-Autoencoder reconstruction error improves the logistic regression baseline when used as an additional feature, following a leak-free protocol.

```Shell
uv run python -m lstm_autoencoder eval-hybrid --dataset paysim_synthetic_financial --window 1 --seeds 42,123,2026
```

The evaluator (HybridClassifierEvaluator) tests whether the AE's reconstruction error adds complementary signal to the LR baseline. The key challenge is avoiding leakage: the AE trained on normal train rows, so their error is in-sample (optimistically low) while val/test error is out-of-sample. A second stage fitted on the same train rows would learn from a feature distribution that test never shows.

**Leak-free protocol:**
- The AE never saw validation rows, so the second stage is fitted on validation and judged on the untouched test set
- All variants are scored on the same test rows for fair comparison

**Variants evaluated (all scored on test):**
- `baseline_train`: LR full, fit on train (the production baseline reference)
- `lr_val_control`: LR full, fit on val (same fit data as the hybrids)
- `hybrid_val`: LR full + log(error), fit on val [primary]
- `stack_val`: LR on [baseline logit, log(error)], fit on val [primary]
- `hybrid_train_naive`: LR full + log(error), fit on train (LEAKY, diagnostic)
- `null_hybrid_val` / `null_stack_val`: same as hybrid/stack with error column shuffled (noise floor)
- `ae_error_only`: reconstruction error alone

**Key comparisons:**
- `hybrid_val` vs `lr_val_control`: isolates the error feature (same fit data, only error differs)
- `hybrid_val` vs `baseline_train`: vs. the production baseline
- `stack_val` vs `baseline_train`: 2-parameter stack on top of baseline score

**Bootstrap analysis:**
- Paired, class-stratified bootstrap of delta AUC / delta AP for primary comparisons
- Fraud and normal rows resampled separately to maintain test prevalence
- All variants scored on the same draw (paired)
- Reports 95% confidence intervals and proportion of seeds where CI stays above 0

**Interpretation:**
The error feature helps only if:
1. `hybrid_val` beats `lr_val_control` (same fit data, only error differs)
2. The paired CI of its delta AUC stays above 0 (`seeds_auc_ci_above_0 == n_seeds`)
3. The `null_*` deltas are near 0 (noise floor)
4. `hybrid_train_naive` minus `baseline_train` shows how much the in-sample (leaky) protocol would have inflated the gain

**Per-bucket analysis:**
Reports test AUC/AP per receiver history bucket (1, 2-4, 5-9, >=10) for:
- `baseline_train`, `lr_val_control`, `hybrid_val`, `stack_val`, and `ae_error_only`

**Configuration (HybridClassifierConfig):**
- `seeds`: Random seeds for repeated experiments (default: (42,))
- `n_bootstrap`: Number of bootstrap resamples for CI calculation (default: 200)
- `error_epsilon`: Small constant added before log-transform to guard against exact zeros (default: 1e-8)
- `random_seed`: Base random seed for reproducibility (default: 42)
- `fpr_targets`: FPR thresholds for recall-at-FPR metrics (default: (0.01, 0.001))

**Artifacts saved:**
- `hybrid_classifier_report.txt`
- `hybrid_classifier_report.json`

**Caveats:**
- The validation set also picked the AE checkpoint (early stopping on val AUC), which mildly favors the error feature on val
- Test set is untouched, so reported test numbers remain valid
- Bootstrap resamples rows, not receivers, so its CIs are somewhat optimistic

**Optional arguments:**
- `--project-root`: Project root containing data/raw/. Defaults to current working directory.
- `--dataset`: Dataset folder name in data/raw/ (e.g., paysim_synthetic_financial). Required.
- `--nrows`: Read only the first N rows. Defaults to all rows.
- `--window`: Window length for per-receiver sequences. Defaults to 5.
- `--bottleneck`: Match the checkpoint's architecture: bottleneck decoder. Mutually exclusive with `--plain` and `--attention-pooling`. Default is False.
- `--plain`: Match the checkpoint's architecture: raw per-step outputs. Mutually exclusive with `--bottleneck` and `--attention-pooling`. Default is False.
- `--attention-pooling`: Match the checkpoint's architecture: attention pooling. Mutually exclusive with `--bottleneck` and `--plain`. Default is False.
- `--seeds`: Comma-separated seeds for multi-seed evaluation (e.g., 42,123,2026). Checkpoints must have been trained with train-model using these seeds. Default is 42.
- `--n-bootstrap`: Number of bootstrap resamples for confidence interval calculation. Default is 200.

### 8. Evaluate Unsupervised Baselines

Train and evaluate unsupervised anomaly detection models (Isolation Forest and Dense Autoencoder) on the same data split used by the LSTM-Autoencoder, providing a fair comparison against supervised baselines.

```Shell
uv run python -m lstm_autoencoder eval-unsupervised --dataset paysim_synthetic_financial --seeds 42,123,2026
```

The evaluator (UnsupervisedBaselineEvaluator) performs the following operations:

- **Receiver-stratified split**: 70/15/15 split by `nameDest`, stratified by per-receiver fraud rate; the training set is restricted to normal (non-fraud) rows only
- **Model training**: Fits each unsupervised model on normal training rows
- **Model evaluation**: Scores the held-out test set (fraud included) by anomaly scores (AUC and average precision)

**Models evaluated:**
- `isolation_forest_all`: Isolation Forest trained on all features (amount, step_diff, log_amount_ratio, history_length, is_new_receiver, type encoded)
- `isolation_forest_seq5`: Isolation Forest trained on the same 5 features used by LSTM-AE (amount, step_diff, log_amount_ratio, history_length, is_new_receiver)
- `dense_ae_seq5`: Dense Autoencoder trained on the same 5 features as LSTM-AE

**Feature handling:**
- `account_code`: Dropped by default to prevent data leakage. Can be retained with `--keep-account-code` flag if needed for specific use cases.

**Dense Autoencoder configuration:**
- Latent dimension: 3 (smaller than the 5 input features, forcing compression)
- Epoch selection: Chosen based on validation loss on normal rows (not validation AUC like the LSTM-AE trainer)

**Training protocol:**
- All models are trained on normal rows from the training set only
- All models are evaluated on the same test rows for fair comparison
- Multi-seed evaluation uses the same receiver-stratified split across seeds

The command saves the following artifacts under `artifacts/`:
- `unsupervised_baseline_report.txt` or `multi_seed_unsupervised_baseline_report.txt` (when `--seeds` is used)
- `unsupervised_baseline_report.json` or `multi_seed_unsupervised_baseline_report.json` (when `--seeds` is used)

Optional arguments:

- `--project-root`: Project root containing data/raw/. Defaults to current working directory.
- `--dataset`: Dataset folder name in data/raw/ (e.g., paysim_synthetic_financial). Required.
- `--nrows`: Read only the first N rows. Defaults to all rows.
- `--seeds`: Comma-separated random seeds for repeated experiments (e.g., 42,123,2026). When set, trains and evaluates once per seed on the same data split and reports mean/std across runs. Default is single run with seed 42.
- `--keep-account-code`: Retain the account_code feature in the feature set. By default, account_code is dropped to prevent data leakage.

### 9. Train All Configurations

Train LSTM-Autoencoder for a grid of window sizes, seeds, and architectures in a single command.

```Shell
uv run python -m lstm_autoencoder train-all-configs --dataset paysim_synthetic_financial --windows 1,3,5,10 --seeds 42,123,2026 --architectures attention,bottleneck,plain,attention_pooling
```

**Features:**
- Resume mechanism: skips configurations where checkpoints already exist
- Error handling: continues on failure, reports summary at end
- Progress tracking: displays current configuration and overall progress

**Options:**
- `--project-root`: Project root containing data/raw/. Defaults to current working directory.
- `--dataset`: Dataset folder name in data/raw/ (e.g., paysim_synthetic_financial). Required.
- `--nrows`: Read only the first N rows. Defaults to all rows.
- `--windows`: Comma-separated window sizes (default: 1,3,5,10).
- `--seeds`: Comma-separated seeds (default: 42,123,2026).
- `--architectures`: Comma-separated architectures (default: all). Choices: attention, bottleneck, plain, attention_pooling.
- `--force`: Retrain even if checkpoints already exist (overwrites them).
- `--dry-run`: Print the commands that would run without training.

The full grid is 48 model configurations (4 windows × 4 architectures × 3 seeds) with the optimized GPU settings (batch_size=16384, num_workers=8).

### 10. Evaluate Hybrid All Configurations

Evaluate hybrid classifier for a grid of window sizes, seeds, and architectures in a single command.

```Shell
uv run python -m lstm_autoencoder eval-hybrid-all-configs --dataset paysim_synthetic_financial --windows 1,3,5,10 --seeds 42,123,2026 --architectures attention,bottleneck,plain,attention_pooling
```

**Features:**
- Resume mechanism: skips configurations where reports already exist
- Error handling: continues on failure, reports summary at end
- Progress tracking: displays current configuration and overall progress

**Options:**
- `--project-root`: Project root containing data/raw/. Defaults to current working directory.
- `--dataset`: Dataset folder name in data/raw/ (e.g., paysim_synthetic_financial). Required.
- `--nrows`: Read only the first N rows. Defaults to all rows.
- `--windows`: Comma-separated window sizes (default: 1,3,5,10).
- `--seeds`: Comma-separated seeds (default: 42,123,2026).
- `--architectures`: Comma-separated architectures (default: all). Choices: attention, bottleneck, plain, attention_pooling.
- `--force`: Re-evaluate even if reports already exist (overwrites them).
- `--dry-run`: Print the commands that would run without evaluating.
- `--bootstrap`: Paired bootstrap resamples per seed for the delta CIs (default: 200).

### 11. Run End-to-End Pipeline

Run the complete evaluation pipeline: hybrid evaluation for all configurations, unsupervised evaluation, and results summarization.

```Shell
uv run python -m lstm_autoencoder run-pipeline --dataset paysim_synthetic_financial --windows 1,3,5,10 --seeds 42,123,2026 --architectures attention,bottleneck,plain,attention_pooling
```

**Pipeline steps:**
1. Evaluate hybrid classifier for all window configurations
2. Evaluate unsupervised baselines (Isolation Forest, Dense Autoencoder)
3. Summarize all results into markdown tables

**Options:**
- `--project-root`: Project root containing data/raw/. Defaults to current working directory.
- `--dataset`: Dataset folder name in data/raw/ (e.g., paysim_synthetic_financial). Required.
- `--nrows`: Read only the first N rows. Defaults to all rows.
- `--windows`: Comma-separated window sizes (default: 1,3,5,10).
- `--seeds`: Comma-separated seeds (default: 42,123,2026).
- `--architectures`: Comma-separated architectures (default: all). Choices: attention, bottleneck, plain, attention_pooling.
- `--bootstrap`: Paired bootstrap resamples per seed for the delta CIs (default: 200).

### 12. Summarize Results

Turn the JSON reports that already exist under `artifacts/report/` into markdown tables for the Results section. This command only reads reports: nothing is trained and no dataset is loaded.

```Shell
uv run python -m lstm_autoencoder summarize-results --dataset paysim_synthetic_financial
```

The summarizer (ResultsSummarizer) writes three sections:

- **LSTM-AE standalone**: test AUC (mean ± std over seeds) per window and architecture. Architecture is taken from the `run_config` block of the training report (verified). Reports without `run_config` are listed separately as legacy, with a label inferred from the typed command (unverified)
- **Baselines**: the latest LR/GBM ablation report and the latest unsupervised baseline report
- **Hybrid classifier**: the latest report per window and architecture, with AUC, AP and recall at FPR 1% for every variant plus the paired deltas

The command saves the following artifact:

- `artifacts/summary/results_summary.md` (or the path given by `--out`)

Optional arguments:

- `--project-root`: Project root containing artifacts/. Defaults to current working directory.
- `--dataset`: Dataset folder name under `artifacts/report/` (e.g., paysim_synthetic_financial). Required.
- `--out`: Output markdown path. Defaults to `artifacts/summary/results_summary.md`.
- `--legacy-since`: Only list unverified training reports (no `run_config`) with this timestamp or later. Defaults to `20261106_103302`.

_Last Updated 6 November 2026_