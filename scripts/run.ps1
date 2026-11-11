$ErrorActionPreference = "Stop"

$DATASET = "paysim_synthetic_financial"
$WINDOWS = "1,3,5,10"
$ARCHS = "attention,bottleneck,plain,attention_pooling"
$SEEDS_ALL = "42,123,2026"
$SEEDS_MISSING = "123,2026" # 42 already re-run post-fix for all W x arch

# 1. Re-verify no leakage under the fixed pipeline (fast, no training)
uv run python -m lstm_autoencoder audit-pipeline `
    --dataset $DATASET

if ($LASTEXITCODE -ne 0) { throw "audit-pipeline failed" }


# 2. Train the missing seeds (123, 2026) for every window x architecture.
# No --force: existing checkpoints (seed=42) will be skipped.
uv run python -m lstm_autoencoder train-all-configs `
    --dataset $DATASET `
    --windows $WINDOWS `
    --seeds $SEEDS_MISSING `
    --architectures $ARCHS

if ($LASTEXITCODE -ne 0) { throw "train-all-configs failed" }


# 3. Select score strategy (validation split only) before evaluating on test
uv run python -m lstm_autoencoder select-score-strategy `
    --dataset $DATASET `
    --windows $WINDOWS `
    --architectures $ARCHS `
    --seeds $SEEDS_ALL

if ($LASTEXITCODE -ne 0) { throw "select-score-strategy failed" }


# 4. Re-aggregate standalone LSTM-AE eval across all 3 seeds
uv run python -m lstm_autoencoder eval-standalone `
    --dataset $DATASET `
    --windows $WINDOWS `
    --seeds $SEEDS_ALL

if ($LASTEXITCODE -ne 0) { throw "eval-standalone failed" }


# 5. Hybrid classifier (LR full + AE error) across all 3 seeds,
# receiver-cluster bootstrap
uv run python -m lstm_autoencoder eval-hybrid-all-configs `
    --dataset $DATASET `
    --windows $WINDOWS `
    --seeds $SEEDS_ALL `
    --architectures $ARCHS `
    --bootstrap 2000 `
    --bootstrap-mode receiver

if ($LASTEXITCODE -ne 0) { throw "eval-hybrid-all-configs failed" }


# 6. Score-definition diagnostics (mean/last/max)
# per window x architecture, validation split only

$ARCH_FLAGS = @(
    @(),
    @("--bottleneck"),
    @("--plain"),
    @("--attention-pooling")
)

foreach ($W in @(1, 3, 5, 10)) {
    foreach ($FLAG in $ARCH_FLAGS) {

        $ARGS = @(
            "-m", "lstm_autoencoder",
            "diagnose-scores",
            "--dataset", $DATASET,
            "--window", "$W"
        )

        # Add architecture flag only when present
        $ARGS += $FLAG
        $ARGS += @(
            "--seeds", $SEEDS_ALL,
            "--split", "val"
        )

        uv run python @ARGS

        if ($LASTEXITCODE -ne 0) {
            throw "diagnose-scores failed for window=$W, flag=$($FLAG -join ' ')"
        }
    }
}


# 7. Rebuild results_summary.md from all the reports above
uv run python -m lstm_autoencoder summarize-results `
    --dataset $DATASET

if ($LASTEXITCODE -ne 0) { throw "summarize-results failed" }

Write-Host "`nAll experiments completed successfully." -ForegroundColor Green