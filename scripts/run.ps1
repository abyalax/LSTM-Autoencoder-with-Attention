# End-to-end pipeline script for LSTM-Autoencoder project
# Runs: ablation, training, hybrid evaluation, unsupervised evaluation, summarize

$ErrorActionPreference = "Stop"

$dataset = "paysim_synthetic_financial"
$windows = @("1", "3", "5", "10")
$seeds = "42,123,2026"
$architectures = "attention,bottleneck,plain,attention_pooling"

# Step 1: Ablation Baselines
# uv run python -m lstm_autoencoder ablation --dataset $dataset
# if ($LASTEXITCODE -ne 0) {
#     exit 1
# }

# Step 2: Train LSTM-AE for all configurations
# uv run python scripts/train_all_configs.py --windows ($windows -join ",") --seeds $seeds --architectures $architectures
# if ($LASTEXITCODE -ne 0) {
#     exit 1
# }

# Step 3: Evaluate Hybrid Classifier for all configurations
foreach ($window in $windows) {

    uv run python scripts/eval_hybrid_all_configs.py `
        --windows $window `
        --seeds $seeds `
        --architectures $architectures 

    if ($LASTEXITCODE -ne 0) {
        Write-Error "Hybrid evaluation failed for window $window"
        exit 1
    }
}

# Step 4: Evaluate Unsupervised Baselines
uv run python -m lstm_autoencoder eval-unsupervised --dataset $dataset --seeds $seeds
if ($LASTEXITCODE -ne 0) {
    exit 1
}

# Step 5: Summarize Results
uv run python -m lstm_autoencoder summarize-results --dataset $dataset
if ($LASTEXITCODE -ne 0) {
    exit 1
}

# Final Summary


# Suspend laptop after pipeline completion
Add-Type -AssemblyName System.Windows.Forms
[System.Windows.Forms.Application]::SetSuspendState(
    [System.Windows.Forms.PowerState]::Suspend,
    $false,
    $false
)
