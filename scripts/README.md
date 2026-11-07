This scripts directory is deprecated. All functionality has been moved to the main CLI commands.

## Migration Guide

The following scripts have been integrated into the main CLI:

### train_all_configs.py → train-all-configs
**Old:**
```bash
uv run python scripts/train_all_configs.py --windows 1,3,5,10 --seeds 42,123,2026 --architectures attention,bottleneck,plain,attention_pooling
```

**New:**
```bash
uv run python -m lstm_autoencoder train-all-configs --dataset paysim_synthetic_financial --windows 1,3,5,10 --seeds 42,123,2026 --architectures attention,bottleneck,plain,attention_pooling
```

### eval_hybrid_all_configs.py → eval-hybrid-all-configs
**Old:**
```bash
uv run python scripts/eval_hybrid_all_configs.py --windows 1,3,5,10 --seeds 42,123,2026 --architectures attention,bottleneck,plain,attention_pooling
```

**New:**
```bash
uv run python -m lstm_autoencoder eval-hybrid-all-configs --dataset paysim_synthetic_financial --windows 1,3,5,10 --seeds 42,123,2026 --architectures attention,bottleneck,plain,attention_pooling
```

### run.ps1 → run-pipeline
**Old:**
```powershell
.\scripts\run.ps1
```

**New:**
```bash
uv run python -m lstm_autoencoder run-pipeline --dataset paysim_synthetic_financial --windows 1,3,5,10 --seeds 42,123,2026 --architectures attention,bottleneck,plain,attention_pooling
```

### capture-command-uv.ps1
Removed - no longer needed. The reporter now has a fallback to `sys.argv` and the pipelines explicitly save structured `run_config` blocks in their JSON reports.

## Validation Scripts

### validate_checkpoints
Python script to validate all trained checkpoints can load and run properly.

**Features:**
- Loads each checkpoint and verifies it can be instantiated
- Runs a dummy forward pass to ensure model works
- Reports validation status for each checkpoint
- Groups results by window and architecture

**Usage:**
```bash
uv run python scripts/validate_checkpoints.py
```

This script scans `artifacts/model/paysim_synthetic_financial/` for all checkpoint files and validates they can be loaded and executed correctly.
