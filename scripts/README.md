This scripts directory used for
- script that run explicitly just for one method on huge class on some pipeline.
- inject uv for capture command with system (powershell) and then delivere it to python

## Training Scripts

### train_all_configs
Python script to train LSTM-Autoencoder for all window sizes, seeds, and architectures.

**Configurations:**
- Windows: 1, 3, 5, 10
- Seeds: 42, 123, 2026
- Architectures: attention, bottleneck, plain, attention_pooling

**Features:**
- Resume mechanism: skips configurations where checkpoints already exist
- Streaming output: real-time training progress
- Error handling: continues on failure, reports summary at end
- Colored output: easy to track progress

**Options:**
- `--architectures a,b`: which architectures to train (default: all). Choices: `attention`, `bottleneck`, `plain`, `attention_pooling`
- `--windows 1,3,5,10`: window sizes (default: 1,3,5,10)
- `--seeds 42,123,2026`: seeds (default: 42,123,2026)
- `--force`: retrain even when checkpoints already exist (overwrites them)
- `--dry-run`: print the commands that would run / be skipped, without training

**Usage:**
```bash
# everything (resumes: skips configurations whose checkpoints all exist)
uv run python scripts/train_all_configs.py

# retrain only attention pooling for all windows (existing checkpoints are invalid -> --force)
uv run python scripts/train_all_configs.py --architectures attention_pooling --force

# preview first
uv run python scripts/train_all_configs.py --architectures attention_pooling --force --dry-run
```

The full grid is 48 model configurations (4 windows × 4 architectures × 3 seeds) with the optimized GPU settings (batch_size=16384, num_workers=8).

**Reports:** each training report JSON now contains a `run_config` block (architecture, window, seeds, model flags, checkpoint names) and each run records its `architecture` and `checkpoint_path`. The `architecture` comes from the model that was actually built; the trainer raises if it differs from the requested config. The free-text `command` field is only what was typed.

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