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

**Usage:**
```bash
uv run python scripts/train_all_configs.py
```

Each script will train 48 model configurations (4 windows × 4 architectures × 3 seeds) from scratch with the optimized GPU settings (batch_size=16384, num_workers=8).

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

## Utility Scripts

### fix_report_commands
Python script to update command fields in JSON reports to include specific architecture flags.

**Features:**
- Updates reports starting from a specified timestamp (20261106_103302)
- Uses checkpoint file modification times to match reports to architectures (more reliable than sequence-based matching)
- Matches reports to checkpoints based on temporal proximity
- Validates time differences (warns if > 5 minutes)
- Reports unmatched reports for manual review
- Updates command field from generic to specific (e.g., `--window 5 --bottleneck`)

**Usage:**
```bash
uv run python scripts/fix_report_commands.py
```

This script fixes reports that were generated before the architecture_config feature was added to the reporter. It uses checkpoint file modification times to reliably match reports to their corresponding architectures. If checkpoints were modified after training, manual review may be needed. 