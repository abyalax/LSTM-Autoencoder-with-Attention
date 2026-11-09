"""PyTorch Dataset wrapping sequence batches for the LSTM-Autoencoder."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from lstm_autoencoder.model.sequence_builder import SequenceBatch


class TransactionSequenceDataset(Dataset):
    """Expose sequences, masks, and labels as PyTorch tensors."""

    def __init__(self, batch: SequenceBatch, indices: np.ndarray | None = None) -> None:
        """Wrap a SequenceBatch, optionally restricted to a subset of rows.

        Arrays are moved to shared memory so multiprocessing DataLoader workers
        map the same region instead of each receiving a full pickled copy
        without this, large windows (e.g. W=10) overflow the IPC pipe on Windows.
        """
        self._indices = indices if indices is not None else np.arange(len(batch.labels))
        self._sequences = torch.from_numpy(batch.sequences).share_memory_()
        self._mask = torch.from_numpy(batch.mask).share_memory_()
        self._labels = torch.from_numpy(batch.labels.astype("float32")).share_memory_()

    def __len__(self) -> int:
        """Number of examples in this subset."""
        return len(self._indices)

    def valid_steps(self) -> np.ndarray:
        """Number of real (non-padded) steps of every example in this subset, in dataset order."""
        rows = torch.from_numpy(np.asarray(self._indices, dtype="int64"))
        return self._mask[rows].sum(dim=1).numpy().astype("int64")

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (sequence, mask, label) for one example."""
        row = self._indices[index]
        return self._sequences[row], self._mask[row], self._labels[row]
