"""Data loading, preprocessing, and PyTorch datasets."""

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset


class SequenceDataset(Dataset):
    """Dataset containing fixed-length sensor sequences and targets."""

    def __init__(self, sequences: torch.Tensor, targets: torch.Tensor) -> None:
        if len(sequences) != len(targets):
            raise ValueError("sequences and targets must have the same length")
        self.sequences = sequences
        self.targets = targets

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.sequences[index], self.targets[index]


def load_dataframe(path: str | Path) -> pd.DataFrame:
    """Load a CSV dataset from disk."""
    dataset_path = Path(path)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    return pd.read_csv(dataset_path)
