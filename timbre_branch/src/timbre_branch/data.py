"""PyTorch dataset helpers for precomputed shared-encoder embeddings."""

from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


class TimbreEmbeddingDataset(Dataset):
    def __init__(
        self,
        track_ids: Sequence[str],
        embeddings: np.ndarray,
        standardized_targets: np.ndarray,
        valid_mask: np.ndarray,
    ) -> None:
        self.track_ids = np.asarray(track_ids, dtype=str)
        self.embeddings = torch.as_tensor(embeddings, dtype=torch.float32)
        self.targets = torch.as_tensor(standardized_targets, dtype=torch.float32)
        self.valid_mask = torch.as_tensor(valid_mask, dtype=torch.bool)
        row_count = len(self.track_ids)
        if self.embeddings.shape != (row_count, 128):
            raise ValueError("embeddings must have shape [samples, 128]")
        if self.targets.shape != (row_count, 35):
            raise ValueError("standardized_targets must have shape [samples, 35]")
        if self.valid_mask.shape != self.targets.shape:
            raise ValueError("valid_mask must match standardized_targets")

    def __len__(self) -> int:
        return len(self.track_ids)

    def __getitem__(self, index: int):
        return (
            self.embeddings[index],
            self.targets[index],
            self.valid_mask[index],
            self.track_ids[index],
        )
