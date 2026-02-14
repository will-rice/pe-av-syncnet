"""PyTorch Lightning DataModule for SyncNet training.

This module implements the data loading pipeline for training SyncNet models
using PyTorch Lightning. It handles dataset splitting.
"""

import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from syncnet.config import Config


class SyncNetDataModule(LightningDataModule):
    """PyTorch Lightning DataModule for SyncNet training and evaluation.

    This DataModule handles dataset splitting. All preprocessing is handled
    by the dataset itself, which returns Batch objects directly.

    Attributes:
        dataset: The underlying dataset containing video files.
        config: Configuration object with training parameters.
        num_workers: Number of worker processes for parallel data loading.
        train_dataset: Training split of the dataset.
        val_dataset: Validation split of the dataset.

    Args:
        dataset: PyTorch Dataset containing audio-visual samples.
        config: Configuration object with model and training settings.
        num_workers: Number of parallel workers for data loading. Default: 4.
    """

    train_dataset: Dataset
    val_dataset: Dataset

    def __init__(self, dataset: Dataset, config: Config, num_workers: int = 4) -> None:
        super().__init__()
        self.dataset = dataset
        self.config = config
        self.num_workers = num_workers

    def setup(self, stage: str | None = None) -> None:
        """Set up datasets for different training stages.

        Splits the dataset into training and validation sets.

        Args:
            stage: Current stage ('fit', 'validate', 'test', or None). If None
                or 'fit', datasets are prepared for training and validation.
        """
        if stage == "fit" or stage is None:
            dataset_size = len(self.dataset)  # ty: ignore[call-non-callable]
            train_size = int((1.0 - self.config.val_split) * dataset_size)
            val_size = dataset_size - train_size
            self.train_dataset, self.val_dataset = torch.utils.data.random_split(
                self.dataset, [train_size, val_size]
            )

    def train_dataloader(self) -> DataLoader:
        """Create and return the training data loader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=2 if self.num_workers > 0 else None,
        )

    def val_dataloader(self) -> DataLoader:
        """Create and return the validation data loader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=2 if self.num_workers > 0 else None,
        )
