"""PyTorch Lightning Module for SyncNet model training.

This module implements the training loop, validation, optimization, and model
checkpointing for the SyncNet audio-visual synchronization model using PyTorch
Lightning framework.
"""

from pathlib import Path

import torch
from lightning.pytorch import LightningModule
from lightning.pytorch.utilities.memory import garbage_collection_cuda
from torch import nn
from torchmetrics import Accuracy, MeanMetric, MetricCollection
from transformers.models.pe_audio_video import PeAudioVideoProcessor

from syncnet.config import Config
from syncnet.datasets import Batch
from syncnet.modeling.model import SyncNet, SyncNetConfig


class SyncNetLightningModule(LightningModule):
    """PyTorch Lightning Module for training SyncNet models.

    This module encapsulates the full training pipeline including model
    initialization, training/validation steps, metric tracking, optimizer
    configuration, and optional model uploading to HuggingFace Hub.

    The module uses Binary Cross-Entropy loss to train the model to distinguish
    between synchronized and out-of-sync audio-visual pairs. It tracks training
    loss, validation loss, and validation accuracy, and automatically saves the
    best model based on validation loss.

    Attributes:
        config: Configuration object with training hyperparameters.
        model: SyncNet model for audio-visual synchronization.
        push_to_hub: Whether to push model checkpoints to HuggingFace Hub.
        sync_dist: Whether to synchronize metrics across distributed processes.
        loss_fn: Binary cross-entropy loss function.
        train_loss: Metric for tracking mean training loss.
        val_loss: Metric for tracking mean validation loss.
        val_metrics: Collection of validation metrics including accuracy.
        processor: HuggingFace processor for preprocessing.
        lowest_val_loss: Best validation loss achieved during training.

    Args:
        config: Configuration object with model and training settings.
        push_to_hub: If True, automatically push model to HuggingFace Hub when
            validation loss improves. Default: False.
        sync_dist: If True, synchronize metrics across distributed processes
            during multi-GPU training. Default: False.

    Example:
        >>> config = Config(learning_rate=1e-4, max_epochs=100)
        >>> module = SyncNetLightningModule(config, push_to_hub=True)
        >>> trainer = Trainer(max_epochs=100)
        >>> trainer.fit(module, datamodule)
    """

    def __init__(
        self, config: Config, push_to_hub: bool = False, sync_dist: bool = False
    ) -> None:
        """Initialize the Lightning Module with config and training options.

        Args:
            config: Configuration object with model and training settings.
            push_to_hub: Whether to push to HuggingFace Hub on improvement.
            sync_dist: Whether to sync metrics across distributed processes.
        """
        super().__init__()
        self.save_hyperparameters(config.model_dump())
        self.config = config
        self.model = SyncNet(SyncNetConfig(**config.model_dump()))
        self.push_to_hub = push_to_hub
        self.sync_dist = sync_dist
        self.loss_fn = nn.BCELoss()
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.val_metrics = MetricCollection({"val_accuracy": Accuracy(task="binary")})
        self.processor = PeAudioVideoProcessor.from_pretrained(config.base_model)
        self.lowest_val_loss = float("inf")

    def training_step(self, batch: Batch, batch_idx: int) -> None:
        """Execute a single training step.

        Performs forward pass through the model, computes loss, and logs metrics.
        The loss is computed using Binary Cross-Entropy between predicted
        similarity scores and ground truth labels.

        Args:
            batch: Batch object containing audio, video, and labels.
            batch_idx: Index of the current batch (unused but required by Lightning).

        Returns:
            Loss tensor for backpropagation.
        """
        similarity = self.model(batch.audio, batch.video)
        with torch.autocast(device_type=self.device.type, enabled=False):
            loss = self.loss_fn(similarity.squeeze(-1), batch.labels.squeeze(-1))
        self.log("train_loss", self.train_loss(loss), prog_bar=True)
        return loss

    def validation_step(self, batch: Batch, batch_idx: int) -> None:
        """Execute a single validation step.

        Performs forward pass and computes validation metrics without gradient
        computation. Updates running metrics for loss and accuracy.

        Args:
            batch: Batch object containing audio, video, and labels.
            batch_idx: Index of the current batch (unused but required by Lightning).

        Returns:
            Loss tensor for the validation batch.
        """
        similarity = self.model(batch.audio, batch.video)
        with torch.autocast(device_type=self.device.type, enabled=False):
            loss = self.loss_fn(similarity.squeeze(-1), batch.labels.squeeze(-1))
        self.val_loss.update(loss)
        self.val_metrics.update(similarity.squeeze(-1), batch.labels.squeeze(-1))
        return loss

    def on_validation_epoch_end(self) -> None:
        """Process metrics and save models at the end of validation epoch.

        This method:
        1. Computes and logs validation metrics (loss and accuracy)
        2. Checks if current validation loss is the best so far
        3. If best, optionally pushes model and processor to HuggingFace Hub
        4. Resets metrics for next epoch
        5. Performs CUDA garbage collection to free memory

        Note:
            Model uploading to Hub requires proper authentication and will
            create a private repository named after the log directory.
        """
        self.log_dict(self.val_metrics.compute(), sync_dist=self.sync_dist)
        val_loss = self.val_loss.compute().item()
        self.log("val_loss", val_loss, prog_bar=True, sync_dist=self.sync_dist)
        if val_loss < self.lowest_val_loss:
            self.lowest_val_loss = val_loss

            log_path = Path(self.trainer.default_root_dir)
            if self.push_to_hub:
                try:
                    # Push main model
                    self.model.push_to_hub(  # type: ignore[call-arg]
                        repo_id=log_path.name,
                        commit_message="Add model checkpoint",
                        token=True,
                        private=True,
                    )

                    self.processor.push_to_hub(
                        repo_id=log_path.name,
                        commit_message="Add tokenizer",
                        token=True,
                        private=True,
                    )
                except Exception as e:
                    print(f"Failed to push to hub: {e}")

        self.val_metrics.reset()
        self.val_loss.reset()
        garbage_collection_cuda()

    def configure_optimizers(self) -> tuple[list[torch.optim.Optimizer], list]:
        """Configure optimizers and learning rate schedulers.

        Sets up AdamW optimizer with weight decay and OneCycleLR scheduler
        for cosine annealing learning rate schedule with warmup.

        Returns:
            Tuple containing:
                - List with single AdamW optimizer
                - List with single OneCycleLR scheduler

        Note:
            The scheduler uses:
            - 10% of training for warmup (pct_start=0.1)
            - Cosine annealing strategy
            - Final learning rate of config.min_learning_rate
        """
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.config.learning_rate,
            total_steps=self.config.max_epochs,
            pct_start=0.1,
            anneal_strategy="cos",
            final_div_factor=self.config.learning_rate / self.config.min_learning_rate,
        )
        return [optimizer], [lr_scheduler]
