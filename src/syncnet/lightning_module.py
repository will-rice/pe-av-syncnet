"""PyTorch Lightning Module for SyncNet model training.

This module implements the training loop, validation, optimization, and model
checkpointing for the SyncNet audio-visual synchronization model using PyTorch
Lightning framework.
"""

from pathlib import Path

import torch
import torch.nn as nn
from lightning.pytorch import LightningModule
from lightning.pytorch.utilities.memory import garbage_collection_cuda
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

    The module uses BCE with logits loss to train the model to output positive
    logits for synchronized audio-visual pairs and negative logits for out-of-sync
    pairs. It tracks training loss, validation loss, and validation accuracy.

    Attributes:
        config: Configuration object with training hyperparameters.
        model: SyncNet model for audio-visual synchronization.
        push_to_hub: Whether to push model checkpoints to HuggingFace Hub.
        sync_dist: Whether to synchronize metrics across distributed processes.
        loss_fn: BCE with logits loss function.
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
    """

    def __init__(
        self, config: Config, push_to_hub: bool = False, sync_dist: bool = False
    ) -> None:
        """Initialize the Lightning Module with config and training options."""
        super().__init__()
        self.save_hyperparameters(config.model_dump())
        self.config = config
        self.model = SyncNet(SyncNetConfig(**config.model_dump()))
        self.push_to_hub = push_to_hub
        self.sync_dist = sync_dist
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.val_metrics = MetricCollection({"val_accuracy": Accuracy(task="binary")})
        self.processor = PeAudioVideoProcessor.from_pretrained(config.base_model)
        self.lowest_val_loss = float("inf")

    def configure_model(self) -> None:
        """Set the model to training mode."""
        self.model.train()

    def training_step(self, batch: Batch, batch_idx: int) -> torch.Tensor:
        """Execute a single training step."""
        logits = self.model(batch.audio, batch.video)
        loss = self.loss_fn(logits, batch.labels.squeeze(-1))
        self.log("train_loss", self.train_loss(loss), prog_bar=True)
        return loss

    def validation_step(self, batch: Batch, batch_idx: int) -> torch.Tensor:
        """Execute a single validation step."""
        logits = self.model(batch.audio, batch.video)
        loss = self.loss_fn(logits, batch.labels.squeeze(-1))
        self.val_loss.update(loss)
        self.val_metrics.update(logits, batch.labels.squeeze(-1))
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
        if self.config.lr_scheduler == "onecycle":
            lr_scheduler: torch.optim.lr_scheduler.LRScheduler = (
                torch.optim.lr_scheduler.OneCycleLR(
                    optimizer,
                    max_lr=self.config.learning_rate,
                    total_steps=self.config.max_epochs,
                    pct_start=0.1,
                    anneal_strategy="cos",
                    final_div_factor=self.config.learning_rate
                    / self.config.min_learning_rate,
                )
            )
        elif self.config.lr_scheduler == "constant":
            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=lambda epoch: 1.0
            )
        else:
            raise ValueError(
                f"Unsupported lr_scheduler: {self.config.lr_scheduler}. "
                "Choose 'onecycle' or 'constant'."
            )
        return [optimizer], [lr_scheduler]
