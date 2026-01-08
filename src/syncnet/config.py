"""Configuration module for SyncNet training and model parameters.

This module defines the main configuration class that controls all aspects
of the SyncNet model training process, including data loading, model
architecture, training hyperparameters, and reproducibility settings.
"""

from pydantic import BaseModel, Field


class Config(BaseModel):
    """Configuration class for SyncNet model training and evaluation.

    This class uses Pydantic for configuration management with automatic
    validation and default values. It organizes settings into logical groups:
    reproducibility, data processing, training hyperparameters, and model
    architecture.

    Attributes:
        seed: Random seed for reproducibility across runs.
        val_split: Fraction of the dataset to use for validation/testing.
        batch_size: Number of samples per batch during training.
        max_epochs: Maximum number of training epochs.
        early_stopping_patience: Number of epochs to wait before early stopping.
        learning_rate: Initial learning rate for the optimizer.
        min_learning_rate: Minimum learning rate for the scheduler.
        weight_decay: L2 regularization coefficient.
        accumulate_grad_batches: Number of batches to accumulate gradients.
        gradient_clip_val: Maximum gradient norm for clipping.
        base_model: HuggingFace model identifier or path to pretrained model.
        num_frames: Number of video frames to process per sample.
        negative_fraction: Fraction of negative (out-of-sync) samples.
        frame_height: Target height for input video frames in pixels.
        frame_width: Target width for input video frames in pixels.

    Example:
        >>> config = Config(batch_size=8, learning_rate=1e-3)
        >>> print(config.seed)
        42
    """

    # Reproducibility
    seed: int = Field(default=42, description="Random seed for reproducibility.")

    # Data
    val_split: float = Field(
        default=0.05, description="Proportion of data for validation."
    )
    batch_size: int = Field(default=4, description="Batch size.")

    # Training
    max_epochs: int = Field(default=200, description="Maximum number of epochs.")
    early_stopping_patience: int = Field(
        default=10, description="Early stopping patience."
    )
    learning_rate: float = Field(default=1e-4, description="Initial learning rate.")
    min_learning_rate: float = Field(default=1e-6, description="Minimum learning rate.")
    weight_decay: float = Field(default=1e-2, description="Weight decay for optimizer.")
    accumulate_grad_batches: int = Field(
        default=1, description="Number of gradient accumulation."
    )
    gradient_clip_val: float = Field(
        default=1.0, description="Gradient clipping value."
    )

    # Model
    base_model: str = Field(
        default="facebook/pe-av-small", description="Base model name or path."
    )

    num_frames: int = Field(default=5, description="Number of video frames to process.")
    negative_fraction: float = Field(
        default=0.5, description="Fraction of negative samples."
    )
    frame_height: int = Field(default=224, description="Height of input video frames.")
    frame_width: int = Field(default=224, description="Width of input video frames.")
