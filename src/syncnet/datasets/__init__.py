"""Dataset utilities and data structures for SyncNet.

This module provides data structures and utilities for handling audio-visual
data in the SyncNet training pipeline. It defines the batch structure used
throughout the training process.
"""

from typing import NamedTuple

import torch


class Batch(NamedTuple):
    """A batch of audio-visual training data with labels.

    This named tuple represents a single training batch containing preprocessed
    audio and video tensors along with synchronization labels.

    Attributes:
        audio: Preprocessed audio tensor of shape (batch_size, audio_features).
            Contains the audio embeddings or spectrograms for each sample.
        video: Preprocessed video tensor of shape
        (batch_size, num_frames, channels, height, width).
            Contains the video frames for each sample.
        labels: Binary labels of shape (batch_size,) indicating whether audio
            and video are synchronized (1.0) or not (0.0).

    Example:
        >>> batch = Batch(
        ...     audio=torch.randn(4, 1024),
        ...     video=torch.randn(4, 5, 3, 224, 224),
        ...     labels=torch.tensor([1.0, 0.0, 1.0, 0.0])
        ... )
        >>> print(batch.audio.shape)
        torch.Size([4, 1024])
    """

    audio: torch.Tensor
    video: torch.Tensor
    labels: torch.Tensor
