"""Contrastive Loss for Audio-Video Synchronization."""

import torch
import torch.nn as nn


class ContrastiveLoss(nn.Module):
    """Contrastive loss for metric learning.

    This loss function minimizes distance for positive pairs (synchronized
    audio-video) and pushes apart negative pairs (out-of-sync).

    For positive pairs: L = D²
    For negative pairs: L = max(margin - D, 0)²

    where D is the Euclidean distance.

    Args:
        margin: Minimum distance for negative pairs. Pairs further than this
            margin incur no loss. Default: 1.0.
    """

    def __init__(self, margin: float = 1.0) -> None:
        super().__init__()
        self.margin = margin

    def forward(self, distance: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute contrastive loss.

        Args:
            distance: Euclidean distance between embeddings. Shape: (batch_size,).
            labels: Binary labels where 1 = synced (positive), 0 = out-of-sync.
                Shape: (batch_size,).

        Returns:
            Scalar loss tensor.
        """
        positive_loss = labels * distance.pow(2)
        negative_loss = (1 - labels) * torch.clamp(self.margin - distance, min=0).pow(2)
        return (positive_loss + negative_loss).mean()
