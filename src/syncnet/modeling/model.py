"""SyncNet model architecture for audio-visual synchronization.

This module implements the SyncNet model that learns to determine whether
audio and video streams are temporally synchronized. The model uses a
pretrained audio-visual encoder and computes the Euclidean distance between
audio and video embeddings.
"""

import torch
import torch.nn as nn
from transformers import PreTrainedConfig, PreTrainedModel
from transformers.models.pe_audio_video import PeAudioVideoModel


class SyncNetConfig(PreTrainedConfig):
    """Configuration class for SyncNet model.

    Extends HuggingFace's PreTrainedConfig to enable model serialization
    and integration with the Transformers ecosystem. All configuration
    parameters from the main Config class can be passed to this.

    Attributes:
        model_type: Identifier for this model architecture ("syncnet").

    Args:
        **kwargs: Arbitrary configuration parameters passed to parent class.
            Typically includes base_model, learning_rate, etc. from Config.

    Example:
        >>> config = SyncNetConfig(base_model="facebook/pe-av-small")
        >>> model = SyncNet(config)
    """

    model_type = "syncnet"

    def __init__(self, **kwargs: object) -> None:
        """Initialize configuration with arbitrary parameters.

        Args:
            **kwargs: Configuration parameters to store.
        """
        super().__init__(**kwargs)  # type: ignore


class SyncNet(PreTrainedModel):
    """Audio-Visual Synchronization Model using pretrained encoder.

    This model computes the Euclidean distance between L2-normalized audio
    and video embeddings. Lower distance means more synchronized.

    Attributes:
        encoder: Pretrained PeAudioVideoModel for feature extraction.

    Args:
        config: SyncNetConfig containing model configuration including
            the pretrained model name/path.

    Example:
        >>> config = SyncNetConfig(base_model="facebook/pe-av-small")
        >>> model = SyncNet(config)
        >>> audio = torch.randn(4, 1024)
        >>> video = torch.randn(4, 5, 3, 224, 224)
        >>> distance = model(audio, video)
        >>> print(distance.shape)
        torch.Size([4])
    """

    def __init__(self, config: SyncNetConfig) -> None:
        """Initialize SyncNet with pretrained encoder.

        Args:
            config: Configuration object containing model parameters.
        """
        super().__init__(config=config)
        self.encoder = PeAudioVideoModel.from_pretrained(config.base_model)
        self.encoder.gradient_checkpointing = True

    def forward(
        self, input_values: torch.Tensor, pixel_values: torch.Tensor
    ) -> torch.Tensor:
        """Compute Euclidean distance between audio and video embeddings.

        Args:
            input_values: Preprocessed audio tensor from the processor.
                Shape: (batch_size, audio_features).
            pixel_values: Preprocessed video frames from the processor.
                Shape: (batch_size, num_frames, channels, height, width).

        Returns:
            Euclidean distance between embeddings. Shape: (batch_size,).
            Lower values indicate better synchronization.
        """
        outputs = self.encoder(
            input_values=input_values, pixel_values_videos=pixel_values
        )
        audio_emb = nn.functional.normalize(outputs.audio_embeds, p=2.0, dim=1)
        video_emb = nn.functional.normalize(outputs.video_embeds, p=2.0, dim=1)
        return nn.functional.pairwise_distance(audio_emb, video_emb)

    def get_sync_probability(
        self,
        input_values: torch.Tensor,
        pixel_values: torch.Tensor,
        margin: float = 1.0,
        temperature: float = 5.0,
    ) -> torch.Tensor:
        """Compute probability that audio and video are synchronized.

        Args:
            input_values: Preprocessed audio tensor from the processor.
            pixel_values: Preprocessed video frames from the processor.
            margin: Decision boundary distance (should match training). Default: 1.0.
            temperature: Controls sharpness of probability transition. Default: 5.0.

        Returns:
            Probability scores in [0, 1]. Shape: (batch_size,).
        """
        distance = self(input_values, pixel_values)
        return torch.sigmoid((margin - distance) * temperature)
