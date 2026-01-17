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

    This model outputs sync logits based on the cosine similarity between
    audio and video embeddings. Positive logits indicate synchronized pairs.

    Attributes:
        encoder: Pretrained PeAudioVideoModel for feature extraction.

    Args:
        config: SyncNetConfig containing model configuration including
            the pretrained model name/path.

    Example:
        >>> config = SyncNetConfig(base_model="facebook/pe-av-small")
        >>> model = SyncNet(config)
        >>> audio = torch.randn(4, 48000)
        >>> video = torch.randn(4, 5, 3, 224, 224)
        >>> logits = model(audio, video)
        >>> print(logits.shape)
        torch.Size([4])
    """

    def __init__(self, config: SyncNetConfig) -> None:
        """Initialize SyncNet with pretrained encoder.

        Args:
            config: Configuration object containing model parameters.
        """
        super().__init__(config=config)
        self.encoder = PeAudioVideoModel.from_pretrained(config.base_model)
        self.encoder.gradient_checkpointing_enable()
        self.logit_scale = nn.Parameter(torch.tensor(0.0))  # exp(0)=1

    def forward(
        self, input_values: torch.Tensor, pixel_values: torch.Tensor
    ) -> torch.Tensor:
        """Compute sync logits for audio-video pairs.

        Args:
            input_values: Preprocessed audio tensor from the processor.
                Shape: (batch_size, audio_features).
            pixel_values: Preprocessed video frames from the processor.
                Shape: (batch_size, num_frames, channels, height, width).

        Returns:
            Sync logits. Shape: (batch_size,).
        """
        outputs = self.encoder(
            input_values=input_values, pixel_values_videos=pixel_values
        )
        audio_emb = nn.functional.normalize(outputs.audio_embeds, 2.0, -1)
        video_emb = nn.functional.normalize(outputs.video_embeds, 2.0, -1)
        sim = (audio_emb * video_emb).sum(dim=-1)
        logits = sim * self.logit_scale.exp()
        return logits
