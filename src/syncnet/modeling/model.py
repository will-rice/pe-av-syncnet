"""SyncNet model architecture for audio-visual synchronization.

This module implements the SyncNet model that learns to determine whether
audio and video streams are temporally synchronized. The model uses a
pretrained audio-visual encoder and computes cosine similarity between
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

    This model determines whether audio and video streams are synchronized
    by computing the similarity between learned embeddings. It uses a
    pretrained audio-visual encoder (PeAudioVideo) to extract separate
    embeddings for audio and video, then computes their cosine similarity.

    The model applies:
    1. Feature extraction using pretrained encoder
    2. Flattening of temporal/spatial dimensions
    3. L2 normalization of embeddings
    4. ReLU activation for non-negative similarities
    5. Cosine similarity computation

    Attributes:
        encoder: Pretrained PeAudioVideoModel for feature extraction.
        similarity_fn: Cosine similarity function for comparing embeddings.

    Args:
        config: SyncNetConfig containing model configuration including
            the pretrained model name/path.

    Example:
        >>> config = SyncNetConfig(base_model="facebook/pe-av-small")
        >>> model = SyncNet(config)
        >>> audio = torch.randn(4, 1024)
        >>> video = torch.randn(4, 5, 3, 224, 224)
        >>> similarity = model(audio, video)
        >>> print(similarity.shape)
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
        self.similarity_fn = nn.CosineSimilarity(dim=-1)

    def forward(
        self, input_values: torch.Tensor, pixel_values: torch.Tensor
    ) -> torch.Tensor:
        """Compute synchronization similarity between audio and video.

        Processes audio and video inputs through the encoder, normalizes
        the resulting embeddings, and computes cosine similarity to determine
        if they are synchronized.

        Args:
            input_values: Preprocessed audio tensor from the processor.
                Shape: (batch_size, audio_features).
            pixel_values: Preprocessed video frames from the processor.
                Shape: (batch_size, num_frames, channels, height, width).

        Returns:
            Similarity scores between 0 and 1, where higher values indicate
            better synchronization. Shape: (batch_size,).

        Note:
            The embeddings are flattened, L2-normalized, and ReLU-activated
            before similarity computation to ensure positive, bounded scores.
        """
        outputs = self.encoder(
            input_values=input_values, pixel_values_videos=pixel_values
        )
        audio_emb = outputs.audio_embeds
        face_emb = outputs.video_embeds

        audio_emb = audio_emb.view(audio_emb.size(0), -1)
        face_emb = face_emb.view(face_emb.size(0), -1)

        audio_emb = nn.functional.normalize(audio_emb, p=2.0, dim=1)
        face_emb = nn.functional.normalize(face_emb, p=2.0, dim=1)

        similarity = self.similarity_fn(audio_emb.relu(), face_emb.relu()).squeeze(-1)
        return similarity
