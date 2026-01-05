"""PyTorch Lightning DataModule for SyncNet training.

This module implements the data loading and preprocessing pipeline for training
SyncNet models using PyTorch Lightning. It handles dataset splitting, batch
collation, audio-visual preprocessing, and data augmentation.
"""

import random

import torch
import torchaudio
import torchvision
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from transformers.models.pe_audio_video import PeAudioVideoProcessor

from syncnet.config import Config
from syncnet.datasets import Batch


class SyncNetDataModule(LightningDataModule):
    """PyTorch Lightning DataModule for SyncNet training and evaluation.

    This DataModule handles all aspects of data loading, preprocessing, and
    augmentation for training audio-visual synchronization models. It manages
    dataset splitting, batch collation with variable-length sequences, audio
    resampling, video resizing, and negative sample generation.

    The module uses a HuggingFace processor for preprocessing and applies
    random temporal shifts to create negative (out-of-sync) samples for
    contrastive learning.

    Attributes:
        dataset: The underlying dataset containing video files.
        config: Configuration object with training parameters.
        num_workers: Number of worker processes for parallel data loading.
        processor: HuggingFace processor for audio-visual preprocessing.
        resample: Audio resampling transform (16kHz -> 48kHz).
        resize: Video frame resizing transform.
        train_dataset: Training split of the dataset.
        val_dataset: Validation split of the dataset.

    Args:
        dataset: PyTorch Dataset containing audio-visual samples.
        config: Configuration object with model and training settings.
        num_workers: Number of parallel workers for data loading. Default: 4.

    Example:
        >>> dataset = SyncNetDataset(Path("/data/videos"))
        >>> config = Config(batch_size=8)
        >>> datamodule = SyncNetDataModule(dataset, config, num_workers=8)
        >>> datamodule.setup("fit")
        >>> train_loader = datamodule.train_dataloader()
    """

    train_dataset: Dataset
    val_dataset: Dataset

    def __init__(self, dataset: Dataset, config: Config, num_workers: int = 4) -> None:
        super().__init__()
        self.dataset = dataset
        self.config = config
        self.num_workers = num_workers
        self.processor = PeAudioVideoProcessor.from_pretrained(config.base_model)
        self.resample = torchaudio.transforms.Resample(16000, 48000)
        self.resize = torchvision.transforms.Resize(
            (config.frame_height, config.frame_width)
        )

    def setup(self, stage: str | None = None) -> None:
        """Set up datasets for different training stages.

        Splits the dataset into training and validation sets using an 80/20 split.
        This method is called automatically by PyTorch Lightning before training.

        Args:
            stage: Current stage ('fit', 'validate', 'test', or None). If None
                or 'fit', datasets are prepared for training and validation.
        """
        if stage == "fit" or stage is None:
            dataset_size = len(self.dataset)  # type: ignore
            train_size = int(0.8 * dataset_size)
            val_size = dataset_size - train_size
            self.train_dataset, self.val_dataset = torch.utils.data.random_split(
                self.dataset, [train_size, val_size]
            )

    def train_dataloader(self) -> DataLoader:
        """Create and return the training data loader.

        Returns:
            DataLoader configured for training with shuffling, multiple workers,
            custom collation, and prefetching enabled.
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.pad_collate_fn,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=2 if self.num_workers > 0 else None,
        )

    def val_dataloader(self) -> DataLoader:
        """Create and return the validation data loader.

        Returns:
            DataLoader configured for validation without shuffling but with
            multiple workers and custom collation.
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.pad_collate_fn,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=2 if self.num_workers > 0 else None,
        )

    def pad_collate_fn(
        self, samples: list[tuple[torch.Tensor, torch.Tensor, dict]]
    ) -> Batch:
        """Custom collate function for batching variable-length audio-visual samples.

        This function processes a list of raw video samples and creates a training
        batch with the following steps:
        1. Extract random temporal segments from each video
        2. Convert stereo audio to mono
        3. Resize video frames to target dimensions
        4. Resample audio to target sample rate
        5. Randomly create negative samples by shifting audio
        6. Preprocess using HuggingFace processor
        7. Stack samples into batch tensors

        Args:
            samples: List of tuples, each containing (video, audio, metadata)
                from the dataset. Video shape: (T, H, W, C), Audio shape: (C, N).

        Returns:
            Batch object containing stacked audio tensors, video tensors, and
            binary labels indicating synchronization (1.0) or not (0.0).

        Note:
            - Samples with insufficient audio length (< 3200 samples) are skipped
            - Approximately 50% of samples are converted to negative examples
            - Errors during processing are caught and logged, skipping bad samples
        """
        video_segments, audio_segments, labels = [], [], []
        for sample in samples:
            try:
                video, audio, metadata = sample
                audio = audio.mean(dim=0, keepdim=True)  # Convert to mono if stereo
                video_segment, audio_segment = self.sample_random_segment(
                    video,
                    audio,
                    num_frames=self.config.num_frames,
                    fps=25,
                    sample_rate=16000,
                )
                if audio_segment.shape[1] < 3200:
                    continue
                video_segment = self.resize(video_segment.permute(0, 3, 1, 2))
                audio_segment = self.resample(audio_segment)
                if random.random() > 0.5:
                    audio_segment = audio_segment.roll(
                        1 if random.random() > 0.5 else -1, dims=-1
                    )
                    label = torch.tensor(0.0)
                else:
                    label = torch.tensor(1.0)

                input_values = self.processor(
                    videos=video_segment,
                    audio=audio_segment.squeeze(0),
                    return_tensors="pt",
                    padding=False,
                    sampling_rate=48000,
                )
                video_segments.append(input_values["pixel_values_videos"][0])
                audio_segments.append(input_values["input_values"][0])
                labels.append(label)
            except Exception as e:
                print(f"Error processing sample: {e}")
                continue

        return Batch(
            audio=torch.stack(audio_segments),
            video=torch.stack(video_segments),
            labels=torch.stack(labels),
        )

    @staticmethod
    def sample_random_segment(
        video: torch.Tensor,
        audio: torch.Tensor,
        num_frames: int,
        fps: int = 25,
        sample_rate: int = 16000,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample a random temporal segment from video with corresponding audio.

        Randomly selects a contiguous segment of video frames and extracts the
        temporally-aligned audio segment. The audio segment duration matches
        the video segment duration based on the fps and sample rate.

        Args:
            video: Video tensor of shape (T, H, W, C) where T is total frames,
                H and W are frame dimensions, and C is number of channels.
            audio: Audio tensor of shape (C, N) where C is number of channels
                and N is total audio samples.
            num_frames: Number of consecutive video frames to extract.
            fps: Frames per second of the video. Default: 25.
            sample_rate: Audio sample rate in Hz. Default: 16000.

        Returns:
            Tuple containing:
                - video_segment: Tensor of shape (num_frames, H, W, C)
                - audio_segment: Tensor of shape (C, num_samples) where
                  num_samples = num_frames / fps * sample_rate

        Raises:
            ValueError: If num_frames exceeds the video length.
        """
        start_frame = random.randint(0, video.shape[0] - num_frames)
        video_segment = video[start_frame : start_frame + num_frames]

        # Calculate corresponding audio segment
        audio_start_frame = int(start_frame / fps * sample_rate)
        num_audio_samples = int(num_frames / fps * sample_rate)
        audio_segment = audio[
            :, audio_start_frame : audio_start_frame + num_audio_samples
        ]

        return video_segment, audio_segment
