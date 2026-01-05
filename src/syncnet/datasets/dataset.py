"""Dataset class for loading audio-visual data for SyncNet training.

This module provides a PyTorch Dataset implementation for loading video files
containing both audio and visual streams. It recursively searches for MP4 files
in a given directory and loads them for training.
"""

import os
from pathlib import Path

from torch.utils.data import Dataset
from torchvision.io import read_video

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class SyncNetDataset(Dataset):
    """PyTorch Dataset for loading audio-visual synchronization training data.

    This dataset loads video files (MP4 format) from a directory tree and
    extracts both video frames and audio waveforms. It's designed for training
    audio-visual synchronization models.

    The dataset recursively searches for all .mp4 files in the provided root
    directory and its subdirectories. Each sample contains the video frames,
    audio waveform, and metadata from the video file.

    Attributes:
        data_root: Path to the root directory containing video files.
        sample_paths: List of paths to all MP4 files found in the directory tree.

    Args:
        root: Root directory path containing the video files. The dataset will
            recursively search this directory for all .mp4 files.

    Example:
        >>> dataset = SyncNetDataset(Path("/path/to/videos"))
        >>> print(len(dataset))
        1000
        >>> video, audio, metadata = dataset[0]
        >>> print(video.shape, audio.shape)
        torch.Size([100, 224, 224, 3]) torch.Size([2, 48000])

    Note:
        This class sets TOKENIZERS_PARALLELISM=false to avoid warnings when
        using HuggingFace tokenizers in multiprocessing data loaders.
    """

    def __init__(self, root: Path) -> None:
        """Initialize the dataset with the root directory path.

        Args:
            root: Path to the root directory containing video files.
        """
        super().__init__()
        self.data_root = root
        self.sample_paths = list(root.rglob("*.mp4"))

    def __len__(self) -> int:
        """Return the number of samples in the dataset.

        Returns:
            The total number of video files found in the dataset.
        """
        return len(self.sample_paths)

    def __getitem__(self, idx: int) -> tuple:
        """Load and return the video, audio, and metadata for a sample.

        Args:
            idx: Index of the sample to load (0 to len(dataset)-1).

        Returns:
            A tuple containing:
                - video: Tensor of shape (T, H, W, C) with video frames.
                - audio: Tensor of shape (C, N) with audio samples.
                - metadata: Dictionary with video metadata (fps, duration, etc).

        Raises:
            IndexError: If idx is out of bounds.
            RuntimeError: If the video file cannot be read or is corrupted.
        """
        video, audio, metadata = read_video(str(self.sample_paths[idx]), pts_unit="sec")
        return video, audio, metadata
