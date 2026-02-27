"""Dataset class for loading audio-visual data for SyncNet training.

This module provides a PyTorch Dataset implementation for loading video files
containing both audio and visual streams. It efficiently loads only the required
frames and audio samples for each training sample.
"""

import os
import random
from pathlib import Path

import torch
import torchaudio
import torchvision
from torch.utils.data import Dataset
from torchcodec.decoders import AudioDecoder, VideoDecoder
from transformers.models.pe_audio_video import PeAudioVideoProcessor

from syncnet.config import Config
from syncnet.datasets import Batch

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class SyncNetDataset(Dataset):
    """PyTorch Dataset for loading audio-visual synchronization training data.

    This dataset loads video files (MP4 format) from a directory tree and
    efficiently extracts only the required video frames and audio samples.
    It handles all preprocessing including resizing, resampling, and
    negative sample generation.

    Attributes:
        data_root: Path to the root directory containing video files.
        sample_paths: List of paths to all MP4 files found in the directory tree.
        config: Configuration object with training parameters.
        processor: HuggingFace processor for audio-visual preprocessing.

    Args:
        root: Root directory path containing the video files.
        config: Configuration object with model and training settings.
    """

    def __init__(self, root: Path, config: Config) -> None:
        super().__init__()
        self.data_root = root
        self.sample_paths = list(root.rglob("*.mp4"))
        self.config = config
        self.processor = PeAudioVideoProcessor.from_pretrained(config.base_model)
        self.target_sample_rate = self.processor.feature_extractor.sampling_rate  # ty: ignore[unresolved-attribute]
        self.resize = torchvision.transforms.Resize(
            (config.frame_height, config.frame_width)
        )

    def __len__(self) -> int:
        """Return the total number of video samples available."""
        return len(self.sample_paths)

    def __getitem__(self, idx: int) -> Batch:
        """Load and return processed video, audio, and label for a sample.

        Efficiently loads only the required frames and audio samples,
        applies preprocessing, and generates positive/negative labels.

        Returns:
            Batch containing processed video, audio, and label tensors.
        """
        video_path = str(self.sample_paths[idx])

        # Get video metadata first to determine valid frame range
        video_decoder = VideoDecoder(video_path)
        total_frames = video_decoder.metadata.num_frames
        fps = video_decoder.metadata.average_fps

        # Pick random start frame for video (ensure non-negative range)
        end_frame = max(0, total_frames - self.config.num_frames)
        video_start = random.randint(0, end_frame) if end_frame > 0 else 0

        # Determine if this is a negative sample (mismatched audio/video)
        is_negative = random.random() > self.config.negative_fraction
        if is_negative and end_frame > 0:
            # Pick a different audio start position
            audio_frame_start = random.randint(0, end_frame)
            while abs(audio_frame_start - video_start) <= 1:
                audio_frame_start = random.randint(0, end_frame)
        else:
            audio_frame_start = video_start

        # Load only the required video frames
        frame_batch = video_decoder.get_frames_in_range(
            video_start, min(video_start + self.config.num_frames, total_frames)
        )
        # (T, C, H, W) format from torchcodec
        video = frame_batch.data

        # Resize video frames
        video = self.resize(video)

        # Pad or truncate video to num_frames
        if video.shape[0] < self.config.num_frames:
            # Pad with zeros
            padding = torch.zeros(
                self.config.num_frames - video.shape[0],
                video.shape[1],
                video.shape[2],
                video.shape[3],
                dtype=video.dtype,
            )
            video = torch.cat([video, padding], dim=0)
        elif video.shape[0] > self.config.num_frames:
            video = video[: self.config.num_frames]

        # Calculate audio time range and load only required samples
        audio_start_time = audio_frame_start / float(fps)
        audio_end_time = (audio_frame_start + self.config.num_frames) / fps

        audio_decoder = AudioDecoder(video_path)
        audio_samples = audio_decoder.get_samples_played_in_range(
            audio_start_time, audio_end_time
        )
        source_sample_rate = audio_samples.sample_rate
        audio = audio_samples.data  # Shape: (C, N)

        # Convert to mono
        audio = audio.mean(dim=0, keepdim=True)

        # Resample audio to target sample rate
        if source_sample_rate != self.target_sample_rate:
            resample = torchaudio.transforms.Resample(
                source_sample_rate, self.target_sample_rate
            )
            audio = resample(audio)

        # Calculate expected audio length and pad/truncate
        expected_audio_len = int(self.config.num_frames / fps * self.target_sample_rate)
        if audio.shape[1] < expected_audio_len:
            # Pad with zeros
            padding = torch.zeros(
                1, expected_audio_len - audio.shape[1], dtype=audio.dtype
            )
            audio = torch.cat([audio, padding], dim=1)
        elif audio.shape[1] > expected_audio_len:
            audio = audio[:, :expected_audio_len]

        # Process through HuggingFace processor
        input_values = self.processor(
            videos=video[..., video.shape[2] // 2 :, :],
            audio=audio.squeeze(0),
            return_tensors="pt",
            padding="do_not_pad",
            sampling_rate=self.target_sample_rate,
        )

        return Batch(
            video=input_values["pixel_values_videos"][0],
            audio=input_values["input_values"][0],
            labels=torch.tensor(0.0 if is_negative else 1.0),
        )
