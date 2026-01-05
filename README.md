# SyncNet: Audio-Visual Synchronization

A PyTorch Lightning implementation of SyncNet for detecting audio-visual synchronization in videos. This project trains deep learning models to determine whether audio and video streams are temporally aligned.

## Overview

SyncNet learns to measure the synchronization between audio and video by computing similarity scores between learned embeddings. The model uses a pretrained audio-visual encoder (PeAudioVideo) and is trained using contrastive learning with both synchronized (positive) and out-of-sync (negative) samples.

### Key Applications

- **Lip-sync detection**: Verify if speech audio matches visible lip movements
- **Video quality assessment**: Detect audio-visual synchronization issues
- **Deepfake detection**: Identify manipulated videos with mismatched audio
- **Video post-production**: Automated sync checking for edited content

## Features

- **Pretrained Encoder**: Built on HuggingFace's PeAudioVideo model
- **PyTorch Lightning**: Clean, scalable training framework with minimal boilerplate
- **Multi-GPU Support**: Distributed training with DeepSpeed Stage 2
- **Mixed Precision Training**: Automatic BF16 mixed precision for faster training
- **Comprehensive Logging**: Weights & Biases integration with metric tracking
- **Data Augmentation**: Automatic negative sample generation via temporal shifts
- **Gradient Checkpointing**: Memory-efficient training for large models
- **Type Safety**: Full type annotations with mypy validation
- **Modern Tooling**: Fast dependency management with `uv`

## Installation

### Prerequisites

- Python 3.12+
- CUDA-capable GPU (recommended)
- Git

### 1. Install uv (Fast Python Package Manager)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Clone the Repository

```bash
git clone https://github.com/yourusername/pe-av-syncnet.git
cd pe-av-syncnet
```

### 3. Install Dependencies

```bash
uv sync
```

This will install all required dependencies including:

- PyTorch with CUDA support
- PyTorch Lightning
- Transformers (for PeAudioVideo model)
- TorchAudio and TorchVision
- Weights & Biases
- And more...

### 4. Set Up Environment Variables

```bash
cp .env.example .env
```

Edit `.env` and add your credentials:

```bash
WANDB_PROJECT=your-project-name
WANDB_ENTITY=your-wandb-username
```

### 5. Install Pre-commit Hooks (Optional)

```bash
uv run pre-commit install
```

## Project Structure

```
pe-av-syncnet/
├── src/syncnet/
│   ├── __init__.py              # Package initialization
│   ├── config.py                # Pydantic configuration with hyperparameters
│   ├── lightning_module.py      # Lightning training module
│   ├── datamodule.py           # Data loading and preprocessing
│   ├── datasets/
│   │   ├── __init__.py         # Batch data structure
│   │   └── dataset.py          # Video dataset loader
│   ├── modeling/
│   │   ├── __init__.py         # Model package
│   │   └── model.py            # SyncNet architecture
│   └── scripts/
│       ├── __init__.py         # Scripts package
│       └── train.py            # Training script
├── tests/
│   └── test_sample.py          # Test suite
├── pyproject.toml              # Project configuration and dependencies
├── .pre-commit-config.yaml     # Code quality hooks
├── .env.example                # Environment variables template
└── README.md                   # This file
```

## Dataset Preparation

SyncNet expects a directory containing MP4 video files with both audio and video streams.

### Dataset Structure

```
data/
├── video1.mp4
├── video2.mp4
├── video3.mp4
├── subfolder/
│   ├── video4.mp4
│   └── video5.mp4
└── ...
```

### Requirements

- **Format**: MP4 files with H.264 video and AAC audio
- **Audio**: Preferably mono or stereo, will be converted to mono
- **Video**: Any resolution (will be resized to 224x224)
- **Frame Rate**: 25 fps recommended
- **Audio Sample Rate**: 16kHz or 48kHz
- **Duration**: At least 0.2 seconds (5 frames at 25fps)

### Dataset Recommendations

- **Minimum size**: 1000+ videos for meaningful training
- **Diversity**: Include various speakers, environments, and scenarios
- **Quality**: Clear audio with visible speakers for best results

## Usage

### Basic Training

Train a model on your video dataset:

```bash
uv run train /path/to/videos --num_devices 1 --num_workers 8
```

### Multi-GPU Training

Train with multiple GPUs using DeepSpeed:

```bash
uv run train /path/to/videos --num_devices 4 --num_workers 16
```

### Resume Training from Checkpoint

```bash
uv run train /path/to/videos --checkpoint_path logs/pe-av-small-abc1234/last.ckpt
```

### Load Pretrained Weights

Initialize model with custom weights:

```bash
uv run train /path/to/videos --weights_path /path/to/weights.pth
```

### Debug Mode

Run training offline without uploading to Weights & Biases:

```bash
uv run train /path/to/videos --debug
```

### Fast Development Run

Test your pipeline with a single batch:

```bash
uv run train /path/to/videos --fast_dev_run
```

### Command-Line Arguments

| Argument            | Type | Default    | Description                         |
| ------------------- | ---- | ---------- | ----------------------------------- |
| `data_root`         | Path | Required   | Directory containing video files    |
| `--project`         | str  | "template" | Project name for logging            |
| `--num_devices`     | int  | 1          | Number of GPUs to use               |
| `--num_workers`     | int  | 12         | Data loading workers                |
| `--log_root`        | Path | "logs"     | Directory for checkpoints and logs  |
| `--checkpoint_path` | Path | None       | Path to checkpoint for resuming     |
| `--weights_path`    | Path | None       | Path to pretrained weights          |
| `--debug`           | flag | False      | Enable debug mode (offline logging) |
| `--fast_dev_run`    | flag | False      | Run single batch for testing        |

## Model Architecture

### SyncNet Model

The SyncNet model consists of:

1. **Pretrained Encoder**: PeAudioVideoModel from HuggingFace
   - Processes audio and video separately
   - Extracts rich multimodal embeddings
   - Gradient checkpointing enabled for memory efficiency

2. **Embedding Processing**:
   - Flatten temporal/spatial dimensions
   - L2 normalization
   - ReLU activation (ensures positive similarity)

3. **Similarity Computation**:
   - Cosine similarity between audio and video embeddings
   - Output: Score from 0 to 1 (higher = better sync)

### Training Process

```
Input Video → Random Segment Sampling
           ↓
Audio + Video Preprocessing
           ↓
[50% chance] Temporal Shift (negative sample)
           ↓
PeAudioVideo Encoder
           ↓
Audio Embedding + Video Embedding
           ↓
Cosine Similarity
           ↓
Binary Cross-Entropy Loss
```

## Configuration

All hyperparameters are defined in `src/syncnet/config.py`:

```python
class Config(BaseModel):
    # Reproducibility
    seed: int = 42

    # Data
    test_split: float = 0.05
    batch_size: int = 4

    # Training
    max_epochs: int = 200
    early_stopping_patience: int = 10
    learning_rate: float = 1e-4
    min_learning_rate: float = 1e-6
    weight_decay: float = 1e-2
    accumulate_grad_batches: int = 1
    gradient_clip_val: float = 1.0

    # Model
    base_model: str = "facebook/pe-av-small"
    num_frames: int = 5
    negative_fraction: float = 0.5
    frame_height: int = 224
    frame_width: int = 224
```

### Key Parameters

- **base_model**: HuggingFace model ID for the pretrained encoder
- **num_frames**: Number of video frames per sample (5 frames = 0.2s at 25fps)
- **negative_fraction**: Proportion of negative samples (0.5 = 50% out-of-sync)
- **batch_size**: Adjust based on GPU memory (4 works well for most GPUs)
- **learning_rate**: Initial learning rate with OneCycleLR scheduler

## Training Details

### Data Augmentation

- **Random temporal cropping**: Samples random 5-frame segments from videos
- **Negative sample generation**: 50% of samples get audio shifted by ±1 frame
- **Stereo to mono conversion**: Automatically handles stereo audio
- **Resampling**: Audio resampled from 16kHz to 48kHz

### Optimization

- **Optimizer**: AdamW with weight decay
- **Scheduler**: OneCycleLR with cosine annealing
  - 10% warmup period
  - Peak learning rate: `config.learning_rate`
  - Final learning rate: `config.min_learning_rate`

### Metrics

- **Training**: Binary Cross-Entropy loss
- **Validation**: BCE loss + binary accuracy
- **Logging**: Real-time metrics to Weights & Biases

### Automatic Model Saving

- Saves best model based on validation loss
- Optionally pushes to HuggingFace Hub (private repos)
- Local checkpointing with automatic resumption

## Development

### Running Tests

```bash
uv run pytest
```

### Type Checking

```bash
uv run mypy src/
```

### Linting and Formatting

```bash
# Check code style
uv run ruff check src/

# Auto-format code
uv run ruff format src/
```

### Pre-commit Hooks

Automatically run linters and formatters before each commit:

```bash
uv run pre-commit run --all-files
```

## Model Inference

After training, use the model for inference:

```python
import torch
from syncnet.modeling.model import SyncNet, SyncNetConfig
from transformers.models.pe_audio_video import PeAudioVideoProcessor

# Load model
config = SyncNetConfig(base_model="facebook/pe-av-small")
model = SyncNet.from_pretrained("your-username/your-model-name")
model.eval()

# Load processor
processor = PeAudioVideoProcessor.from_pretrained("facebook/pe-av-small")

# Process inputs
inputs = processor(
    videos=video_frames,  # Shape: (num_frames, H, W, C)
    audio=audio_samples,   # Shape: (num_samples,)
    return_tensors="pt",
    sampling_rate=48000
)

# Inference
with torch.no_grad():
    similarity = model(
        inputs["input_values"],
        inputs["pixel_values_videos"]
    )

print(f"Synchronization score: {similarity.item():.4f}")
# Higher score = better synchronization
```

## Troubleshooting

### Common Issues

**Out of Memory (OOM)**

- Reduce `batch_size` in config.py
- Reduce `num_workers` to decrease memory overhead
- Enable gradient accumulation: `accumulate_grad_batches=2`

**Slow Data Loading**

- Increase `num_workers` (recommended: 2-4x number of GPUs)
- Ensure videos are on fast storage (SSD preferred)
- Enable `persistent_workers=True` (already enabled)

**Low Accuracy**

- Ensure dataset has sufficient diversity
- Increase training epochs
- Adjust `negative_fraction` (try 0.3-0.7)
- Verify audio and video are actually synchronized in source data

**WANDB Authentication Error**

- Set `WANDB_PROJECT` and `WANDB_ENTITY` in `.env`
- Run `wandb login` to authenticate
- Use `--debug` flag to train offline

### Related Work

- **SyncNet**: [Out of time: automated lip sync in the wild](https://www.robots.ox.ac.uk/~vgg/software/lipsync/)
- **PeAudioVideo**: HuggingFace Transformers multimodal encoder

## License

See [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes with tests
4. Ensure all tests and pre-commit hooks pass
5. Submit a pull request

## Acknowledgments

- Built with [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/)
- Uses [HuggingFace Transformers](https://huggingface.co/docs/transformers/)
- Dependency management by [uv](https://github.com/astral-sh/uv)
- Experiment tracking with [Weights & Biases](https://wandb.ai/)

## Support

For questions or issues:

- Open an issue on GitHub
- Check existing issues for solutions
- Refer to documentation in docstrings

---

**Note**: This is a research/educational implementation. For production use, additional validation and optimization may be required.
