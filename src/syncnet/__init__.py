"""SyncNet: Audio-Visual Synchronization Package.

This package provides tools for training and evaluating SyncNet models,
which determine whether audio and video streams are synchronized.

The package includes:
- Configuration management for model training
- PyTorch Lightning modules for training
- Data loading and preprocessing utilities
- Pre-trained model architectures

Example:
    Basic usage for training a SyncNet model:

    >>> from syncnet.config import Config
    >>> from syncnet.lightning_module import SyncNetLightningModule
    >>> config = Config()
    >>> model = SyncNetLightningModule(config)
"""
