"""Train script."""

import logging
import os
from argparse import ArgumentParser
from pathlib import Path

import torch
from dotenv import load_dotenv
from git import Repo
from lightning import Trainer, seed_everything
from lightning.pytorch import callbacks, loggers

from syncnet.config import Config
from syncnet.datamodule import SyncNetDataModule
from syncnet.datasets.dataset import SyncNetDataset
from syncnet.lightning_module import SyncNetLightningModule


def main() -> None:
    r"""Execute the training pipeline for SyncNet models.

    This function orchestrates the entire training process:
    1. Parses command-line arguments
    2. Loads environment variables and validates configuration
    3. Initializes dataset, datamodule, and model
    4. Configures logging, callbacks, and trainer
    5. Executes training with automatic checkpointing

    The training uses PyTorch Lightning for training orchestration,
    Weights & Biases for experiment tracking, and supports distributed
    training across multiple GPUs with DeepSpeed.

    Command-line Arguments:
        data_root: Path to directory containing video files
        --project: Project name for logging (default: "template")
        --num_devices: Number of GPUs to use (default: 1)
        --num_workers: Number of data loading workers (default: 12)
        --log_root: Directory for saving logs and checkpoints (default: "logs")
        --checkpoint_path: Path to checkpoint for resuming training
        --weights_path: Path to pretrained weights to initialize model
        --debug: Enable debug mode with offline logging
        --fast_dev_run: Run single batch for testing

    Environment Variables:
        WANDB_PROJECT: Required. Weights & Biases project name
        WANDB_ENTITY: Required. Weights & Biases entity/username

    Raises:
        ValueError: If required environment variables are not set.

    Example:
        >>> # In command line:
        >>> python -m syncnet.scripts.train /data/videos \\
        ...     --num_devices 4 --num_workers 16 --project my_experiment
    """
    parser = ArgumentParser(description="Train script.")
    parser.add_argument("data_root", type=Path)
    parser.add_argument("--project", default="template", type=str)
    parser.add_argument("--num_devices", default=1, type=int)
    parser.add_argument("--num_workers", default=12, type=int)
    parser.add_argument("--log_root", default="logs", type=Path)
    parser.add_argument("--checkpoint_path", default=None, type=Path)
    parser.add_argument("--weights_path", type=Path, default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--fast_dev_run", action="store_true")
    args = parser.parse_args()
    load_dotenv()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Validate required environment variables
    wandb_project = os.environ.get("WANDB_PROJECT")
    wandb_entity = os.environ.get("WANDB_ENTITY")

    if not wandb_project or not wandb_entity:
        raise ValueError(
            "WANDB_PROJECT and WANDB_ENTITY must be set in environment variables. "
            "Copy .env.example to .env and set these values."
        )

    config = Config()

    seed_everything(config.seed, workers=True)

    git_repo = Repo()
    git_hash = git_repo.head.object.hexsha[:7]
    model_name = config.base_model.split("/")[-1]
    log_path = args.log_root / f"{model_name}-{git_hash}"
    log_path.mkdir(exist_ok=True, parents=True)

    dataset = SyncNetDataset(args.data_root)
    datamodule = SyncNetDataModule(dataset, config=config, num_workers=args.num_workers)
    lightning_module = SyncNetLightningModule(
        config=config,
        sync_dist=args.num_devices > 1,
        push_to_hub=not (args.fast_dev_run or args.debug),
    )
    # Load weights if specified
    if args.weights_path:
        logger.info(f"Loading weights from {args.weights_path}")
        lightning_module.model.load_state_dict(torch.load(args.weights_path))

    # Setup logger
    experiment_logger = loggers.WandbLogger(
        project=wandb_project,
        name=log_path.name,
        offline=args.debug,
        entity=wandb_entity,
    )

    # Setup callbacks
    lr_monitor = callbacks.LearningRateMonitor(logging_interval="step")
    early_stopping = callbacks.EarlyStopping(
        monitor="val_loss", patience=config.early_stopping_patience, mode="min"
    )
    model_checkpoint = callbacks.ModelCheckpoint(
        dirpath=log_path,
        filename="best-{epoch}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )

    callbacks_list = [lr_monitor, early_stopping, model_checkpoint]

    # Setup trainer
    last_checkpoint = log_path / "last.ckpt"
    if args.checkpoint_path:
        last_checkpoint = args.checkpoint_path

    trainer = Trainer(
        default_root_dir=log_path,
        max_epochs=config.max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=args.num_devices,
        logger=experiment_logger,
        precision="bf16-mixed" if torch.cuda.is_available() else 32,
        accumulate_grad_batches=config.accumulate_grad_batches,
        gradient_clip_val=config.gradient_clip_val,
        callbacks=callbacks_list,
        enable_checkpointing=True,
        val_check_interval=1000,
        limit_val_batches=100,
        fast_dev_run=args.fast_dev_run,
        strategy="deepspeed_stage_2" if args.num_devices > 1 else "auto",
    )

    trainer.fit(
        lightning_module,
        datamodule=datamodule,
        ckpt_path=last_checkpoint if last_checkpoint.exists() else None,
    )


if __name__ == "__main__":
    main()
