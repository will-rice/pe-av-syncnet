#!/usr/bin/env python3
"""Filter out videos that fail to load in the SyncNet dataset.

This script tests all MP4 files in a directory by attempting to load them
using the same decoders used in the SyncNetDataset. Videos that fail to load
are either removed or moved to a separate directory.

Usage:
    python filter_bad_videos.py /path/to/videos --remove
    python filter_bad_videos.py /path/to/videos --move-to /path/to/bad_videos
    python filter_bad_videos.py /path/to/videos --dry-run
"""

import argparse
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from torchcodec.decoders import AudioDecoder, VideoDecoder
from tqdm import tqdm


def test_video(video_path: Path) -> tuple[bool, str | None]:
    """Test if a video can be loaded successfully.

    Args:
        video_path: Path to the video file to test.

    Returns:
        Tuple of (success, error_message). If success is True, error_message is None.
    """
    try:
        # Try to load video frames
        video_decoder = VideoDecoder(str(video_path))
        frame_batch = video_decoder.get_frames_in_range(
            0, video_decoder.metadata.num_frames
        )
        _ = frame_batch.data.permute(0, 2, 3, 1)

        # Try to load audio
        audio_decoder = AudioDecoder(str(video_path))
        audio_samples = audio_decoder.get_all_samples()
        _ = audio_samples.data

        # Check minimum requirements
        if video_decoder.metadata.num_frames < 5:
            return False, "Too few frames (< 5)"

        if audio_samples.data.shape[1] < 3200:
            return False, "Audio too short (< 3200 samples)"

        return True, None

    except Exception as e:
        return False, str(e)


def filter_videos(
    data_root: Path,
    remove: bool = False,
    move_to: Path | None = None,
    dry_run: bool = False,
    max_workers: int | None = None,
) -> None:
    """Filter out bad videos from the dataset.

    Args:
        data_root: Root directory containing video files.
        remove: If True, delete bad videos.
        move_to: If provided, move bad videos to this directory.
        dry_run: If True, only report bad videos without taking action.
        max_workers: Number of parallel workers. Defaults to CPU count.
    """
    # Find all MP4 files
    video_paths = list(data_root.rglob("*.mp4"))
    print(f"Found {len(video_paths)} video files")

    if max_workers is None:
        max_workers = os.cpu_count() or 4
    print(f"Using {max_workers} worker threads")

    if move_to and not dry_run:
        move_to.mkdir(parents=True, exist_ok=True)

    bad_videos = []
    good_videos = []

    # Test all videos in parallel
    print("\nTesting videos...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_path = {
            executor.submit(test_video, video_path): video_path
            for video_path in video_paths
        }

        # Process results as they complete with progress bar
        for future in tqdm(as_completed(future_to_path), total=len(video_paths)):
            video_path = future_to_path[future]
            success, error = future.result()

            if success:
                good_videos.append(video_path)
            else:
                bad_videos.append((video_path, error))

    # Report results
    print(f"\n{'=' * 80}")
    print(f"Results:")
    print(f"  Good videos: {len(good_videos)}")
    print(f"  Bad videos:  {len(bad_videos)}")
    print(f"  Success rate: {len(good_videos) / len(video_paths) * 100:.1f}%")
    print(f"{'=' * 80}\n")

    if bad_videos:
        print("Bad videos:")
        for video_path, error in bad_videos:
            rel_path = video_path.relative_to(data_root)
            print(f"  {rel_path}: {error}")

        # Take action on bad videos
        if dry_run:
            print(f"\n[DRY RUN] No action taken")
        elif remove:
            print(f"\nRemoving {len(bad_videos)} bad videos...")
            for video_path, _ in tqdm(bad_videos):
                video_path.unlink()
            print("Done!")
        elif move_to:
            print(f"\nMoving {len(bad_videos)} bad videos to {move_to}...")
            for video_path, _ in tqdm(bad_videos):
                # Preserve directory structure
                rel_path = video_path.relative_to(data_root)
                dest_path = move_to / rel_path
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(video_path), str(dest_path))
            print("Done!")
        else:
            print("\nNo action specified. Use --remove or --move-to to handle bad videos.")

    else:
        print("All videos loaded successfully!")

    # Save list of good videos
    good_list_path = data_root / "good_videos.txt"
    if not dry_run:
        with open(good_list_path, "w") as f:
            for video_path in sorted(good_videos):
                rel_path = video_path.relative_to(data_root)
                f.write(f"{rel_path}\n")
        print(f"\nSaved list of good videos to {good_list_path}")


def main() -> None:
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Filter out videos that fail to load in SyncNet dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "data_root",
        type=Path,
        help="Root directory containing video files",
    )
    parser.add_argument(
        "--remove",
        action="store_true",
        help="Remove bad videos (WARNING: destructive operation)",
    )
    parser.add_argument(
        "--move-to",
        type=Path,
        help="Move bad videos to this directory",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report bad videos without taking action",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: CPU count)",
    )

    args = parser.parse_args()

    if not args.data_root.exists():
        print(f"Error: {args.data_root} does not exist")
        return

    if args.remove and args.move_to:
        print("Error: Cannot specify both --remove and --move-to")
        return

    if args.remove and not args.dry_run:
        response = input(
            "WARNING: This will permanently delete bad videos. Continue? [y/N] "
        )
        if response.lower() != "y":
            print("Aborted")
            return

    filter_videos(
        data_root=args.data_root,
        remove=args.remove,
        move_to=args.move_to,
        dry_run=args.dry_run,
        max_workers=args.max_workers,
    )


if __name__ == "__main__":
    main()