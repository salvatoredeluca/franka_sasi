#!/usr/bin/env python3
"""
convert_lerobot_to_zarr.py

Offline converter: LeRobot dataset  ->  Zarr dataset
for a dual Franka FR3 teleoperation setup.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INPUT  — LeRobot dataset (produced by record_lerobot_dataset.py)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  lerobot_data/
  ├── data/chunk-000/episode_XXXXXX.parquet
  │     columns: observation.state (16,)
  │              action            (14,)
  │              timestamp, frame_index, episode_index, task_index
  ├── videos/
  │   ├── observation.images.camera_left/chunk-000/episode_XXXXXX.mp4
  │   └── observation.images.camera_right/chunk-000/episode_XXXXXX.mp4
  └── meta/
      ├── info.json
      └── episodes/chunk-000/file-000.parquet

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT — Zarr dataset
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  dataset.zarr/
  ├── data/
  │   ├── state        (N_total, 16)  float32
  │   │     [0..6]  left  EE pose (x,y,z,qx,qy,qz,qw)
  │   │     [7]     left  gripper_state {0.0, 1.0}
  │   │     [8..14] right EE pose
  │   │     [15]    right gripper_state
  │   │
  │   ├── action       (N_total, 14)  float32
  │   │     [0..5]  left  EE twist (dx,dy,dz,dwx,dwy,dwz)
  │   │     [6]     left  gripper_cmd
  │   │     [7..12] right EE twist
  │   │     [13]    right gripper_cmd
  │   │
  │   ├── camera_left  (N_total, H, W, 3)  uint8
  │   └── camera_right (N_total, H, W, 3)  uint8
  │
  └── meta/
      └── episode_ends (n_episodes,)  int64
            episode_ends[i] = index of the LAST frame of episode i
            (same convention as the original ROS1 zarr script)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
USAGE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  python convert_lerobot_to_zarr.py \
      --input_dir  ./lerobot_data \
      --output_dir ./dataset.zarr \
      --image_size 224

Dependencies:
  pip install zarr pandas pyarrow opencv-python-headless tqdm
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import zarr
from tqdm import tqdm


# ── helpers ───────────────────────────────────────────────────────────────────

def load_info(lerobot_dir: Path) -> dict:
    """Read meta/info.json from the LeRobot dataset."""
    info_path = lerobot_dir / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"info.json not found at {info_path}")
    with open(info_path) as f:
        return json.load(f)


def find_episode_parquets(lerobot_dir: Path) -> list[Path]:
    """
    Return all episode parquet files sorted by episode index.
    LeRobot v2.1 layout: data/chunk-XXX/episode_XXXXXX.parquet
    """
    parquets = sorted(
        (lerobot_dir / "data").rglob("episode_*.parquet")
    )
    if not parquets:
        raise FileNotFoundError(
            f"No episode parquet files found under {lerobot_dir / 'data'}")
    return parquets


def find_video(lerobot_dir: Path, camera_key: str, episode_idx: int) -> Path:
    """
    Locate the MP4 file for a given camera and episode index.
    LeRobot v2.1 layout:
      videos/<camera_key>/chunk-XXX/episode_XXXXXX.mp4
    """
    pattern = f"episode_{episode_idx:06d}.mp4"
    matches = list(
        (lerobot_dir / "videos" / camera_key).rglob(pattern)
    )
    if not matches:
        raise FileNotFoundError(
            f"Video not found: camera={camera_key} episode={episode_idx}")
    return matches[0]


def decode_video_frames(video_path: Path,
                        image_size: int) -> np.ndarray:
    """
    Decode all frames of an MP4 file into a uint8 numpy array.

    Returns:
        frames: (T, H, W, 3) uint8  RGB order
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frames = []
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        # OpenCV reads BGR -> convert to RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        # resize if needed
        if frame_rgb.shape[:2] != (image_size, image_size):
            frame_rgb = cv2.resize(
                frame_rgb, (image_size, image_size),
                interpolation=cv2.INTER_LINEAR)
        frames.append(frame_rgb)

    cap.release()

    if not frames:
        raise RuntimeError(f"No frames decoded from: {video_path}")

    return np.stack(frames, axis=0)   # (T, H, W, 3)


# ── main conversion ───────────────────────────────────────────────────────────

def convert(lerobot_dir: Path,
            output_dir:  Path,
            image_size:  int) -> None:

    print(f"Input  (LeRobot): {lerobot_dir}")
    print(f"Output (Zarr)   : {output_dir}")

    # ── read dataset metadata ─────────────────────────────────────────────────
    info        = load_info(lerobot_dir)
    fps         = info["fps"]
    n_episodes  = info["total_episodes"]
    total_frames = info["total_frames"]

    print(f"Episodes : {n_episodes}")
    print(f"Frames   : {total_frames}")
    print(f"FPS      : {fps}")

    # ── find all episode parquet files ────────────────────────────────────────
    parquet_files = find_episode_parquets(lerobot_dir)
    assert len(parquet_files) == n_episodes, (
        f"Expected {n_episodes} parquet files, found {len(parquet_files)}")

    # ── open Zarr store ───────────────────────────────────────────────────────
    store      = zarr.open(str(output_dir), mode="w")
    data_group = store.create_group("data")
    meta_group = store.create_group("meta")

    # ── pre-allocate Zarr arrays ──────────────────────────────────────────────
    # Using appendable arrays (chunks along axis 0) so we can fill
    # episode by episode without loading everything into RAM at once.
    z_state = data_group.zeros(
        "state",
        shape=(total_frames, 16),
        chunks=(256, 16),
        dtype=np.float32,
    )
    z_action = data_group.zeros(
        "action",
        shape=(total_frames, 14),
        chunks=(256, 14),
        dtype=np.float32,
    )
    z_cam_left = data_group.zeros(
        "camera_left",
        shape=(total_frames, image_size, image_size, 3),
        chunks=(32, image_size, image_size, 3),
        dtype=np.uint8,
    )
    z_cam_right = data_group.zeros(
        "camera_right",
        shape=(total_frames, image_size, image_size, 3),
        chunks=(32, image_size, image_size, 3),
        dtype=np.uint8,
    )

    # episode_ends[i] = index of the LAST frame of episode i
    # Example: episode 0 has 120 frames -> episode_ends[0] = 119
    #          episode 1 has  90 frames -> episode_ends[1] = 209
    episode_ends = np.zeros(n_episodes, dtype=np.int64)

    # ── iterate episodes ──────────────────────────────────────────────────────
    frame_offset = 0   # running index into the global frame arrays

    for ep_idx, parquet_path in enumerate(
            tqdm(parquet_files, desc="Converting episodes")):

        # ── load parquet ──────────────────────────────────────────────────────
        df = pd.read_parquet(parquet_path)
        n_frames = len(df)

        # ── state (N, 16) ─────────────────────────────────────────────────────
        # LeRobot stores list columns as Python lists inside parquet.
        # np.stack converts them to a proper 2D array.
        state = np.stack(
            df["observation.state"].values
        ).astype(np.float32)   # (N, 16)

        # ── action (N, 14) ────────────────────────────────────────────────────
        action = np.stack(
            df["action"].values
        ).astype(np.float32)   # (N, 14)

        # ── decode video frames ───────────────────────────────────────────────
        video_left  = find_video(lerobot_dir,
                                 "observation.images.camera_left",  ep_idx)
        video_right = find_video(lerobot_dir,
                                 "observation.images.camera_right", ep_idx)

        frames_left  = decode_video_frames(video_left,  image_size)  # (N,H,W,3)
        frames_right = decode_video_frames(video_right, image_size)  # (N,H,W,3)

        # sanity check: parquet rows must match video frames
        assert frames_left.shape[0]  == n_frames, (
            f"Episode {ep_idx}: parquet has {n_frames} rows but "
            f"camera_left video has {frames_left.shape[0]} frames")
        assert frames_right.shape[0] == n_frames, (
            f"Episode {ep_idx}: parquet has {n_frames} rows but "
            f"camera_right video has {frames_right.shape[0]} frames")

        # ── write to Zarr ─────────────────────────────────────────────────────
        end = frame_offset + n_frames

        z_state[frame_offset:end]      = state
        z_action[frame_offset:end]     = action
        z_cam_left[frame_offset:end]   = frames_left
        z_cam_right[frame_offset:end]  = frames_right

        # episode_ends: index of last frame (inclusive), same as ROS1 script
        episode_ends[ep_idx] = end - 1

        frame_offset = end

    # ── write episode_ends metadata ───────────────────────────────────────────
    meta_group.array(
        "episode_ends",
        data=episode_ends,
        chunks=True,
        dtype=np.int64,
    )

    # ── summary ───────────────────────────────────────────────────────────────
    print("\nZarr dataset written successfully.")
    print(f"  data/state        : {z_state.shape}   {z_state.dtype}")
    print(f"  data/action       : {z_action.shape}  {z_action.dtype}")
    print(f"  data/camera_left  : {z_cam_left.shape} {z_cam_left.dtype}")
    print(f"  data/camera_right : {z_cam_right.shape} {z_cam_right.dtype}")
    print(f"  meta/episode_ends : {episode_ends.shape}  {episode_ends.dtype}")
    print(f"\nTotal frames written : {frame_offset}")
    print(f"Output               : {output_dir}")


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Convert a LeRobot dataset (dual FR3) to Zarr format."
        )
    )
    parser.add_argument(
        "--input_dir",  type=Path,
        default=Path("./lerobot_data"),
        help="Path to the LeRobot dataset root directory.")
    parser.add_argument(
        "--output_dir", type=Path,
        default=Path("./dataset.zarr"),
        help="Path where the Zarr dataset will be written.")
    parser.add_argument(
        "--image_size", type=int, default=224,
        help="Image resolution. Must match the recording image_size.")
    args = parser.parse_args()

    if not args.input_dir.exists():
        raise FileNotFoundError(
            f"LeRobot dataset not found: {args.input_dir}")

    convert(
        lerobot_dir=args.input_dir,
        output_dir=args.output_dir,
        image_size=args.image_size,
    )


if __name__ == "__main__":
    main()