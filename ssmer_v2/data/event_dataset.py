"""
TripleEventDatasetV2: on-the-fly event representation loader for SSMER v2.

For each sample the dataset:
  1. Reads raw events from an H5 file (shape: (N,4), columns t,x,y,p).
  2. Splits the event stream by 25 fps intervals (~40 ms each).
  3. Selects the single interval with the highest event count
     (max-event selection, matching v1 paper Fig. 5).
  4. Converts that interval on-the-fly into all three representations
     using tonic.transforms:
       frame       → tonic.transforms.ToFrame  (time_window=40000 µs)
       voxel       → tonic.transforms.ToVoxelGrid (n_time_bins=5)
       timesurface → tonic.transforms.ToTimesurface
  5. Returns (frame_tensor, voxel_tensor, ts_tensor) with shapes
       frame       : (1,  H, W)   float32
       voxel       : (5,  H, W)   float32
       timesurface : (1,  H, W)   float32

H5 column order: (t, x, y, p)  — t is column 0.

Data root is set via config.data.root_dir; sensor dimensions are read from
config.data.sensor_h / sensor_w; the JSON split file is at
config.data.split_path.

Reference: utills/h5_to_frame.py for H5 reading pattern + tonic usage.
Spec reference: §4 data/event_dataset.py; user data format answer.
"""
from __future__ import annotations

import json
import os

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from omegaconf import DictConfig

import tonic.transforms as tonic_t


# ---------------------------------------------------------------------------
# Representation conversion helpers
# ---------------------------------------------------------------------------

def _read_h5_events(h5_path: str) -> np.ndarray:
    """
    Read events from an H5 file.

    H5 structure: single dataset key "events", shape (N, 4), dtype int64.
    Column order: (t, x, y, p).

    Returns:
        Structured numpy array with fields ('x','y','t','p') as expected
        by tonic transforms.
    """
    with h5py.File(h5_path, "r") as f:
        raw = f["events"][()]              # (N, 4), int64

    dtype = np.dtype([
        ("t", np.int64),
        ("x", np.int64),
        ("y", np.int64),
        ("p", np.int64),
    ])
    events = np.zeros(raw.shape[0], dtype=dtype)
    events["t"] = raw[:, 0]   # column 0 = t
    events["x"] = raw[:, 1]
    events["y"] = raw[:, 2]
    events["p"] = raw[:, 3]
    return events


def _select_max_event_interval(
    events: np.ndarray,
    fps: float = 25.0,
) -> np.ndarray:
    """
    Split the event stream into 1/fps intervals and return the single
    interval that contains the most events (max-event selection).

    Args:
        events: Structured array with field 't' in microseconds.
        fps:    Frame rate used to define interval length.

    Returns:
        Subset of *events* for the highest-count interval.
    """
    if len(events) == 0:
        return events

    interval_us = int(1e6 / fps)          # 40 000 µs for 25 fps
    t_start = int(events["t"].min())
    t_end   = int(events["t"].max())

    best_slice = events            # fallback: entire stream
    best_count = -1

    t = t_start
    while t < t_end:
        mask = (events["t"] >= t) & (events["t"] < t + interval_us)
        count = int(mask.sum())
        if count > best_count:
            best_count = count
            best_slice = events[mask]
        t += interval_us

    return best_slice


def _to_representations(
    events: np.ndarray,
    sensor_size: tuple[int, int, int],
    t_bins: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert a structured event array into the three representations.

    Args:
        events:      Structured array with fields (t, x, y, p).
        sensor_size: (H, W, 2) tonic convention.
        t_bins:      Number of temporal bins for voxel grid.

    Returns:
        frame_arr:  (2, H, W) int16 — tonic ToFrame output (2 polarities)
        voxel_arr:  (t_bins, H, W) float32
        ts_arr:     (t_bins, H, W) float32 — ToTimesurface output, we keep
                    only the last bin
    """
    frame_arr = tonic_t.ToFrame(
        sensor_size=sensor_size,
        time_window=40_000,        # one 25-fps interval
    )(events)                      # (n_frames, 2, H, W); we take frame [0]

    voxel_arr = tonic_t.ToVoxelGrid(
        sensor_size=sensor_size,
        n_time_bins=t_bins,
    )(events)                      # (t_bins, H, W)

    ts_arr = tonic_t.ToTimesurface(
        sensor_size=sensor_size,
        dt=40_000,
        tau=100_000,
    )(events)                      # (n_frames, 2, H, W); we take frame [0]

    return frame_arr, voxel_arr, ts_arr


def _normalise(arr: np.ndarray) -> np.ndarray:
    """Min-max normalise to [0, 1], safe against zero-range arrays."""
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - lo) / (hi - lo)).astype(np.float32)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TripleEventDatasetV2(Dataset):
    """
    Loads frame / voxel / timesurface triplets from raw H5 event files.

    Each __getitem__ call:
      - Opens the H5 file for the requested video_id.
      - Selects the max-event 40 ms interval.
      - Converts to all three representations on-the-fly.
      - Returns float32 tensors ready for MultiStemHRNet.

    Args:
        cfg:   Full OmegaConf DictConfig.
        split: "train", "validation", or "test".
    """

    def __init__(self, cfg: DictConfig, split: str = "train") -> None:
        super().__init__()
        self.cfg   = cfg
        self.split = split

        data_cfg = cfg.data
        self.root_dir   = data_cfg.root_dir
        self.sensor_h   = data_cfg.sensor_h    # 512
        self.sensor_w   = data_cfg.sensor_w    # 512
        self.t_bins     = cfg.model.t_bins     # 5

        # tonic uses (H, W, num_polarities) convention
        self.sensor_size = (self.sensor_h, self.sensor_w, 2)

        # Load split file
        split_path = data_cfg.split_path
        with open(split_path, "r") as f:
            split_data = json.load(f)

        if split not in split_data:
            raise ValueError(
                f"Split '{split}' not found in {split_path}. "
                f"Available: {list(split_data.keys())}"
            )
        self.video_ids: list[str] = split_data[split]

    def __len__(self) -> int:
        return len(self.video_ids)

    def _h5_path(self, video_id: str) -> str:
        """Construct the H5 file path for a given video_id."""
        filename = f"{video_id}-ceil_10-fps_25.0-release-events.h5"
        return os.path.join(self.root_dir, filename)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            frame:       (1,  H, W) float32 tensor — polarity difference image
            voxel:       (T,  H, W) float32 tensor — normalised voxel grid
            timesurface: (1,  H, W) float32 tensor — most-recent surface
        """
        video_id = self.video_ids[idx]
        h5_path  = self._h5_path(video_id)

        # 1. Read raw events
        events = _read_h5_events(h5_path)

        # 2. Max-event interval selection
        events = _select_max_event_interval(events, fps=25.0)

        # 3. Convert to all three representations
        frame_arr, voxel_arr, ts_arr = _to_representations(
            events, self.sensor_size, self.t_bins
        )

        # 4. Post-process into single-channel float32 tensors
        #    frame: ToFrame returns (n, 2, H, W); take first frame,
        #    compute polarity difference (pos - neg) → (H, W) → (1, H, W)
        if frame_arr.ndim == 4:
            frame_arr = frame_arr[0]       # (2, H, W)
        frame_2ch  = frame_arr.astype(np.float32)
        frame_1ch  = (frame_2ch[1] - frame_2ch[0])   # pos - neg → (H, W)
        frame_norm = _normalise(frame_1ch)[np.newaxis, ...]  # (1, H, W)

        #    voxel: (T, H, W) — normalise per-bin independently
        voxel_norm = np.stack(
            [_normalise(voxel_arr[b]) for b in range(self.t_bins)], axis=0
        )   # (T, H, W)

        #    timesurface: ToTimesurface returns (n, 2, H, W); take first
        #    frame, average over polarities → (H, W) → (1, H, W)
        if ts_arr.ndim == 4:
            ts_arr = ts_arr[0]             # (2, H, W)
        ts_1ch  = ts_arr.astype(np.float32).mean(axis=0)   # (H, W)
        ts_norm = _normalise(ts_1ch)[np.newaxis, ...]       # (1, H, W)

        frame_t = torch.from_numpy(frame_norm)
        voxel_t = torch.from_numpy(voxel_norm)
        ts_t    = torch.from_numpy(ts_norm)

        return frame_t, voxel_t, ts_t
