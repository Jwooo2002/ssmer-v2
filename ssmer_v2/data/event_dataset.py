"""
TripleEventDatasetV2: event representation loader for SSMER v2.

For each sample the dataset:
  1. Reads raw events from an H5 file (shape: (N,4), columns t,x,y,p).
  2. Splits the event stream by 25 fps intervals (~40 ms each).
  3. Selects the single interval with the highest event count
     (max-event selection, matching v1 paper Fig. 5).
  4. Converts that interval into all three representations via tonic.transforms:
       frame       → tonic.transforms.ToFrame  (time_window=40000 µs)
       voxel       → tonic.transforms.ToVoxelGrid (n_time_bins=5)
       timesurface → tonic.transforms.ToTimesurface
  5. Returns (frame_tensor, voxel_tensor, ts_tensor) with shapes
       frame       : (1,  H, W)   float32
       voxel       : (5,  H, W)   float32
       timesurface : (1,  H, W)   float32

Disk cache (config.data.use_cache = true):
  On the first call for a given video_id the converted arrays are saved as
  .npy files under:
      {cache_dir}/{split}/{video_id}_frame.npy
      {cache_dir}/{split}/{video_id}_voxel.npy
      {cache_dir}/{split}/{video_id}_ts.npy
  Subsequent calls load from cache — skipping H5 read and tonic conversion.
  Set use_cache: false to force re-conversion (e.g. after changing t_bins).

H5 column order: (t, x, y, p)  — t is column 0.
"""
from __future__ import annotations

import json
import os

import h5py
import numpy as np
import torch
import torch.nn.functional as F
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
        Structured numpy array with fields ('t','x','y','p') as expected
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
    interval with the most events (max-event selection).
    """
    if len(events) == 0:
        return events

    interval_us = int(1e6 / fps)          # 40 000 µs for 25 fps
    t_start = int(events["t"].min())
    t_end   = int(events["t"].max())

    best_slice = events
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
    """Convert structured event array → (frame_arr, voxel_arr, ts_arr)."""
    frame_arr = tonic_t.ToFrame(
        sensor_size=sensor_size,
        time_window=40_000,
    )(events)                      # (n_frames, 2, H, W)

    voxel_arr = tonic_t.ToVoxelGrid(
        sensor_size=sensor_size,
        n_time_bins=t_bins,
    )(events)                      # (t_bins, H, W)

    ts_arr = tonic_t.ToTimesurface(
        sensor_size=sensor_size,
        dt=40_000,
        tau=100_000,
    )(events)                      # (n_frames, 2, H, W)

    return frame_arr, voxel_arr, ts_arr


def _normalise(arr: np.ndarray) -> np.ndarray:
    """Min-max normalise to [0, 1], safe against zero-range arrays."""
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - lo) / (hi - lo)).astype(np.float32)


def _postprocess(
    frame_arr: np.ndarray,
    voxel_arr: np.ndarray,
    ts_arr: np.ndarray,
    t_bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert raw tonic outputs to normalised float32 arrays ready for tensors.

    Returns:
        frame_norm : (1, H, W)   float32
        voxel_norm : (T, H, W)   float32
        ts_norm    : (1, H, W)   float32
    """
    # frame: (n, 2, H, W) → take [0] → polarity diff (pos - neg) → (1, H, W)
    if frame_arr.ndim == 4:
        frame_arr = frame_arr[0]
    frame_2ch  = frame_arr.astype(np.float32)
    frame_1ch  = frame_2ch[1] - frame_2ch[0]
    frame_norm = _normalise(frame_1ch)[np.newaxis, ...]

    # voxel: (T, H, W) — normalise per-bin
    voxel_norm = np.stack(
        [_normalise(voxel_arr[b]) for b in range(t_bins)], axis=0
    )

    # timesurface: (n, 2, H, W) → take [0] → mean over polarities → (1, H, W)
    if ts_arr.ndim == 4:
        ts_arr = ts_arr[0]
    ts_1ch  = ts_arr.astype(np.float32).mean(axis=0)
    ts_norm = _normalise(ts_1ch)[np.newaxis, ...]

    return frame_norm, voxel_norm, ts_norm


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TripleEventDatasetV2(Dataset):
    """
    Loads frame / voxel / timesurface triplets from raw H5 event files,
    with optional disk caching under ssmer_v2/cache/.

    Cache layout:
        {cache_dir}/{split}/{video_id}_frame.npy   (1, H, W)  float32
        {cache_dir}/{split}/{video_id}_voxel.npy   (T, H, W)  float32
        {cache_dir}/{split}/{video_id}_ts.npy      (1, H, W)  float32

    Args:
        cfg:   Full OmegaConf DictConfig.
        split: "train", "validation", or "test".
    """

    def __init__(self, cfg: DictConfig, split: str = "train") -> None:
        super().__init__()
        self.split = split

        data_cfg = cfg.data
        self.root_dir    = data_cfg.root_dir
        self.sensor_h    = data_cfg.sensor_h
        self.sensor_w    = data_cfg.sensor_w
        self.t_bins      = cfg.model.t_bins
        self.sensor_size = (self.sensor_h, self.sensor_w, 2)

        self.input_size = data_cfg.input_size   # resize target (e.g. 224)

        self.use_cache  = data_cfg.use_cache
        self.cache_dir  = os.path.join(data_cfg.cache_dir, split)
        if self.use_cache:
            os.makedirs(self.cache_dir, exist_ok=True)

        # Load split file
        with open(data_cfg.split_path, "r") as f:
            split_data = json.load(f)

        if split not in split_data:
            raise ValueError(
                f"Split '{split}' not found in {data_cfg.split_path}. "
                f"Available: {list(split_data.keys())}"
            )
        self.video_ids: list[str] = split_data[split]

    def __len__(self) -> int:
        return len(self.video_ids)

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _cache_paths(self, video_id: str) -> tuple[str, str, str]:
        # The voxel filename embeds t_bins so that changing model.t_bins
        # automatically invalidates stale (T_old, H, W) caches instead of
        # silently loading them and feeding wrong-shape arrays into the
        # stem_voxel conv (which would crash with a cryptic shape error,
        # or worse, silently mismatch if the stem was rebuilt for the new T).
        base = os.path.join(self.cache_dir, video_id)
        return (
            f"{base}_frame.npy",
            f"{base}_voxel_t{self.t_bins}.npy",
            f"{base}_ts.npy",
        )

    def _cache_exists(self, video_id: str) -> bool:
        return all(os.path.isfile(p) for p in self._cache_paths(video_id))

    def _save_cache(
        self,
        video_id: str,
        frame_norm: np.ndarray,
        voxel_norm: np.ndarray,
        ts_norm: np.ndarray,
    ) -> None:
        fp, vp, tp = self._cache_paths(video_id)
        np.save(fp, frame_norm)
        np.save(vp, voxel_norm)
        np.save(tp, ts_norm)

    def _load_cache(
        self, video_id: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        fp, vp, tp = self._cache_paths(video_id)
        return np.load(fp), np.load(vp), np.load(tp)

    # ------------------------------------------------------------------
    # Core conversion
    # ------------------------------------------------------------------

    def _convert(
        self, video_id: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Read H5 → max-event selection → tonic conversion → normalise."""
        h5_path = os.path.join(
            self.root_dir,
            f"{video_id}-ceil_10-fps_25.0-release-events.h5",
        )
        events = _read_h5_events(h5_path)
        events = _select_max_event_interval(events, fps=25.0)
        frame_arr, voxel_arr, ts_arr = _to_representations(
            events, self.sensor_size, self.t_bins
        )
        return _postprocess(frame_arr, voxel_arr, ts_arr, self.t_bins)

    # ------------------------------------------------------------------
    # __getitem__
    # ------------------------------------------------------------------

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            frame:       (1, H, W) float32
            voxel:       (T, H, W) float32
            timesurface: (1, H, W) float32
        """
        video_id = self.video_ids[idx]

        if self.use_cache and self._cache_exists(video_id):
            frame_norm, voxel_norm, ts_norm = self._load_cache(video_id)
        else:
            frame_norm, voxel_norm, ts_norm = self._convert(video_id)
            if self.use_cache:
                self._save_cache(video_id, frame_norm, voxel_norm, ts_norm)

        frame_t = torch.from_numpy(frame_norm)
        voxel_t = torch.from_numpy(voxel_norm)
        ts_t    = torch.from_numpy(ts_norm)

        # Resize all representations to (input_size, input_size) so that
        # the HRNet stem produces feature maps of known size
        # (input_size/4)^2, matching the pos_embed and n_tokens set at
        # model construction time.
        # Without this, sensor_h=512 → N=16384 tokens but pos_embed only
        # has max_tokens=3136 → broadcast crash on first batch.
        s = self.input_size
        frame_t = F.interpolate(frame_t.unsqueeze(0), size=(s, s), mode="bilinear", align_corners=False).squeeze(0)
        voxel_t = F.interpolate(voxel_t.unsqueeze(0), size=(s, s), mode="bilinear", align_corners=False).squeeze(0)
        ts_t    = F.interpolate(ts_t.unsqueeze(0),    size=(s, s), mode="bilinear", align_corners=False).squeeze(0)

        return frame_t, voxel_t, ts_t
