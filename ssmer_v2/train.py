"""
SSMER v2 training entry point.

Usage:
    # Single GPU
    python -m ssmer_v2.train --config ssmer_v2/config/ablation_A4_lambda05.yaml

    # Multi-GPU (torchrun)
    torchrun --nproc_per_node=4 -m ssmer_v2.train \\
        --config ssmer_v2/config/ablation_A4_lambda05.yaml \\
        --override training.batch_size=32

    # Resume
    python -m ssmer_v2.train --config ssmer_v2/config/default.yaml \\
        --override checkpoint.resume=/path/to/checkpoint.pth

Config loading:
    The specified YAML is merged on top of ssmer_v2/config/default.yaml via
    OmegaConf.  Any key can be overridden on the command line with
    --override key=value (repeatable).

Checkpoints:
    Saved to  {checkpoint.save_dir}/{experiment_name}/checkpoint_{epoch:04d}.pth
    Backbone-only weights saved alongside:
              {checkpoint.save_dir}/{experiment_name}/backbone_{epoch:04d}.pth

Stage 2 handoff:
    After pre-training, pass the backbone checkpoint to Stage 2:
        model.hrnet_encoder.load_state_dict(
            torch.load("backbone_0200.pth"), strict=False
        )
"""
from __future__ import annotations

import argparse
import logging
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from omegaconf import OmegaConf

from ssmer_v2.config import load_config
from ssmer_v2.models.ssmer_v2 import SSMERv2
from ssmer_v2.losses.combined_loss import CombinedLoss
from ssmer_v2.data.event_dataset import TripleEventDatasetV2


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SSMER v2 pre-training")
    parser.add_argument(
        "--config", default="default",
        help="Path to experiment YAML or 'default'.",
    )
    parser.add_argument(
        "--override", action="append", default=[],
        metavar="KEY=VALUE",
        help="Override a config key (repeatable).  E.g. --override training.lr=0.01",
    )
    parser.add_argument(
        "--world-size", type=int, default=1,
        help="Total number of distributed processes.",
    )
    parser.add_argument(
        "--rank", type=int, default=0,
        help="Rank of this process (used when launched without torchrun).",
    )
    parser.add_argument(
        "--dist-url", default="env://",
        help="URL for distributed init.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class AverageMeter:
    """Running mean of a scalar metric."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.reset()

    def reset(self) -> None:
        self.val = self.avg = self.sum = self.count = 0.0

    def update(self, val: float, n: int = 1) -> None:
        self.val   = val
        self.sum  += val * n
        self.count += n
        self.avg   = self.sum / self.count

    def __str__(self) -> str:
        return f"{self.name} {self.val:.4f} ({self.avg:.4f})"


def build_optimizer(model: nn.Module, cfg) -> torch.optim.Optimizer:
    t = cfg.training
    if t.optimizer == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=t.lr,
            momentum=t.momentum,
            weight_decay=t.weight_decay,
        )
    if t.optimizer == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=t.lr,
            weight_decay=t.weight_decay,
        )
    raise ValueError(f"Unknown optimizer: {t.optimizer!r}")


def build_scheduler(optimizer: torch.optim.Optimizer, cfg):
    """
    Build the LR scheduler.  If cfg.training.warmup_epochs > 0, the requested
    main scheduler (cosine / step) is wrapped in a SequentialLR with a leading
    LinearLR warmup phase.  scheduler.step() is called once per epoch.
    """
    t = cfg.training
    warmup_epochs = int(t.get("warmup_epochs", 0) or 0)
    main_epochs = max(1, t.epochs - warmup_epochs)

    if t.lr_scheduler == "cosine":
        main = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=main_epochs, eta_min=0.0
        )
    elif t.lr_scheduler == "step":
        main = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=max(1, main_epochs // 3), gamma=0.1
        )
    else:
        raise ValueError(f"Unknown lr_scheduler: {t.lr_scheduler!r}")

    if warmup_epochs <= 0:
        return main

    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1e-3,
        end_factor=1.0,
        total_iters=warmup_epochs,
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, main],
        milestones=[warmup_epochs],
    )


def _atomic_torch_save(obj: object, path: Path) -> None:
    """torch.save via tmp file + os.replace so a killed job never leaves a
    half-written .pth that crashes the next --resume."""
    tmp = Path(str(path) + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def save_checkpoint(
    state: dict,
    save_dir: Path,
    epoch: int,
    model: SSMERv2,
) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = save_dir / f"checkpoint_{epoch:04d}.pth"
    _atomic_torch_save(state, ckpt_path)

    # Backbone-only checkpoint for Stage 2
    backbone_path = save_dir / f"backbone_{epoch:04d}.pth"
    # Unwrap DDP if needed
    raw_model = model.module if isinstance(model, DDP) else model
    _atomic_torch_save(raw_model.backbone_state_dict(), backbone_path)

    logging.info(f"Saved checkpoint  → {ckpt_path}")
    logging.info(f"Saved backbone    → {backbone_path}")


# ---------------------------------------------------------------------------
# Training loop (runs in each worker process)
# ---------------------------------------------------------------------------

def train_one_epoch(
    loader: DataLoader,
    model: nn.Module,
    criterion: CombinedLoss,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    cfg,
    gpu: int | None,
    is_main: bool,
    scaler: torch.cuda.amp.GradScaler | None = None,
) -> dict[str, float]:
    model.train()
    meters: dict[str, AverageMeter] = {}

    use_amp = scaler is not None
    clip_grad = float(cfg.training.get("clip_grad_norm", 0.0) or 0.0)

    for step, (frame, voxel, timesurface) in enumerate(loader):
        if gpu is not None:
            frame       = frame.cuda(gpu, non_blocking=True)
            voxel       = voxel.cuda(gpu, non_blocking=True)
            timesurface = timesurface.cuda(gpu, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.cuda.amp.autocast():
                output = model(frame, voxel, timesurface)
                total_loss, log_dict = criterion(output)
            scaler.scale(total_loss).backward()
            if clip_grad > 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            scaler.step(optimizer)
            scaler.update()
        else:
            output = model(frame, voxel, timesurface)
            total_loss, log_dict = criterion(output)
            total_loss.backward()
            if clip_grad > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()

        # Update meters
        bs = frame.size(0)
        for k, v in log_dict.items():
            if k not in meters:
                meters[k] = AverageMeter(k)
            meters[k].update(v.item(), bs)

        if is_main and step % cfg.logging.print_freq == 0:
            parts = [f"Epoch [{epoch}/{cfg.training.epochs}]",
                     f"Step [{step}/{len(loader)}]"]
            parts += [str(m) for m in meters.values()]
            logging.info("  ".join(parts))

    return {k: m.avg for k, m in meters.items()}


# ---------------------------------------------------------------------------
# Main worker
# ---------------------------------------------------------------------------

def _set_seed(seed: int, rank: int = 0) -> None:
    """Seed python / numpy / torch with rank-offset for distributed runs."""
    s = int(seed) + int(rank)
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def main_worker(gpu: int | None, cfg, args: argparse.Namespace) -> None:
    # --- Distributed detection ---
    # Source of truth: torch.distributed itself, not cfg (cfg never carries
    # world_size, and reading it from cfg always evaluated to False, silently
    # disabling DDP / DistributedSampler under torchrun).
    is_distributed = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if is_distributed else 0
    is_main = rank == 0

    # --- Logging ---
    # Configure on every rank — main at INFO, others at WARNING — so non-main
    # ranks still surface real errors instead of silently swallowing them.
    logging.basicConfig(
        level=logging.INFO if is_main else logging.WARNING,
        format=f"%(asctime)s %(levelname)s [rank{rank}]  %(message)s",
    )

    # --- Seed ---
    _set_seed(cfg.training.get("seed", 42), rank=rank)

    # --- Model ---
    pretrained = cfg.checkpoint.get("pretrained_hrnet", None)
    model = SSMERv2(cfg, pretrained_hrnet=pretrained)

    if gpu is not None:
        torch.cuda.set_device(gpu)
        model = model.cuda(gpu)

    if is_distributed:
        model = DDP(model, device_ids=[gpu] if gpu is not None else None)

    # --- Loss ---
    criterion = CombinedLoss(cfg)
    if gpu is not None:
        criterion = criterion.cuda(gpu)

    # --- Data ---
    train_dataset = TripleEventDatasetV2(cfg, split="train")
    sampler = DistributedSampler(train_dataset) if is_distributed else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=cfg.training.num_workers,
        pin_memory=cfg.training.pin_memory,
        drop_last=True,
    )

    # --- Optimizer & scheduler ---
    optimizer  = build_optimizer(model, cfg)
    scheduler  = build_scheduler(optimizer, cfg)

    # --- AMP ---
    use_amp = bool(cfg.training.get("use_amp", False)) and torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    if is_main:
        logging.info(f"AMP enabled: {use_amp}")

    # --- Resume ---
    start_epoch = 1
    resume_path = cfg.checkpoint.get("resume", None)
    if resume_path:
        if not os.path.isfile(resume_path):
            raise FileNotFoundError(
                f"checkpoint.resume is set but file does not exist: {resume_path}"
            )
        ckpt = torch.load(resume_path, map_location="cpu")
        raw_model = model.module if isinstance(model, DDP) else model
        raw_model.load_state_dict(ckpt["state_dict"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        if scaler is not None and ckpt.get("scaler") is not None:
            scaler.load_state_dict(ckpt["scaler"])
        # Restore RNG state so resumed runs are bit-identical to non-resumed.
        rng = ckpt.get("rng")
        if rng is not None:
            torch.set_rng_state(rng["torch"])
            if rng.get("torch_cuda") is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(rng["torch_cuda"])
            random.setstate(rng["python"])
            np.random.set_state(rng["numpy"])
        # Restore sequential masking counter (only matters for sequential strategy)
        seq_counter = ckpt.get("sequential_counter")
        if seq_counter is not None:
            raw_model._sequential_counter = int(seq_counter)
        start_epoch = ckpt["epoch"] + 1
        if is_main:
            logging.info(f"Resumed from epoch {ckpt['epoch']}  ({resume_path})")

    save_dir = Path(cfg.checkpoint.save_dir) / Path(args.config).stem

    # --- Training loop ---
    for epoch in range(start_epoch, cfg.training.epochs + 1):
        if sampler is not None:
            sampler.set_epoch(epoch)

        log = train_one_epoch(
            train_loader, model, criterion, optimizer,
            epoch, cfg, gpu, is_main, scaler=scaler,
        )
        scheduler.step()

        if is_main:
            logging.info(
                f"Epoch {epoch:4d} complete — "
                + "  ".join(f"{k}={v:.4f}" for k, v in log.items())
            )

        if is_main and epoch % cfg.checkpoint.save_freq == 0:
            raw_model = model.module if isinstance(model, DDP) else model
            save_checkpoint(
                state={
                    "epoch":       epoch,
                    "state_dict":  raw_model.state_dict(),
                    "optimizer":   optimizer.state_dict(),
                    "scheduler":   scheduler.state_dict(),
                    "scaler":      scaler.state_dict() if scaler is not None else None,
                    "rng": {
                        "torch":      torch.get_rng_state(),
                        "torch_cuda": (
                            torch.cuda.get_rng_state_all()
                            if torch.cuda.is_available() else None
                        ),
                        "python":     random.getstate(),
                        "numpy":      np.random.get_state(),
                    },
                    "sequential_counter": raw_model._sequential_counter,
                    "config":      OmegaConf.to_container(cfg, resolve=True),
                },
                save_dir=save_dir,
                epoch=epoch,
                model=raw_model,
            )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args  = parse_args()
    cfg   = load_config(args.config, args.override)

    world_size = int(os.environ.get("WORLD_SIZE", args.world_size))

    if world_size > 1:
        # Launched via torchrun — each process calls main_worker directly.
        dist.init_process_group(
            backend="nccl",
            init_method=args.dist_url,
            world_size=world_size,
            rank=int(os.environ.get("RANK", args.rank)),
        )
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        main_worker(local_rank, cfg, args)
        dist.destroy_process_group()
    else:
        gpu = 0 if torch.cuda.is_available() else None
        main_worker(gpu, cfg, args)


if __name__ == "__main__":
    main()
