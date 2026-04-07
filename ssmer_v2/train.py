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
import time
from pathlib import Path

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


def build_scheduler(optimizer: torch.optim.Optimizer, cfg) -> torch.optim.lr_scheduler._LRScheduler:
    t = cfg.training
    if t.lr_scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=t.epochs, eta_min=0.0
        )
    if t.lr_scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=t.epochs // 3, gamma=0.1
        )
    raise ValueError(f"Unknown lr_scheduler: {t.lr_scheduler!r}")


def save_checkpoint(
    state: dict,
    save_dir: Path,
    epoch: int,
    model: SSMERv2,
) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = save_dir / f"checkpoint_{epoch:04d}.pth"
    torch.save(state, ckpt_path)

    # Backbone-only checkpoint for Stage 2
    backbone_path = save_dir / f"backbone_{epoch:04d}.pth"
    # Unwrap DDP if needed
    raw_model = model.module if isinstance(model, DDP) else model
    torch.save(raw_model.backbone_state_dict(), backbone_path)

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
) -> dict[str, float]:
    model.train()
    meters: dict[str, AverageMeter] = {}

    for step, (frame, voxel, timesurface) in enumerate(loader):
        if gpu is not None:
            frame       = frame.cuda(gpu, non_blocking=True)
            voxel       = voxel.cuda(gpu, non_blocking=True)
            timesurface = timesurface.cuda(gpu, non_blocking=True)

        output = model(frame, voxel, timesurface)
        total_loss, log_dict = criterion(output)

        optimizer.zero_grad()
        total_loss.backward()
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

def main_worker(gpu: int | None, cfg, args: argparse.Namespace) -> None:
    is_distributed = cfg.training.get("world_size", 1) > 1
    is_main = (not is_distributed) or (dist.get_rank() == 0)

    # --- Logging ---
    if is_main:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s  %(message)s",
        )

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
            epoch, cfg, gpu, is_main,
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
