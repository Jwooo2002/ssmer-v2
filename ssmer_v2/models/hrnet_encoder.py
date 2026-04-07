"""
Multi-stem HRNet encoder for SSMER v2.

Three separate stem layers (one per event representation) feed into a single
shared HRNet body.  Only HRNet body weights are transferred to Stage 2 after
pre-training; everything else in this file is a disposable pretext component.

Spec reference: §2.1 step 1 / §2.2 "Why HRNet + Transformer hybrid".
"""
import os
import sys

import torch
import torch.nn as nn

# Allow imports from the project root regardless of CWD
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from models.hrnet.hrnet import HighResolutionNet
from models.hrnet.config import cfg as _default_hrnet_cfg
from ssmer_v2.models.temporal_embedding import TemporalEmbedding


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class StemBlock(nn.Module):
    """
    Two-layer stride-2 convolutional stem that mirrors the original HRNet stem
    but accepts an arbitrary number of input channels.

    Input:  (B, in_channels, H, W)
    Output: (B, 64, H/4, W/4)  — ready for HRNet.forward_from_stem()
    """

    def __init__(self, in_channels: int, out_channels: int = 64) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3,
                      stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels, momentum=0.01),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3,
                      stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels, momentum=0.01),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stem(x)


# ---------------------------------------------------------------------------
# Main encoder
# ---------------------------------------------------------------------------

class MultiStemHRNet(nn.Module):
    """
    HRNet backbone with three independent stem layers (frame / voxel /
    timesurface) and a single shared body.

    The shared body is a full HighResolutionNet instance.  Its own conv1/conv2
    stem layers are allocated by __init__ but never called during v2 inference;
    only layer1 through stage4 (accessed via forward_from_stem) are used.

    Args:
        t_bins:   Number of temporal bins for the voxel representation
                  (stem_voxel in_channels).  Must equal config.model.t_bins.
        use_temporal_encoding: If True, add a learnable (1,T,1,1) bias to the
                  voxel input before stem_voxel.  Set False for ablation A8.
        hrnet_cfg: YACS config object for HighResolutionNet.  Defaults to the
                  v1 W18 config (Stage4 channels [18,36,72,144], sum=270).
        pretrained: Optional path to a v1 HRNet checkpoint.  Only the shared
                  body weights (layer1, stage2-4) are loaded; head weights and
                  the new stem weights are ignored.
    """

    # Feature map channels output by the shared body: sum of Stage4 channels
    # [18, 36, 72, 144] = 270 for the default W18 config.
    BODY_OUT_CHANNELS: int = 270

    def __init__(
        self,
        t_bins: int = 5,
        use_temporal_encoding: bool = True,
        hrnet_cfg=None,
        pretrained: str | None = None,
    ) -> None:
        super().__init__()

        if hrnet_cfg is None:
            hrnet_cfg = _default_hrnet_cfg

        # --- Stems (separate, trainable per representation) -----------------
        self.stem_frame = StemBlock(in_channels=1)
        self.stem_voxel = StemBlock(in_channels=t_bins)
        self.stem_ts = StemBlock(in_channels=1)

        # --- Optional temporal encoding for voxel ---------------------------
        self.use_temporal_encoding = use_temporal_encoding
        if use_temporal_encoding:
            self.temporal_embedding = TemporalEmbedding(t_bins)

        # --- Shared HRNet body ----------------------------------------------
        # num_classes is unused (head is never called), but HighResolutionNet
        # requires it; pass BODY_OUT_CHANNELS as a harmless placeholder.
        self.shared_body = HighResolutionNet(
            hrnet_cfg, num_classes=self.BODY_OUT_CHANNELS
        )

        # --- Initialise weights ---------------------------------------------
        self._init_stems()
        if pretrained is not None:
            self._load_pretrained_body(pretrained)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _init_stems(self) -> None:
        """Kaiming-normal init for all three stems."""
        for stem in (self.stem_frame, self.stem_voxel, self.stem_ts):
            for m in stem.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(
                        m.weight, mode="fan_out", nonlinearity="relu"
                    )
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def _load_pretrained_body(self, pretrained: str) -> None:
        """
        Load a v1 HRNet checkpoint into shared_body, skipping the head and
        the original conv1/conv2 stem (which belong to v1's single 3-channel
        input stem, not our per-representation stems).
        """
        ckpt = torch.load(pretrained, map_location="cpu")
        # v1 checkpoints may be wrapped under 'state_dict'
        state = ckpt.get("state_dict", ckpt)

        body_dict = self.shared_body.state_dict()
        skip_prefixes = ("head.", "conv1.", "bn1.", "conv2.", "bn2.")
        filtered = {
            k: v for k, v in state.items()
            if k in body_dict and not any(k.startswith(p) for p in skip_prefixes)
        }
        body_dict.update(filtered)
        self.shared_body.load_state_dict(body_dict, strict=False)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        frame: torch.Tensor,
        voxel: torch.Tensor,
        timesurface: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract feature maps for all three event representations.

        Args:
            frame:       (B, 1, H, W)        accumulated event frame
            voxel:       (B, T_bins, H, W)   temporal voxel grid
            timesurface: (B, 1, H, W)        per-pixel recency surface

        Returns:
            feat_f, feat_v, feat_ts — each (B, 270, H/4, W/4)
        """
        # Temporal encoding is applied to raw voxel tensor BEFORE stem conv
        if self.use_temporal_encoding:
            voxel = self.temporal_embedding(voxel)

        feat_f = self.shared_body.forward_from_stem(self.stem_frame(frame))
        feat_v = self.shared_body.forward_from_stem(self.stem_voxel(voxel))
        feat_ts = self.shared_body.forward_from_stem(self.stem_ts(timesurface))

        return feat_f, feat_v, feat_ts
