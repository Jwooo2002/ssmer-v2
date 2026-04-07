import torch
import torch.nn as nn


class TemporalEmbedding(nn.Module):
    """
    Learnable per-bin positional bias added to the voxel input tensor
    BEFORE the first convolution of stem_voxel.

    The bias has shape (1, T_bins, 1, 1) and broadcasts over (B, T, H, W),
    so it encodes temporal ordering without assuming any fixed spatial size.

    Spec reference: §2.1 step 3 / §2.2 "Temporal Encoding for Voxel".
    Ablation A8 disables this module (voxel_temporal_encoding: false).
    """

    def __init__(self, t_bins: int) -> None:
        super().__init__()
        self.t_bins = t_bins
        self.bias = nn.Parameter(torch.zeros(1, t_bins, 1, 1))

    def extra_repr(self) -> str:
        return f"t_bins={self.t_bins}"

    def forward(self, voxel: torch.Tensor) -> torch.Tensor:
        """
        Args:
            voxel: (B, T_bins, H, W) raw voxel grid tensor

        Returns:
            (B, T_bins, H, W) with per-bin bias added
        """
        return voxel + self.bias
