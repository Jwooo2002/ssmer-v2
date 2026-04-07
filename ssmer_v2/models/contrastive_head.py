"""
ContrastiveHead: SimSiam projector + predictor adapted for token sequences.

Ported from simsiam/builder.py:HRNetSimSiam.  The key difference from v1 is
that the input is a token sequence (B, N, D) rather than a 1D vector (B, D).
Global average pooling over N collapses the sequence to a single vector before
the MLP, which:
  1. Preserves exact v1 loss semantics (same BN + MLP structure).
  2. Makes ablation A1 (use_recon=False, use_contrastive=True) reproduce the
     v1 SimSiam result on the same data.

Spec reference: §2.1 step 9 (contrastive path); §7 A1 baseline note.
"""
import torch
import torch.nn as nn


class ContrastiveHead(nn.Module):
    """
    SimSiam-style projector + predictor that operates on token sequences.

    Args:
        embed_dim:  Input token dimension, D (default 256).
        proj_dim:   Projector output dimension (default 256).
        pred_dim:   Predictor bottleneck dimension (default 128).
    """

    def __init__(
        self,
        embed_dim: int = 256,
        proj_dim: int = 256,
        pred_dim: int = 128,
    ) -> None:
        super().__init__()

        # 3-layer projector — mirrors HRNetSimSiam.encoder.head in v1.
        # BN + ReLU → Linear + BN + ReLU → Linear + BN(affine=False)
        self.projector = nn.Sequential(
            nn.Linear(embed_dim, embed_dim, bias=False),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim, bias=False),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, proj_dim, bias=False),
            nn.BatchNorm1d(proj_dim, affine=False),
        )
        # Disable bias on the last linear (followed by BN), matching v1 hack.
        self.projector[-2].bias = None   # type: ignore[assignment]

        # 2-layer predictor — mirrors HRNetSimSiam.predictor in v1.
        self.predictor = nn.Sequential(
            nn.Linear(proj_dim, pred_dim, bias=False),
            nn.BatchNorm1d(pred_dim),
            nn.ReLU(inplace=True),
            nn.Linear(pred_dim, proj_dim),
        )

    def forward(
        self,
        tokens_a: torch.Tensor,
        tokens_b: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            tokens_a: (B, N, embed_dim) token sequence for representation A
            tokens_b: (B, N, embed_dim) token sequence for representation B

        Returns:
            (p1, p2, z1, z2) where:
                z1, z2 are stop-gradient projector outputs — targets
                p1, p2 are predictor outputs — predictions
        """
        # Pool over spatial tokens → (B, embed_dim)
        vec_a = tokens_a.mean(dim=1)
        vec_b = tokens_b.mean(dim=1)

        z1 = self.projector(vec_a)   # (B, proj_dim)
        z2 = self.projector(vec_b)

        p1 = self.predictor(z1)      # (B, proj_dim)
        p2 = self.predictor(z2)

        # Stop-gradient on targets — same as v1 SimSiam
        return p1, p2, z1.detach(), z2.detach()
