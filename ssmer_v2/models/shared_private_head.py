"""
SharedPrivateHead: decomposes per-representation encoder tokens into a
shared subspace (cross-modal content) and a modality-specific private
subspace.

Applied AFTER the Transformer encoder so that cross-modal interaction has
already occurred before decomposition.  The orthogonality loss then pushes
shared and private subspaces apart.

Used only when config.model.use_shared_private = True  (ablation A6).

Spec reference: §2.2 "Shared-Private head location"; §7 shared_private notes.
"""
import torch
import torch.nn as nn


class SharedPrivateHead(nn.Module):
    """
    Per-representation projection into shared and private subspaces.

    A single shared projector is applied identically to all representations;
    separate private projectors per representation allow modality-specific
    features to be captured without contaminating the shared component.

    Args:
        embed_dim:   Input token dimension (output of Transformer encoder).
        shared_dim:  Dimension of the shared subspace (default 128).
        private_dim: Dimension of each private subspace (default 128).
    """

    # Canonical representation names — used to retrieve the right private head.
    REPS = ("frame", "voxel", "timesurface")

    def __init__(
        self,
        embed_dim: int = 256,
        shared_dim: int = 128,
        private_dim: int = 128,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.shared_dim = shared_dim
        self.private_dim = private_dim

        # One shared projector for all modalities
        self.shared_proj = nn.Sequential(
            nn.Linear(embed_dim, shared_dim),
            nn.LayerNorm(shared_dim),
            nn.ReLU(inplace=True),
        )

        # One private projector per modality
        self.private_projs = nn.ModuleDict({
            rep: nn.Sequential(
                nn.Linear(embed_dim, private_dim),
                nn.LayerNorm(private_dim),
                nn.ReLU(inplace=True),
            )
            for rep in self.REPS
        })

    def forward(
        self,
        tokens: torch.Tensor,
        rep_name: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Project token sequence into shared + private components.

        Args:
            tokens:   (B, N, embed_dim) — encoder output tokens for one
                      representation.
            rep_name: One of "frame", "voxel", "timesurface".

        Returns:
            shared:  (B, N, shared_dim)
            private: (B, N, private_dim)
        """
        if rep_name not in self.private_projs:
            raise ValueError(
                f"rep_name must be one of {self.REPS}, got {rep_name!r}"
            )
        shared = self.shared_proj(tokens)
        private = self.private_projs[rep_name](tokens)
        return shared, private
