"""
TypeEmbedding: learnable per-representation type tokens.

Adds a fixed learnable vector to every token in a representation's sequence,
allowing the Transformer encoder/decoder to distinguish frame, voxel, and
timesurface tokens even after they have been projected to the same embed_dim.

Type indices:
    0 → frame
    1 → voxel
    2 → timesurface

Spec reference: §2.1 step 3.
"""
import torch
import torch.nn as nn


class TypeEmbedding(nn.Module):
    """
    Three learnable type embeddings, one per event representation.

    Args:
        num_types: Number of representation types (default 3).
        embed_dim: Token embedding dimension (default 256).
    """

    # Public constants so callers can use symbolic names instead of raw ints.
    FRAME = 0
    VOXEL = 1
    TIMESURFACE = 2

    def __init__(self, num_types: int = 3, embed_dim: int = 256) -> None:
        super().__init__()
        self.embed = nn.Embedding(num_types, embed_dim)
        nn.init.trunc_normal_(self.embed.weight, std=0.02)

    def forward(self, tokens: torch.Tensor, type_id: int) -> torch.Tensor:
        """
        Add the type embedding for *type_id* to every token in *tokens*.

        Args:
            tokens:  (B, N, embed_dim)
            type_id: Integer index (0, 1, or 2).

        Returns:
            (B, N, embed_dim) — tokens with type embedding broadcast-added.
        """
        device = tokens.device
        idx = torch.tensor(type_id, dtype=torch.long, device=device)
        type_emb = self.embed(idx)                  # (embed_dim,)
        return tokens + type_emb.unsqueeze(0).unsqueeze(0)   # broadcast (1,1,D)
