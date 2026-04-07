"""
FeatureToToken: converts a spatial feature map into a sequence of tokens.

(B, C, H', W')  →  flatten spatial  →  (B, N, C)  →  Linear  →  (B, N, D)

where N = H' * W' and D = embed_dim.

A separate FeatureToToken instance is created per representation so the model
can learn modality-specific projection weights.

Spec reference: §2.1 step 2; §7 "embed_dim and HRNet output".
"""
import torch
import torch.nn as nn


class FeatureToToken(nn.Module):
    """
    Flattens an HRNet feature map to a token sequence and projects to
    embed_dim via a single linear layer.

    Args:
        in_channels: Channel depth of the input feature map.
                     For the default W18 HRNet this is 270.
        embed_dim:   Target token dimension (default 256).
    """

    def __init__(self, in_channels: int = 270, embed_dim: int = 256) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.proj = nn.Linear(in_channels, embed_dim)

        nn.init.trunc_normal_(self.proj.weight, std=0.02)
        nn.init.zeros_(self.proj.bias)

    def extra_repr(self) -> str:
        return f"in_channels={self.in_channels}, embed_dim={self.embed_dim}"

    def forward(self, feat_map: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat_map: (B, C, H', W')

        Returns:
            tokens: (B, N, embed_dim) where N = H' * W'
        """
        # (B, C, H', W') → (B, N, C)
        B, C, H, W = feat_map.shape
        tokens = feat_map.flatten(2).transpose(1, 2)   # (B, N, C)
        tokens = self.proj(tokens)                      # (B, N, D)
        return tokens
