"""
CrossModalTransformerEncoder: 4-layer Transformer encoder.

Processes the concatenated token sequences from the two *visible*
representations (i.e., the pair that was not masked).  Full self-attention
over the joint (B, 2*N, D) sequence enables cross-modal interaction — each
token can attend to every token from both modalities.

The encoder is a disposable pretext component; only HRNet weights transfer
to Stage 2.

Spec reference: §2.1 step 5; config keys model.encoder_layers, model.nhead,
model.ffn_dim, model.dropout.
"""
import torch
import torch.nn as nn


class CrossModalTransformerEncoder(nn.Module):
    """
    Standard Transformer encoder (batch_first=True) with a learnable spatial
    positional embedding shared across all representations.

    Args:
        embed_dim:  Token / model dimension (default 256).
        nhead:      Number of attention heads (default 8).
        ffn_dim:    Feed-forward hidden dimension (default 1024).
        num_layers: Number of encoder layers (default 4).
        dropout:    Dropout rate (default 0.1).
        max_tokens: Maximum sequence length per representation for the
                    positional embedding table.  Defaults to 3136 = 56*56
                    for 224×224 input with W18 HRNet.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        nhead: int = 8,
        ffn_dim: int = 1024,
        num_layers: int = 4,
        dropout: float = 0.1,
        max_tokens: int = 3136,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.max_tokens = max_tokens

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,        # pre-norm (more stable for small batches)
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(embed_dim),
        )

        # Learnable spatial positional embedding — shared across modalities.
        # Shape (1, N, D) is added to each per-representation token block
        # inside SSMERv2.forward() before tokens are concatenated, so the
        # encoder sees absolute spatial positions regardless of modality.
        self.pos_embed = nn.Parameter(torch.zeros(1, max_tokens, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: (B, 2*N, embed_dim) — concatenated visible pair tokens,
                    already type-embedded and positionally-encoded.

        Returns:
            (B, 2*N, embed_dim) — cross-modally encoded tokens.
        """
        return self.encoder(tokens)
