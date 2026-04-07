"""
CrossReconstructionDecoder: 2-layer Transformer decoder.

Takes a set of learnable query tokens representing the *masked* representation's
spatial positions and cross-attends to the encoder output to reconstruct the
masked HRNet feature map in feature space.

The decoder is entirely disposable — it is discarded after pre-training.

Spec reference: §2.1 steps 6–7; §7 "Reconstruction target and stop_gradient".
"""
import torch
import torch.nn as nn


class CrossReconstructionDecoder(nn.Module):
    """
    Transformer decoder that reconstructs the masked representation's
    HRNet feature map from encoder context.

    Args:
        embed_dim:    Token dimension (default 256).
        nhead:        Number of attention heads (default 8).
        ffn_dim:      Feed-forward hidden dimension (default 1024).
        num_layers:   Number of decoder layers (default 2).
        dropout:      Dropout rate (default 0.1).
        hrnet_out_ch: HRNet body output channels; the decoder output is
                      projected to this dimension to match the reconstruction
                      target.  Default 270 (W18 Stage4 sum).
        max_tokens:   Spatial token count per representation; used to
                      initialise the learnable mask token and query pos embed.
                      Default 3136 = 56*56.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        nhead: int = 8,
        ffn_dim: int = 1024,
        num_layers: int = 2,
        dropout: float = 0.1,
        hrnet_out_ch: int = 270,
        max_tokens: int = 3136,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.hrnet_out_ch = hrnet_out_ch
        self.max_tokens = max_tokens

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(embed_dim),
        )

        # Learnable mask token: a single vector broadcast to fill all N
        # query positions before type/pos embeddings are added.
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        # Learnable spatial positional embedding for decoder queries.
        # Separate from the encoder's pos_embed so the decoder can learn
        # query-specific position biases.
        self.query_pos_embed = nn.Parameter(
            torch.zeros(1, max_tokens, embed_dim)
        )
        nn.init.trunc_normal_(self.query_pos_embed, std=0.02)

        # Project decoded tokens back to HRNet feature space for the
        # reconstruction loss.  Output dim = hrnet_out_ch (e.g. 270),
        # NOT pixel space.
        self.out_proj = nn.Linear(embed_dim, hrnet_out_ch)
        nn.init.trunc_normal_(self.out_proj.weight, std=0.02)
        nn.init.zeros_(self.out_proj.bias)

    def build_query(
        self,
        batch_size: int,
        n_tokens: int,
        type_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Construct the N query tokens for the masked representation.

        Query = mask_token (broadcast) + type_embedding[mask_idx] + pos_embed

        Args:
            batch_size: B
            n_tokens:   N (spatial token count, e.g. 3136)
            type_emb:   (embed_dim,) type embedding for the masked modality,
                        retrieved from TypeEmbedding outside this module.

        Returns:
            (B, N, embed_dim) query tensor — no gradient flows through
            mask_token into the reconstruction target (that is handled by
            stop_gradient on the target, not on the query).
        """
        # mask_token: (1,1,D) → (B, N, D)
        query = self.mask_token.expand(batch_size, n_tokens, -1).clone()
        # Add type identity so the decoder knows which modality to reconstruct
        query = query + type_emb.view(1, 1, -1)
        # Add spatial position — use only the first n_tokens positions to
        # support variable spatial sizes at inference time
        query = query + self.query_pos_embed[:, :n_tokens, :]
        return query

    def forward(
        self,
        query: torch.Tensor,
        encoder_memory: torch.Tensor,
    ) -> torch.Tensor:
        """
        Reconstruct the masked feature map from encoder context.

        Args:
            query:          (B, N, embed_dim) — output of build_query()
            encoder_memory: (B, 2*N, embed_dim) — encoder output

        Returns:
            (B, N, hrnet_out_ch) — projected decoded tokens, to be reshaped
            to (B, hrnet_out_ch, H', W') before computing the loss against
            the stop-gradient target.
        """
        decoded = self.decoder(query, encoder_memory)   # (B, N, D)
        return self.out_proj(decoded)                   # (B, N, hrnet_out_ch)
