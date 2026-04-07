"""
SSMERv2: top-level pre-training model.

Orchestrates the full 10-step forward pass described in the v2 spec §2.1:

  1. HRNet encode all 3 representations → (B, 270, H', W') each
  2. stop_gradient targets from HRNet outputs
  3. Tokenise + add type & positional embeddings
  4. Random (or fixed) masking — remove one of the three representations
  5. Concatenate visible pair → (B, 2*N, D)
  6. Transformer encoder
  7. Optional shared/private decomposition on encoder output
  8. Transformer decoder → reconstruct masked feature map
  9. Optional contrastive head on visible token pairs
 10. Return output dict consumed by CombinedLoss

Only MultiStemHRNet.shared_body weights are saved and transferred to Stage 2.
Every other component in this file is a disposable pretext task head.

Spec reference: §2.1, §2.2, §7.
"""
from __future__ import annotations

import random
import math

import torch
import torch.nn as nn
from omegaconf import DictConfig

from ssmer_v2.models.hrnet_encoder import MultiStemHRNet
from ssmer_v2.models.feature_to_token import FeatureToToken
from ssmer_v2.models.type_embedding import TypeEmbedding
from ssmer_v2.models.transformer_encoder import CrossModalTransformerEncoder
from ssmer_v2.models.transformer_decoder import CrossReconstructionDecoder
from ssmer_v2.models.contrastive_head import ContrastiveHead
from ssmer_v2.models.shared_private_head import SharedPrivateHead


# ---------------------------------------------------------------------------
# Representation index constants (matches TypeEmbedding)
# ---------------------------------------------------------------------------
_FRAME = TypeEmbedding.FRAME        # 0
_VOXEL = TypeEmbedding.VOXEL        # 1
_TS    = TypeEmbedding.TIMESURFACE  # 2
_REP_NAMES = ("frame", "voxel", "timesurface")


class SSMERv2(nn.Module):
    """
    SSMER v2 pre-training model.

    Args:
        cfg: Full OmegaConf DictConfig (uses cfg.model, cfg.masking).
        pretrained_hrnet: Optional path to v1 HRNet checkpoint for the shared
                          body.  Stems are always randomly initialised.
    """

    def __init__(
        self,
        cfg: DictConfig,
        pretrained_hrnet: str | None = None,
    ) -> None:
        super().__init__()
        m = cfg.model
        mask_cfg = cfg.masking

        embed_dim   = m.embed_dim          # 256
        nhead       = m.nhead              # 8
        ffn_dim     = m.ffn_dim            # 1024
        dropout     = m.dropout            # 0.1
        t_bins      = m.t_bins             # 5
        hrnet_out   = MultiStemHRNet.BODY_OUT_CHANNELS  # 270

        # Spatial token count: (input_size / 4)^2 — for 224×224 → 3136
        input_size  = cfg.data.input_size  # 224
        n_tokens    = (input_size // 4) ** 2   # 3136

        self.n_tokens   = n_tokens
        self.embed_dim  = embed_dim
        self.hrnet_out  = hrnet_out

        # --- Masking strategy -------------------------------------------
        self.mask_strategy      = mask_cfg.strategy       # "random" / "sequential" / "fixed"
        self.fixed_target_idx   = mask_cfg.fixed_target   # None or int
        self.dual_masking_ratio = mask_cfg.dual_masking_ratio  # 0.0 or e.g. 0.3
        self._sequential_counter = 0                       # internal state for sequential

        # --- Feature extraction -----------------------------------------
        self.hrnet = MultiStemHRNet(
            t_bins=t_bins,
            use_temporal_encoding=m.voxel_temporal_encoding,
            pretrained=pretrained_hrnet,
        )

        # Separate FeatureToToken per representation (modality-specific proj)
        self.f2t_frame = FeatureToToken(hrnet_out, embed_dim)
        self.f2t_voxel = FeatureToToken(hrnet_out, embed_dim)
        self.f2t_ts    = FeatureToToken(hrnet_out, embed_dim)

        # --- Token embeddings -------------------------------------------
        self.type_embed = TypeEmbedding(num_types=3, embed_dim=embed_dim)

        # Spatial positional embedding is owned by the encoder but also added
        # to tokens before concatenation (shared across modalities).
        # We store a reference so forward() can access it without going through
        # the encoder module every time.
        self.encoder = CrossModalTransformerEncoder(
            embed_dim=embed_dim,
            nhead=nhead,
            ffn_dim=ffn_dim,
            num_layers=m.encoder_layers,
            dropout=dropout,
            max_tokens=n_tokens,
        )

        # --- Decoder (reconstruction head) ------------------------------
        self.use_recon = m.use_recon
        if self.use_recon:
            self.decoder = CrossReconstructionDecoder(
                embed_dim=embed_dim,
                nhead=nhead,
                ffn_dim=ffn_dim,
                num_layers=m.decoder_layers,
                dropout=dropout,
                hrnet_out_ch=hrnet_out,
                max_tokens=n_tokens,
            )

        # --- Contrastive head -------------------------------------------
        self.use_contrastive = m.use_contrastive
        if self.use_contrastive:
            self.contrastive_head = ContrastiveHead(
                embed_dim=embed_dim,
                proj_dim=embed_dim,
                pred_dim=embed_dim // 2,
            )

        # --- Shared / private head --------------------------------------
        self.use_shared_private = m.use_shared_private
        if self.use_shared_private:
            sp = m.shared_private
            self.sp_head = SharedPrivateHead(
                embed_dim=embed_dim,
                shared_dim=sp.shared_dim,
                private_dim=sp.private_dim,
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _pick_mask_idx(self) -> int:
        """Choose which representation to mask based on strategy."""
        if self.mask_strategy == "random":
            return random.randint(0, 2)
        if self.mask_strategy == "fixed":
            return int(self.fixed_target_idx)
        if self.mask_strategy == "sequential":
            idx = self._sequential_counter % 3
            self._sequential_counter += 1
            return idx
        raise ValueError(f"Unknown mask strategy: {self.mask_strategy!r}")

    def _add_embeddings(
        self,
        tokens: torch.Tensor,
        type_id: int,
        pos_embed: torch.Tensor,
    ) -> torch.Tensor:
        """Apply type token + spatial positional embedding to a token sequence."""
        tokens = self.type_embed(tokens, type_id)
        tokens = tokens + pos_embed[:, : tokens.size(1), :]
        return tokens

    @staticmethod
    def _apply_dual_masking(
        tokens: torch.Tensor,
        ratio: float,
    ) -> torch.Tensor:
        """
        Randomly zero out `ratio` fraction of tokens in the visible sequence.
        Applied independently per sample.

        Args:
            tokens: (B, N, D)
            ratio:  Fraction of tokens to mask (0.0 = no masking).

        Returns:
            (B, N, D) with masked positions set to zero.
        """
        if ratio <= 0.0:
            return tokens
        B, N, D = tokens.shape
        n_mask = max(1, int(N * ratio))
        # Random indices to zero out — different per sample
        noise = torch.rand(B, N, device=tokens.device)
        mask_idx = noise.argsort(dim=1)[:, :n_mask]           # (B, n_mask)
        mask = torch.ones(B, N, device=tokens.device)
        mask.scatter_(1, mask_idx, 0.0)
        return tokens * mask.unsqueeze(-1)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        frame: torch.Tensor,
        voxel: torch.Tensor,
        timesurface: torch.Tensor,
    ) -> dict:
        """
        Full SSMERv2 forward pass.

        Args:
            frame:       (B, 1, H, W)
            voxel:       (B, T_bins, H, W)
            timesurface: (B, 1, H, W)

        Returns:
            dict with keys:
                recon_pred           (B, hrnet_out, H', W') or None
                recon_target         (B, hrnet_out, H', W') or None  — detached
                contrastive_outputs  list[(p1,p2,z1,z2)] or None
                shared_tokens        list[(B,N,Ds)] per visible rep, or None
                private_tokens       list[(B,N,Dp)] per visible rep, or None
                mask_idx             int (which representation was masked)
        """
        B = frame.size(0)
        pos_embed = self.encoder.pos_embed   # (1, N, D)

        # ----------------------------------------------------------------
        # Step 1: HRNet feature extraction
        # ----------------------------------------------------------------
        feat_f, feat_v, feat_ts = self.hrnet(frame, voxel, timesurface)
        # each: (B, 270, H', W')

        # ----------------------------------------------------------------
        # Step 2: Stop-gradient targets — MUST happen before tokenisation
        # so no gradient path exists from reconstruction loss to targets.
        # ----------------------------------------------------------------
        target_f  = feat_f.detach()
        target_v  = feat_v.detach()
        target_ts = feat_ts.detach()

        # ----------------------------------------------------------------
        # Step 3: Tokenise + embed
        # ----------------------------------------------------------------
        tok_f  = self._add_embeddings(self.f2t_frame(feat_f),  _FRAME, pos_embed)
        tok_v  = self._add_embeddings(self.f2t_voxel(feat_v),  _VOXEL, pos_embed)
        tok_ts = self._add_embeddings(self.f2t_ts(feat_ts),    _TS,    pos_embed)
        # each: (B, N, D)

        # ----------------------------------------------------------------
        # A1 short-circuit: contrastive-only path (no masking, no encoder,
        # no decoder).  All three representations are treated symmetrically,
        # matching the trio structure of v1 trio_train_jsd.
        # NOTE: this is v1-flavoured, not numerically identical to v1 —
        # embed_dim=256 vs v1's 2048, and the projector input is a spatially
        # averaged token rather than a pooled conv feature.
        # ----------------------------------------------------------------
        if not self.use_recon:
            contrastive_outputs = None
            if self.use_contrastive:
                contrastive_outputs = [
                    self.contrastive_head(tok_f,  tok_v),
                    self.contrastive_head(tok_v,  tok_ts),
                    self.contrastive_head(tok_f,  tok_ts),
                ]
            return {
                "recon_pred":           None,
                "recon_target":         None,
                "contrastive_outputs":  contrastive_outputs,
                "shared_tokens":        None,
                "private_tokens":       None,
                "mask_idx":             None,
            }

        # ----------------------------------------------------------------
        # Step 4: Random masking — choose which representation to mask
        # ----------------------------------------------------------------
        mask_idx = self._pick_mask_idx()

        # Map index → (visible_A, visible_B, masked_tokens, masked_target)
        _all_toks    = (tok_f,  tok_v,  tok_ts)
        _all_targets = (target_f, target_v, target_ts)
        _all_type_ids = (_FRAME, _VOXEL, _TS)
        _all_rep_names = _REP_NAMES

        visible_indices = [i for i in range(3) if i != mask_idx]
        tok_A = _all_toks[visible_indices[0]]
        tok_B = _all_toks[visible_indices[1]]
        masked_target = _all_targets[mask_idx]      # (B, 270, H', W'), detached
        masked_type_id = _all_type_ids[mask_idx]
        rep_name_A = _all_rep_names[visible_indices[0]]
        rep_name_B = _all_rep_names[visible_indices[1]]

        # ----------------------------------------------------------------
        # Optional dual masking (ablation A7): patch-level zeroing within
        # visible tokens.
        # ----------------------------------------------------------------
        if self.dual_masking_ratio > 0.0:
            tok_A = self._apply_dual_masking(tok_A, self.dual_masking_ratio)
            tok_B = self._apply_dual_masking(tok_B, self.dual_masking_ratio)

        # ----------------------------------------------------------------
        # Step 5: Concatenate visible pair
        # ----------------------------------------------------------------
        visible_tokens = torch.cat([tok_A, tok_B], dim=1)   # (B, 2*N, D)

        # ----------------------------------------------------------------
        # Step 6: Transformer encoder (cross-modal interaction)
        # ----------------------------------------------------------------
        encoded = self.encoder(visible_tokens)   # (B, 2*N, D)

        # Split back into per-representation halves.
        # Use tok_A.size(1) (actual token count) not self.n_tokens so that
        # this stays correct if spatial size ever deviates from the default.
        N_vis = tok_A.size(1)
        enc_A = encoded[:, :N_vis, :]    # (B, N, D)
        enc_B = encoded[:, N_vis:, :]    # (B, N, D)

        # ----------------------------------------------------------------
        # Step 7: Optional shared/private decomposition
        # ----------------------------------------------------------------
        shared_tokens_out  = None
        private_tokens_out = None

        if self.use_shared_private:
            sh_A, pr_A = self.sp_head(enc_A, rep_name_A)
            sh_B, pr_B = self.sp_head(enc_B, rep_name_B)
            shared_tokens_out  = [sh_A, sh_B]
            private_tokens_out = [pr_A, pr_B]

        # ----------------------------------------------------------------
        # Step 8: Decoder — reconstruct masked feature map
        # ----------------------------------------------------------------
        recon_pred   = None
        recon_target = None

        if self.use_recon:
            # Derive actual spatial dims from the target — handles any sensor
            # size and non-square inputs without hardcoding H_prime=56.
            _, _, Hp, Wp = masked_target.shape
            N_actual = Hp * Wp

            masked_type_emb = self.type_embed.embed(
                torch.tensor(masked_type_id, device=frame.device)
            )                                           # (D,)

            query = self.decoder.build_query(B, N_actual, masked_type_emb)
            decoded = self.decoder(query, encoded)      # (B, N_actual, hrnet_out)

            recon_pred = decoded.transpose(1, 2).reshape(
                B, self.hrnet_out, Hp, Wp
            )
            recon_target = masked_target                # already detached

        # ----------------------------------------------------------------
        # Step 9: Optional contrastive head on visible encoded tokens
        # ----------------------------------------------------------------
        contrastive_outputs = None

        if self.use_contrastive:
            # Compute all three pairwise contrastive losses, consistent with
            # v1 trio_train_jsd.  For the masked rep we use the original
            # (pre-encoder) token mean since it was never passed through the
            # encoder — use the raw projected tokens instead.
            tok_f_enc  = enc_A if visible_indices[0] == _FRAME  else (enc_B if visible_indices[1] == _FRAME  else tok_f)
            tok_v_enc  = enc_A if visible_indices[0] == _VOXEL  else (enc_B if visible_indices[1] == _VOXEL  else tok_v)
            tok_ts_enc = enc_A if visible_indices[0] == _TS     else (enc_B if visible_indices[1] == _TS     else tok_ts)

            contrastive_outputs = [
                self.contrastive_head(tok_f_enc,  tok_v_enc),
                self.contrastive_head(tok_v_enc,  tok_ts_enc),
                self.contrastive_head(tok_f_enc,  tok_ts_enc),
            ]

        return {
            "recon_pred":           recon_pred,
            "recon_target":         recon_target,
            "contrastive_outputs":  contrastive_outputs,
            "shared_tokens":        shared_tokens_out,
            "private_tokens":       private_tokens_out,
            "mask_idx":             mask_idx,
        }

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def backbone_state_dict(self) -> dict:
        """
        Returns only the trained HRNet *body* weights (layer1, stage2-4,
        transition layers).

        Deliberately excludes:
          - conv1 / bn1 / conv2 / bn2  — the original 3-ch RGB stem inside
            shared_body, which is NEVER called in v2 (MultiStemHRNet uses its
            own stem_frame/voxel/ts instead).  These weights are random init
            and would silently corrupt Stage 2's event stem.
          - head.*  — the classification head, also never called in v2.

        This filter is the symmetric inverse of _load_pretrained_body in
        hrnet_encoder.py, which uses the same prefix list when loading.

        Usage:
            torch.save(model.backbone_state_dict(), "backbone_0200.pth")
            # Stage 2: stage2_hrnet.load_state_dict(ckpt, strict=False)
        """
        skip_prefixes = ("conv1.", "bn1.", "conv2.", "bn2.", "head.")
        return {
            k: v
            for k, v in self.hrnet.shared_body.state_dict().items()
            if not any(k.startswith(p) for p in skip_prefixes)
        }
