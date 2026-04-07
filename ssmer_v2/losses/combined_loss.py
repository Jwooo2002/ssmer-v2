"""
CombinedLoss: config-driven weighted sum of all v2 loss components.

    L = L_recon  (if use_reconstruction)
      + lambda_contrastive * L_contrastive  (if use_contrastive)
      + lambda_orth * L_orth  (if use_shared_private)

Returns both the scalar total and a component dict for per-term logging.

Spec reference: §2.1 step 8; §3 config keys loss.*.
"""
import torch
import torch.nn as nn
from omegaconf import DictConfig

from ssmer_v2.losses.reconstruction_loss import ReconstructionLoss
from ssmer_v2.losses.contrastive_loss import ContrastiveLoss
from ssmer_v2.losses.orthogonality_loss import OrthogonalityLoss


class CombinedLoss(nn.Module):
    """
    Aggregates all active loss components according to the experiment config.

    Args:
        cfg: OmegaConf DictConfig (full config, uses cfg.loss and cfg.model).
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        loss_cfg = cfg.loss
        model_cfg = cfg.model

        self.use_recon = loss_cfg.use_reconstruction
        self.use_contrastive = loss_cfg.use_contrastive
        self.use_orth = loss_cfg.use_shared_private

        self.lambda_contrastive = loss_cfg.lambda_contrastive
        self.lambda_orth = getattr(loss_cfg, "lambda_orth", 0.1)

        if self.use_recon:
            self.recon_loss = ReconstructionLoss(
                loss_type=loss_cfg.recon_type,
                normalize_features=loss_cfg.normalize_recon_features,
            )
        if self.use_contrastive:
            self.contrastive_loss = ContrastiveLoss()
        if self.use_orth:
            self.orth_loss = OrthogonalityLoss()

    def forward(self, model_output: dict) -> tuple[torch.Tensor, dict]:
        """
        Args:
            model_output: dict returned by SSMERv2.forward(), containing:
                'recon_pred'          (B, C, H', W') or None
                'recon_target'        (B, C, H', W') or None  — detached
                'contrastive_outputs' list of (p1,p2,z1,z2) or None
                'shared_tokens'       list of (B,N,Ds) per modality or None
                'private_tokens'      list of (B,N,Dp) per modality or None

        Returns:
            (total_loss, log_dict) — log_dict has one entry per active term.
        """
        device = self._infer_device(model_output)
        total = torch.zeros(1, device=device).squeeze()
        log: dict[str, torch.Tensor] = {}

        # --- Reconstruction -----------------------------------------------
        if self.use_recon:
            L_recon = self.recon_loss(
                model_output["recon_pred"],
                model_output["recon_target"],
            )
            total = total + L_recon
            log["recon_loss"] = L_recon.detach()

        # --- Contrastive --------------------------------------------------
        if self.use_contrastive:
            L_cont, cont_log = self.contrastive_loss(
                model_output["contrastive_outputs"]
            )
            total = total + self.lambda_contrastive * L_cont
            log["contrastive_loss"] = L_cont.detach()
            log.update(cont_log)

        # --- Orthogonality (shared/private) --------------------------------
        if self.use_orth:
            shared_list = model_output["shared_tokens"]   # list[Tensor]
            private_list = model_output["private_tokens"] # list[Tensor]
            L_orth = torch.zeros(1, device=device).squeeze()
            for sh, pr in zip(shared_list, private_list):
                L_orth = L_orth + self.orth_loss(sh, pr)
            total = total + self.lambda_orth * L_orth
            log["orth_loss"] = L_orth.detach()

        log["total_loss"] = total.detach()
        return total, log

    @staticmethod
    def _infer_device(model_output: dict) -> torch.device:
        """Extract device from the first available tensor in model_output."""
        for v in model_output.values():
            if isinstance(v, torch.Tensor):
                return v.device
            if isinstance(v, list) and v and isinstance(v[0], torch.Tensor):
                return v[0].device
            if isinstance(v, list) and v and isinstance(v[0], (list, tuple)):
                return v[0][0].device
        return torch.device("cpu")
