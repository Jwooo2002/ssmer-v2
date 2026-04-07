"""
ReconstructionLoss: feature-space reconstruction loss.

Computes the distance between the decoder's output feature map and the
stop-gradient HRNet feature map of the masked representation.

Loss is computed in FEATURE space, NOT pixel space.  The target must already
have gradients detached (stop_gradient applied in SSMERv2.forward).

Spec reference: §2.1 step 8; §7 "Reconstruction target and stop_gradient".
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ReconstructionLoss(nn.Module):
    """
    Args:
        loss_type:           "smooth_l1" (default, robust to outliers) or
                             "mse".
        normalize_features:  If True, L2-normalise both pred and target along
                             the channel dimension before computing the loss.
                             Keeps gradients well-scaled when feature norms
                             vary across modalities.
    """

    def __init__(
        self,
        loss_type: str = "smooth_l1",
        normalize_features: bool = False,
    ) -> None:
        super().__init__()
        if loss_type not in ("smooth_l1", "mse"):
            raise ValueError(f"Unknown loss_type: {loss_type!r}")
        self.loss_type = loss_type
        self.normalize_features = normalize_features

    def extra_repr(self) -> str:
        return (
            f"loss_type={self.loss_type!r}, "
            f"normalize_features={self.normalize_features}"
        )

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred:   (B, C, H', W') — decoder reconstruction, gradients flow.
            target: (B, C, H', W') — HRNet output of masked rep,
                    requires_grad must be False (asserted at runtime).

        Returns:
            Scalar loss value.
        """
        assert not target.requires_grad, (
            "ReconstructionLoss: target must be stop-gradient "
            "(call target.detach() before passing here)."
        )

        if self.normalize_features:
            pred = F.normalize(pred, dim=1)
            target = F.normalize(target, dim=1)

        if self.loss_type == "smooth_l1":
            return F.smooth_l1_loss(pred, target)
        return F.mse_loss(pred, target)
