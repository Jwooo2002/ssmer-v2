"""
ContrastiveLoss: negative cosine similarity + symmetric JSD.

Direct port of calculate_losses() from simsiam/trainer.py, re-packaged as a
stateless nn.Module so it integrates with CombinedLoss and can be toggled via
config without changing the training loop.

Spec reference: §2.1 step 9; config keys loss.use_contrastive,
loss.lambda_contrastive.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """
    Computes SimSiam contrastive loss over a list of (p1, p2, z1, z2) pairs.

    For each pair:
        L_cos  = -(cos_sim(p1, z2) + cos_sim(p2, z1)) / 2
        L_jsd  = 0.5 * [KL(p1_log || p2) + KL(p2_log || p1)]
        L_pair = L_cos + L_jsd

    Total loss = sum over all pairs.

    This matches the three-pair (frame-voxel, voxel-ts, frame-ts) calculation
    used in v1's trio_train_jsd exactly, enabling A1 ablation verification.
    """

    def __init__(self) -> None:
        super().__init__()
        self.cosine = nn.CosineSimilarity(dim=1)
        self.kl = nn.KLDivLoss(reduction="batchmean")

    def _pair_loss(
        self,
        p1: torch.Tensor,
        p2: torch.Tensor,
        z1: torch.Tensor,
        z2: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (cosine_loss, jsd_loss) for a single representation pair."""
        neg_cos = -(self.cosine(p1, z2).mean() + self.cosine(p2, z1).mean()) * 0.5
        jsd = 0.5 * (
            self.kl(p1.log_softmax(dim=1), p2.softmax(dim=1))
            + self.kl(p2.log_softmax(dim=1), p1.softmax(dim=1))
        )
        return neg_cos, jsd

    def forward(
        self,
        outputs: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Args:
            outputs: List of (p1, p2, z1, z2) tuples — one per representation
                     pair.  Typically 3 pairs for trio training.

        Returns:
            (total_loss, component_dict) where component_dict contains
            'cos_loss' and 'jsd_loss' for logging.
        """
        total_cos = torch.tensor(0.0, device=outputs[0][0].device)
        total_jsd = torch.tensor(0.0, device=outputs[0][0].device)

        for p1, p2, z1, z2 in outputs:
            cos, jsd = self._pair_loss(p1, p2, z1, z2)
            total_cos = total_cos + cos
            total_jsd = total_jsd + jsd

        total = total_cos + total_jsd
        return total, {"cos_loss": total_cos.detach(), "jsd_loss": total_jsd.detach()}
