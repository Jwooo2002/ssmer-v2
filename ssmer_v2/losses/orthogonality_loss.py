"""
OrthogonalityLoss: enforces independence between shared and private features.

Used only when shared_private.enabled=True (ablation A6).

Loss = || shared^T @ private ||_F^2  / (B * N)^2

Minimising the Frobenius norm of the cross-correlation matrix pushes the
shared and private subspaces towards orthogonality.

Spec reference: §2.1 step 7 (shared/private path); config key loss.lambda_orth.
"""
import torch
import torch.nn as nn


class OrthogonalityLoss(nn.Module):
    """
    Computes the orthogonality penalty between shared and private token
    sequences output by SharedPrivateHead.

    Args:
        eps: Small constant for numerical stability.
    """

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps

    def forward(
        self,
        shared: torch.Tensor,
        private: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            shared:  (B, N, shared_dim)
            private: (B, N, private_dim)

        Returns:
            Scalar orthogonality penalty.
        """
        B, N, Ds = shared.shape
        _, _, Dp = private.shape

        # Flatten batch and spatial dims → (B*N, D)
        s = shared.reshape(B * N, Ds)
        p = private.reshape(B * N, Dp)

        # Normalise rows so loss scale is independent of feature magnitude
        s = s / (s.norm(dim=1, keepdim=True) + self.eps)
        p = p / (p.norm(dim=1, keepdim=True) + self.eps)

        # Cross-correlation: (Ds, Dp)
        cross = s.T @ p   # (Ds, Dp)

        return (cross ** 2).sum() / ((B * N) ** 2)
