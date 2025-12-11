import torch
import torch.nn.functional as F

def la_loss(anchors, positives, negatives, tau: float = 1.5):
    """
    La loss as described in Section 3.1.3.

    anchors:   visual features f_a,      [B, D]
    positives: positive text features f_p, [B, D]
    negatives: negative text features f_n, [B, D]
    All assumed to be in latent space; we normalize them to get angular space.

    La = -1/B * sum_i log(γ_i / (γ_i + δ_i))
      γ_i = exp(-d(cos(θ_i,a), cos(θ_i,p)) / τ)
      δ_i = exp(-d(cos(θ_i,a), cos(θ_i,n)) / τ)
    """
    # Normalize to unit vectors (angular space)
    a = F.normalize(anchors, p=2, dim=-1)
    p = F.normalize(positives, p=2, dim=-1)
    n = F.normalize(negatives, p=2, dim=-1)

    # Euclidean distance in angular space
    d_ap = torch.norm(a - p, dim=-1)  # [B]
    d_an = torch.norm(a - n, dim=-1)  # [B]

    gamma = torch.exp(-d_ap / tau)
    delta = torch.exp(-d_an / tau)

    frac = gamma / (gamma + delta + 1e-12)
    loss = -torch.log(frac + 1e-12).mean()
    return loss
