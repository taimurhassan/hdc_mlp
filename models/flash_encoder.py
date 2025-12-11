import torch
import torch.nn as nn
import torch.nn.functional as F

class FLASHEncoder(nn.Module):
    """
    Simplified FLASH-style HDC encoder.
    - Learns a generator network that maps fixed noise vectors to rows
      of the encoding matrix W in R^{D x in_dim}.
    - Encodes input features x in R^{in_dim} into hypervectors h in R^{D}.
    """
    def __init__(self, in_dim: int, hd_dim: int = 4096, noise_dim: int = 32, hidden_dim: int = 128):
        super().__init__()
        self.in_dim = in_dim
        self.hd_dim = hd_dim
        self.noise_dim = noise_dim

        # Fixed noise for each HD dimension
        self.register_buffer("eps", torch.randn(hd_dim, noise_dim))

        # Generator f_theta: epsilon -> row of encoding matrix
        self.generator = nn.Sequential(
            nn.Linear(noise_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, in_dim)
        )

    def get_encoding_matrix(self):
        """
        Returns learned encoding matrix W in R^{D x in_dim}.
        """
        W = self.generator(self.eps)  # [D, in_dim]
        return W

    def forward(self, x):
        """
        x: [B, in_dim]
        returns: hypervectors h: [B, hd_dim]
        """
        W = self.get_encoding_matrix()  # [D, in_dim]
        # Encode with cosine-like transform
        # [B, D] = [B, in_dim] @ [in_dim, D]
        proj = x @ W.t()
        # Nonlinearity: sinusoidal to mimic random Fourier features
        h = torch.cos(proj)
        h = F.normalize(h, p=2, dim=-1)
        return h
