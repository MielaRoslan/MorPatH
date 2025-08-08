import torch
import torch.nn as nn

def _ensure_batch(x: torch.Tensor) -> torch.Tensor:
    return x.unsqueeze(0) if x.dim() == 1 else x

class LowRankBilinearFuse(nn.Module):
    """
    Low-rank bilinear pooling:
      h1 = W1 z1 -> r
      h2 = W2 z2 -> r
      b  = h1 * h2
      out = MLP(b) -> out_dim
    """
    def __init__(self, dim1: int, dim2: int, out_dim: int, rank: int = 128, dropout: float = 0.0, act: str = "relu"):
        super().__init__()
        self.proj1 = nn.Linear(dim1, rank, bias=True)
        self.proj2 = nn.Linear(dim2, rank, bias=True)
        act_layer = nn.ReLU() if act.lower() == "relu" else nn.GELU()
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(rank, out_dim),
            act_layer
        )

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        z1, z2 = _ensure_batch(z1), _ensure_batch(z2)
        h1 = self.proj1(z1)
        h2 = self.proj2(z2)
        b = h1 * h2
        return self.head(b)
