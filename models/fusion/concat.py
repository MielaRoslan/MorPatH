import torch
import torch.nn as nn

def _ensure_batch(x: torch.Tensor) -> torch.Tensor:
    return x.unsqueeze(0) if x.dim() == 1 else x

class ConcatFuse(nn.Module):
    """
    Concatenate -> MLP -> out_dim
    """
    def __init__(self, dim1: int, dim2: int, out_dim: int, hidden: int = None, dropout: float = 0.0, act: str = "relu"):
        super().__init__()
        hidden = hidden or out_dim
        act_layer = nn.ReLU() if act.lower() == "relu" else nn.GELU()
        self.net = nn.Sequential(
            nn.Linear(dim1 + dim2, hidden),
            act_layer,
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
            act_layer,
        )

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        z1, z2 = _ensure_batch(z1), _ensure_batch(z2)
        x = torch.cat([z1, z2], dim=-1)
        return self.net(x)
