# models/wsi/deepmisl_encoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F

def _init_weights(m: nn.Module):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(m.bias, 0.0)

class DeepMILEncoder(nn.Module):
    """
    DeepMISL-style WSI encoder (Deep Sets) -> slide embedding.

    Inputs:
      - data_WSI: [N_patches, in_dim]  or  [1, N_patches, in_dim]

    Returns:
      - z_wsi: [out_dim]
      - if return_attn=True, returns (z_wsi, None) for API parity
        (Deep Sets has no attention weights)
    """
    def __init__(
        self,
        in_dim: int = 1024,
        mid_dim: int = 512,
        out_dim: int = 256,
        dropout: float = 0.25,
        act: str = "relu",
        pool: str = "sum",            # "sum" | "mean"
    ):
        super().__init__()
        act_layer = nn.GELU() if act.lower() == "gelu" else nn.ReLU()

        # φ: per-instance transform
        self.phi = nn.Sequential(
            nn.Linear(in_dim, mid_dim),
            act_layer,
            nn.Dropout(dropout),
        )

        # ρ: bag-level transform
        self.rho = nn.Sequential(
            nn.Linear(mid_dim, out_dim),
            act_layer,
            nn.Dropout(dropout),
        )

        pool = pool.lower()
        if pool not in ("sum", "mean"):
            raise ValueError("pool must be 'sum' or 'mean'")
        self.pool = pool

        self.apply(_init_weights)

    def forward(self, data_WSI: torch.Tensor, return_attn: bool = False):
        """
        data_WSI: [N, in_dim] or [1, N, in_dim]
        """
        x = data_WSI
        if x.dim() == 3:
            x = x.squeeze(0)                 # [N, in_dim]

        h = self.phi(x)                      # [N, mid_dim]

        if self.pool == "sum":
            bag = h.sum(dim=0)               # [mid_dim]
        else:
            bag = h.mean(dim=0)              # [mid_dim]

        z = self.rho(bag)                    # [out_dim]

        if return_attn:
            return z, None                   # API parity with attn encoders
        return z
