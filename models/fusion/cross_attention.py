import torch
import torch.nn as nn

def _ensure_batch(x: torch.Tensor) -> torch.Tensor:
    return x.unsqueeze(0) if x.dim() == 1 else x

class CrossAttentionFuse(nn.Module):
    """
    Project z1, z2 -> common dim; run 1-layer MHA over tokens [z1, z2];
    pool (mean) -> out_dim.
    """
    def __init__(self, dim1: int, dim2: int, out_dim: int, common_dim: int = 256, heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.p1 = nn.Linear(dim1, common_dim)
        self.p2 = nn.Linear(dim2, common_dim)
        self.mha = nn.MultiheadAttention(embed_dim=common_dim, num_heads=heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.LayerNorm(common_dim),
            nn.Linear(common_dim, out_dim),
            nn.GELU()
        )

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        z1, z2 = _ensure_batch(z1), _ensure_batch(z2)   # [B,d]
        t1, t2 = self.p1(z1).unsqueeze(1), self.p2(z2).unsqueeze(1)  # [B,1,C]
        tokens = torch.cat([t1, t2], dim=1)             # [B,2,C]
        attn_out, _ = self.mha(tokens, tokens, tokens)  # [B,2,C]
        pooled = attn_out.mean(dim=1)                   # [B,C]
        return self.ff(pooled)
