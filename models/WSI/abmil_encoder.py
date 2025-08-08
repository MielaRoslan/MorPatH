import torch
import torch.nn as nn
import torch.nn.functional as F

class ABMIL(nn.Module):
    """
    WSI -> slide embedding (no classifier), using your CLAM-style attention.
    Matches your DAttention / GatedAttention math (L=512, D=128, K=1).
    Returns:
      z_wsi: [out_dim]
      attn:  [N] attention over instances (optional)
    """
    def __init__(self,
                 in_dim: int = 1024,
                 out_dim: int = 256,
                 attn_type: str = "gated",     # "gated" | "dot" (DAttention)
                 act: str = "relu",
                 dropout: bool = True):
        super().__init__()
        L = 512
        D = 128
        K = 1

        # feature projection (same as your code)
        feat_layers = [nn.Linear(in_dim, L)]
        if act.lower() == "gelu":
            feat_layers += [nn.GELU()]
        else:
            feat_layers += [nn.ReLU()]

        if dropout:
            feat_layers += [nn.Dropout(0.25)]

        self.feature = nn.Sequential(*feat_layers)

        # attention head(s)
        self.attn_type = attn_type.lower()
        if self.attn_type == "gated":
            self.attn_V = nn.Sequential(nn.Linear(L, D), nn.Tanh())
            self.attn_U = nn.Sequential(nn.Linear(L, D), nn.Sigmoid())
            self.attn_w = nn.Linear(D, K)
        elif self.attn_type in ("dot", "dattention"):
            self.attn = nn.Sequential(
                nn.Linear(L, D), nn.Tanh(), nn.Linear(D, K)
            )
        else:
            raise ValueError("attn_type must be 'gated' or 'dot'")

        # final projector to desired embedding size
        self.proj = nn.Linear(L * K, out_dim)

    def forward(self, data_WSI: torch.Tensor, return_attn: bool = False):
        """
        data_WSI: [N, in_dim]  bag of patch features (batch size = 1 slide)
        """
        H = self.feature(data_WSI.squeeze())            # [N, L]

        # attention -> weights A (K x N), pooled M (K x L)
        if self.attn_type == "gated":
            A_V = self.attn_V(H)                        # [N, D]
            A_U = self.attn_U(H)                        # [N, D]
            A = self.attn_w(A_V * A_U)                  # [N, K]
        else:
            A = self.attn(H)                            # [N, K]

        A = A.transpose(1, 0)                           # [K, N]
        A = F.softmax(A, dim=-1)                        # softmax over N
        M = torch.mm(A, H)                              # [K, L]

        z = self.proj(M).squeeze(0)                     # [out_dim]
        if return_attn:
            return z, A.squeeze(0)                      # A: [N]
        return z
