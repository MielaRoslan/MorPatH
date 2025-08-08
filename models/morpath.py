# models/morpath.py

import torch
import torch.nn as nn

# --- WSI encoders (only these three) ---
from models.wsi.abmil_encoder import ABMILEncoder
from models.wsi.deepmisl_encoder import DeepMILEncoder
from models.wsi.transmil_encoder import TransMILEncoder

# --- Histogram encoder (FT-Transformer-based wrapper) ---
from models.histogram.hist_encoder import HistEncoder

# --- Fusion modules (only these three) ---
from models.fusion import (
    ConcatFuse,              # concat -> MLP
    LowRankBilinearFuse,     # low-rank bilinear pooling
    CrossAttentionFuse,      # 2-token cross-attn
)


class MorPath(nn.Module):
    """
    Encoders: (WSI backbone: ABMIL / DeepMISL / TransMIL) + FT-Transformer (Histogram).
    Fuses embeddings and outputs survival quantities.

    Forward kwargs:
      data_WSI  : [N_patches, wsi_feat_dim]  or [1, N_patches, wsi_feat_dim]
      data_hist : [hist_bins] or [1, hist_bins]
      return_attn: bool (ABMIL / TransMIL can return attention; DeepMISL returns None)
      **wsi_kwargs: forwarded to the WSI encoder (kept for flexibility)

    Returns:
      {
        "logits":   [1, n_classes],
        "hazards":  [1, n_classes],
        "survival": [1, n_classes],
        "risk":     [1],
        "z_wsi":    [wsi_out_dim],
        "z_hist":   [hist_embed_dim],
        "attn":     [N] or None (only when return_attn=True and encoder supports it)
      }
    """

    def __init__(
        self,
        # encoders / outputs
        wsi_backbone: str = "abmil",          # "abmil" | "deepmisl" | "transmil"
        fusion: str = "concat",               # "concat" | "bilinear" | "xattn"
        n_classes: int = 4,

        # WSI encoder (common)
        wsi_feat_dim: int = 1024,
        wsi_out_dim: int = 256,
        wsi_act: str = "relu",
        wsi_dropout: float = 0.25,

        # ABMIL-only
        wsi_attn_type: str = "gated",         # "gated" | "dot" (if your ABMILEncoder uses it)

        # Histogram encoder (FT-Transformer)
        hist_bins: int = 90,
        hist_embed_dim: int = 256,
        ftt_dim: int = 256, ftt_depth: int = 4, ftt_heads: int = 8, ftt_dim_head: int = 16,
        ftt_attn_dropout: float = 0.3, ftt_ff_dropout: float = 0.5,

        # fusion hyperparams
        fuse_hidden: int | None = None,
        fuse_rank: int = 128,
        fuse_common_dim: int = 256,
        fuse_heads: int = 4,
        fuse_dropout: float = 0.1,
        fuse_act: str = "relu",
    ):
        super().__init__()

        self.fusion = fusion.lower()
        self.wsi_backbone = wsi_backbone.lower()

        # ----- WSI encoder -----
        if self.wsi_backbone == "abmil":
            # If your ABMILEncoder doesn't take attn_type, remove that kwarg.
            self.wsi_enc = ABMILEncoder(
                in_dim=wsi_feat_dim, out_dim=wsi_out_dim,
                act=wsi_act, dropout=wsi_dropout, attn_type=wsi_attn_type
            )
        elif self.wsi_backbone == "deepmisl":
            self.wsi_enc = DeepMILEncoder(
                in_dim=wsi_feat_dim, mid_dim=512, out_dim=wsi_out_dim,
                dropout=wsi_dropout, act=wsi_act, pool="sum"
            )
        elif self.wsi_backbone == "transmil":
            self.wsi_enc = TransMILEncoder(
                in_dim=wsi_feat_dim, out_dim=wsi_out_dim,
                act=wsi_act, dropout=wsi_dropout
            )
        else:
            raise ValueError("wsi_backbone must be one of: 'abmil' | 'deepmisl' | 'transmil'")

        # ----- Histogram encoder -----
        self.hist_enc = HistEncoder(
            hist_bins=hist_bins, hist_embed_dim=hist_embed_dim,
            dim=ftt_dim, depth=ftt_depth, heads=ftt_heads, dim_head=ftt_dim_head,
            attn_dropout=ftt_attn_dropout, ff_dropout=ftt_ff_dropout
        )

        # ----- Fusion -----
        # All fusers output wsi_out_dim so the head stays consistent.
        if self.fusion == "concat":
            self.fuse = ConcatFuse(
                dim1=wsi_out_dim, dim2=hist_embed_dim, out_dim=wsi_out_dim,
                hidden=fuse_hidden or wsi_out_dim, dropout=fuse_dropout, act=fuse_act
            )
        elif self.fusion == "bilinear":
            self.fuse = LowRankBilinearFuse(
                dim1=wsi_out_dim, dim2=hist_embed_dim, out_dim=wsi_out_dim,
                rank=fuse_rank, dropout=fuse_dropout, act=fuse_act
            )
        elif self.fusion == "xattn":
            self.fuse = CrossAttentionFuse(
                dim1=wsi_out_dim, dim2=hist_embed_dim, out_dim=wsi_out_dim,
                common_dim=fuse_common_dim, heads=fuse_heads, dropout=fuse_dropout
            )
        else:
            raise ValueError("fusion must be one of: 'concat' | 'bilinear' | 'xattn'")

        # ----- Survival head -----
        self.classifier = nn.Linear(wsi_out_dim, n_classes)

    def forward(self, *, data_WSI, data_hist, return_attn: bool = False, **wsi_kwargs):
        # --- WSI ---
        if return_attn:
            out = self.wsi_enc(data_WSI, return_attn=True, **wsi_kwargs)
            if isinstance(out, tuple) and len(out) == 2:
                z_wsi, attn = out
            else:
                z_wsi, attn = out, None
        else:
            out = self.wsi_enc(data_WSI, return_attn=False, **wsi_kwargs)
            z_wsi, attn = (out, None) if not isinstance(out, tuple) else (out[0], None)

        # --- Histogram ---
        z_hist = self.hist_enc(data_hist)  # [hist_embed_dim]

        # --- Fuse ---
        z = self.fuse(z_wsi, z_hist)
        if z.dim() == 2 and z.size(0) == 1:
            z = z.squeeze(0)

        # --- Survival ---
        logits  = self.classifier(z).unsqueeze(0)    # [1, n_classes]
        hazards = torch.sigmoid(logits)
        survival = torch.cumprod(1 - hazards, dim=1)
        risk = -torch.sum(survival, dim=1)           # [1]

        out = {
            "logits": logits,
            "hazards": hazards,
            "survival": survival,
            "risk": risk,
            "z_wsi": z_wsi,
            "z_hist": z_hist,
        }
        if return_attn:
            out["attn"] = attn
        return out

    # convenience
    @torch.no_grad()
    def risk_only(self, *, data_WSI, data_hist, **wsi_kwargs):
        return self.forward(data_WSI=data_WSI, data_hist=data_hist, **wsi_kwargs)["risk"]

    @torch.no_grad()
    def embeddings(self, *, data_WSI, data_hist, **wsi_kwargs):
        out = self.forward(data_WSI=data_WSI, data_hist=data_hist, return_attn=False, **wsi_kwargs)
        return out["z_wsi"], out["z_hist"]
