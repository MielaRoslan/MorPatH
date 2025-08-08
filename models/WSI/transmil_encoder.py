import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from torch import einsum
from math import ceil

# -----------------------
# utils
# -----------------------
def exists(x): return x is not None

def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

# -----------------------
# Nystrom Attention stack
# -----------------------
def moore_penrose_iter_pinv(x, iters=6):
    device = x.device
    abs_x = torch.abs(x)
    col = abs_x.sum(dim=-1)
    row = abs_x.sum(dim=-2)
    z = rearrange(x, '... i j -> ... j i') / (torch.max(col) * torch.max(row))

    I = torch.eye(x.shape[-1], device=device)
    I = rearrange(I, 'i j -> () i j')

    for _ in range(iters):
        xz = x @ z
        z = 0.25 * z @ (13 * I - (xz @ (15 * I - (xz @ (7 * I - xz)))))
    return z


class NystromAttention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head=64,
        heads=8,
        num_landmarks=256,
        pinv_iterations=6,
        residual=True,
        residual_conv_kernel=33,
        eps=1e-8,
        dropout=0.0,
    ):
        super().__init__()
        self.eps = eps
        inner_dim = heads * dim_head
        self.num_landmarks = num_landmarks
        self.pinv_iterations = pinv_iterations
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

        self.residual = residual
        if residual:
            padding = residual_conv_kernel // 2
            self.res_conv = nn.Conv2d(
                heads, heads, (residual_conv_kernel, 1),
                padding=(padding, 0), groups=heads, bias=False
            )

    def forward(self, x, mask=None, return_attn=False):
        # x: [B, N, D]
        b, n, d = x.shape
        h = self.heads
        m = self.num_landmarks
        iters, eps = self.pinv_iterations, self.eps

        # keep original length before padding
        orig_n = n

        # pad on the left so n % m == 0
        remainder = n % m
        if remainder > 0:
            padding = m - remainder
            x = F.pad(x, (0, 0, padding, 0), value=0.0)  # pad N-dim (left)
            if exists(mask):
                mask = F.pad(mask, (padding, 0), value=False)
            n = x.shape[1]

        # qkv
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))

        if exists(mask):
            mask = rearrange(mask, 'b n -> b () n')
            q = q * mask[..., None]
            k = k * mask[..., None]
            v = v * mask[..., None]

        q = q * self.scale

        # landmarks mean
        l = ceil(n / m)
        landmark_eq = '... (n l) d -> ... n d'
        q_land = reduce(q, landmark_eq, 'sum', l=l)
        k_land = reduce(k, landmark_eq, 'sum', l=l)

        divisor = l
        if exists(mask):
            mask_land_sum = reduce(mask, '... (n l) -> ... n', 'sum', l=l)
            divisor = mask_land_sum[..., None] + eps
            mask_land = mask_land_sum > 0

        q_land /= divisor
        k_land /= divisor

        # sims
        sim1 = einsum('... i d, ... j d -> ... i j', q,      k_land)
        sim2 = einsum('... i d, ... j d -> ... i j', q_land, k_land)
        sim3 = einsum('... i d, ... j d -> ... i j', q_land, k)

        if exists(mask):
            mask_value = -torch.finfo(q.dtype).max
            sim1.masked_fill_(~(mask[..., None] * mask_land[..., None, :]), mask_value)
            sim2.masked_fill_(~(mask_land[..., None] * mask_land[..., None, :]), mask_value)
            sim3.masked_fill_(~(mask_land[..., None] * mask[..., None, :]), mask_value)

        attn1, attn2, attn3 = map(lambda t: t.softmax(dim=-1), (sim1, sim2, sim3))
        attn2_inv = moore_penrose_iter_pinv(attn2, iters)

        out = (attn1 @ attn2_inv) @ (attn3 @ v)

        if self.residual:
            out += self.res_conv(v)

        out = rearrange(out, 'b h n d -> b n (h d)', h=h)
        out = self.to_out(out)

        # slice back to original length (we padded on the left)
        out = out[:, -orig_n:]

        if return_attn:
            attn = attn1 @ attn2_inv @ attn3
            return out, attn
        return out


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim),
        )

    def forward(self, x):
        return self.net(x)


class Nystromformer(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        dim_head=64,
        heads=8,
        num_landmarks=256,
        pinv_iterations=6,
        attn_values_residual=True,
        attn_values_residual_conv_kernel=33,
        attn_dropout=0.0,
        ff_dropout=0.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList([
                    PreNorm(dim, NystromAttention(
                        dim=dim, dim_head=dim_head, heads=heads,
                        num_landmarks=num_landmarks, pinv_iterations=pinv_iterations,
                        residual=attn_values_residual,
                        residual_conv_kernel=attn_values_residual_conv_kernel,
                        dropout=attn_dropout
                    )),
                    PreNorm(dim, FeedForward(dim=dim, dropout=ff_dropout)),
                ])
            )

    def forward(self, x, mask=None):
        for attn, ff in self.layers:
            x = attn(x, mask=mask) + x
            x = ff(x) + x
        return x

# -----------------------
# PPEG + TransMIL encoder
# -----------------------
class TransLayer(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = NystromAttention(
            dim=dim, dim_head=dim // 8, heads=8,
            num_landmarks=dim // 2, pinv_iterations=6,
            residual=True, dropout=0.1,
        )

    def forward(self, x):
        return x + self.attn(self.norm(x))


class PPEG(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.proj  = nn.Conv2d(dim, dim, 7, 1, 7 // 2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5 // 2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3 // 2, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]                 # [B,1,C], [B,N,C]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)    # [B,C,H,W]
        x = self.proj(cnn_feat) + cnn_feat + self.proj1(cnn_feat) + self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)                          # [B,N,C]
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)         # [B,1+N,C]
        return x


class TransMILEncoder(nn.Module):
    """
    WSI -> slide embedding using TransMIL backbone (no classifier inside).

    Input:
      data_WSI: [N_patches, in_dim] or [1, N_patches, in_dim]

    Output:
      z_wsi: [out_dim]  (default 512, set out_dim to 256 if you want to match other encoders)
    """
    def __init__(self, in_dim=1024, out_dim=512, act='gelu', dropout=0.25):
        super().__init__()
        self.embed_dim = 512
        # patch MLP
        fc = [nn.Linear(in_dim, self.embed_dim)]
        fc += [nn.GELU() if act.lower() == 'gelu' else nn.ReLU()]
        if dropout and dropout > 0:
            fc += [nn.Dropout(dropout)]
        self._fc1 = nn.Sequential(*fc)

        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embed_dim))
        nn.init.normal_(self.cls_token, std=1e-6)

        self.pos_layer = PPEG(dim=self.embed_dim)
        self.layer1 = TransLayer(dim=self.embed_dim)
        self.layer2 = TransLayer(dim=self.embed_dim)
        self.norm = nn.LayerNorm(self.embed_dim)

        # optional projection to desired out_dim
        self.proj = nn.Identity() if out_dim == self.embed_dim else nn.Linear(self.embed_dim, out_dim)

        self.apply(initialize_weights)

    def forward(self, data_WSI: torch.Tensor) -> torch.Tensor:
        """
        Returns:
          z: [out_dim]
        """
        x = data_WSI
        if x.dim() == 2:                      # [N, D] -> [1, N, D]
            x = x.unsqueeze(0)
        x = x.float()

        h = self._fc1(x)                      # [B, N, 512]

        # pad to square number of tokens
        B, N, C = h.shape
        H_len = int(np.ceil(np.sqrt(N)))
        add_len = H_len * H_len - N
        if add_len > 0:
            h = torch.cat([h, h[:, :add_len, :]], dim=1)  # simple repeat padding

        # prepend CLS (on correct device)
        cls = self.cls_token.expand(h.size(0), -1, -1).to(h.device)
        h = torch.cat((cls, h), dim=1)        # [B, 1+Npad, C]

        # layers + pos enhancement
        h = self.layer1(h)
        h = self.pos_layer(h, H_len, H_len)
        h = self.layer2(h)

        # take CLS and normalize
        h = self.norm(h)[:, 0]                # [B, 512]
        z = self.proj(h).squeeze(0)           # [out_dim]
        return z
