# models/histogram/ft_transformer.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from einops import rearrange, repeat


# ----------------------------
# Feedforward + Attention bits
# ----------------------------

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


def FeedForward(dim, mult: float = 4.0, dropout: float = 0.0):
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, int(dim * mult) * 2),
        GEGLU(),
        nn.Dropout(dropout),
        nn.Linear(int(dim * mult), dim)
    )


class Attention(nn.Module):
    def __init__(self, dim, heads: int = 8, dim_head: int = 64, dropout: float = 0.0):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: [B, N, D]
        returns:
          out:   [B, N, D]
          attn:  [B, H, N, N]  (post-softmax)
        """
        h = self.heads
        x = self.norm(x)
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)                # [B, N, H*Dh] * 3
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))
        q = q * self.scale
        sim = einsum('b h i d, b h j d -> b h i j', q, k)        # [B, H, N, N]
        attn = sim.softmax(dim=-1)
        dropped_attn = self.dropout(attn)
        out = einsum('b h i j, b h j d -> b h i d', dropped_attn, v)  # [B, H, N, Dh]
        out = rearrange(out, 'b h n d -> b n (h d)', h=h)        # [B, N, H*Dh]
        out = self.to_out(out)                                   # [B, N, D]
        return out, attn


class Transformer(nn.Module):
    def __init__(self, dim, depth: int, heads: int, dim_head: int,
                 attn_dropout: float, ff_dropout: float):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head, dropout=attn_dropout),
                FeedForward(dim, dropout=ff_dropout),
            ]))

    def forward(self, x, return_attn: bool = False):
        """
        x: [B, N, D]
        returns:
          x: [B, N, D] (or with stacked attention maps if return_attn=True)
        """
        post_softmax_attns = []
        for attn, ff in self.layers:
            attn_out, post_softmax_attn = attn(x)
            post_softmax_attns.append(post_softmax_attn)
            x = attn_out + x
            x = ff(x) + x

        if not return_attn:
            return x
        return x, torch.stack(post_softmax_attns)   # [L, B, H, N, N]


# ----------------------------
# Embeddings
# ----------------------------

class NumericalEmbedder(nn.Module):
    """
    Learns a per-feature (per-bin) affine embedding.
    Expects x as:
      - [B, C] and will unsqueeze to [B, C, 1], or
      - [B, C, 1] directly
    Outputs:
      - [B, C, D] after broadcasting weights/biases per feature.
    """
    def __init__(self, dim: int, num_numerical_types: int):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(num_numerical_types, dim))
        self.biases  = nn.Parameter(torch.randn(num_numerical_types, dim))

    def forward(self, x):
        if x.ndim == 2:                # [B, C] -> [B, C, 1]
            x = x.unsqueeze(-1)
        elif x.shape[-1] != 1:
            raise ValueError(f"Expected x with shape [B, C] or [B, C, 1], got {tuple(x.shape)}")
        return x * self.weights.unsqueeze(0) + self.biases.unsqueeze(0)


class PositionalEncoding(nn.Module):
    """
    Learnable positional encodings for a fixed number of tokens.
    num_tokens: e.g., number of bins (C) or number of features (F)
    dim: transformer embedding dimension
    """
    def __init__(self, num_tokens: int, dim: int):
        super().__init__()
        self.encoding = nn.Parameter(torch.randn(1, num_tokens, dim))

    def forward(self, x):
        # x: [B, N, D]
        return x + self.encoding


# ----------------------------
# FT-Transformer (extended but backward-compatible)
# ----------------------------

class FTTransformer(nn.Module):
    """
    Tabular FT-Transformer for (optional) categorical + numerical inputs.

    Extra knobs:
      token_strategy: "replicate" (default), "per_bin", "per_feature"
      bins_per_feature: int, required for "per_feature" (e.g., 10 for 9×10)
      pool: "cls" (default) or "mean"
      input_norm: if True, LayerNorm on x_numer before tokenization

    Forward (unchanged API):
      forward(x_categ, x_numer, return_attn=False)
      returns [B, output_dim] (and optionally attention maps)
    """
    def __init__(self,
                 categories,
                 num_continuous: int,
                 dim: int,
                 depth: int,
                 heads: int,
                 dim_head: int = 16,
                 output_dim: int = 256,
                 num_special_tokens: int = 2,
                 attn_dropout: float = 0.3,
                 ff_dropout: float = 0.5,
                 # NEW:
                 token_strategy: str = "replicate",   # "replicate" | "per_bin" | "per_feature"
                 bins_per_feature: int = None,
                 pool: str = "cls",                   # "cls" | "mean"
                 input_norm: bool = False):
        super().__init__()

        # ---- categorical (kept for API) ----
        self.num_categories = len(categories)
        self.num_unique_categories = sum(categories)
        self.num_special_tokens = num_special_tokens
        total_tokens = self.num_unique_categories + num_special_tokens

        if self.num_unique_categories > 0:
            categories_offset = F.pad(torch.tensor(list(categories)), (1, 0), value=num_special_tokens)
            categories_offset = categories_offset.cumsum(dim=-1)[:-1]
            self.register_buffer('categories_offset', categories_offset)
            self.categorical_embeds = nn.Embedding(total_tokens, dim)

        # ---- numerical setup ----
        self.num_continuous = num_continuous
        self.token_strategy = token_strategy.lower()
        self.pool = pool.lower()
        self.input_norm = nn.LayerNorm(num_continuous) if input_norm else None

        if self.token_strategy not in ("replicate", "per_bin", "per_feature"):
            raise ValueError("token_strategy must be 'replicate', 'per_bin', or 'per_feature'")

        # replicate: Linear to D then repeat across C tokens
        if self.token_strategy == "replicate":
            self.numerical_embedder_rep = nn.Linear(num_continuous, dim)
            self.pos_rep = PositionalEncoding(num_continuous, dim)

        # per_bin: each scalar bin -> token via affine NumericalEmbedder
        if self.token_strategy == "per_bin":
            self.numerical_embedder_bin = NumericalEmbedder(dim=dim, num_numerical_types=num_continuous)
            self.pos_bin = PositionalEncoding(num_continuous, dim)

        # per_feature: reshape [B, F*B] -> [B, F, B]; tokenize each feature via small MLP
        self.bins_per_feature = bins_per_feature
        if self.token_strategy == "per_feature":
            if bins_per_feature is None or num_continuous % bins_per_feature != 0:
                raise ValueError("For token_strategy='per_feature', provide bins_per_feature dividing num_continuous.")
            num_features = num_continuous // bins_per_feature
            self.num_features = num_features
            self.feature_tokenizer = nn.Sequential(
                nn.LayerNorm(bins_per_feature),
                nn.Linear(bins_per_feature, dim),
                nn.GELU(),
                nn.Dropout(ff_dropout),
                nn.Linear(dim, dim)
            )
            self.pos_feat = PositionalEncoding(num_features, dim)

        # ---- transformer backbone ----
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.transformer = Transformer(
            dim=dim, depth=depth, heads=heads, dim_head=dim_head,
            attn_dropout=attn_dropout, ff_dropout=ff_dropout
        )

        # ---- output projection (pool -> proj) ----
        self.to_cls_embeddings = nn.Sequential(
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Dropout(ff_dropout),
            nn.Linear(dim, output_dim)
        )

    def _tokens_from_numerical(self, x_numer: torch.Tensor) -> torch.Tensor:
        """
        Convert x_numer [B, C] -> token sequence [B, N, D] based on token_strategy.
        """
        if self.input_norm is not None:
            x_numer = self.input_norm(x_numer)

        if self.token_strategy == "replicate":
            h = self.numerical_embedder_rep(x_numer)                 # [B, D]
            tokens = h.unsqueeze(1).repeat(1, self.num_continuous, 1)  # [B, C, D]
            tokens = self.pos_rep(tokens)
            return tokens

        if self.token_strategy == "per_bin":
            tokens = self.numerical_embedder_bin(x_numer)            # [B, C, D]
            tokens = self.pos_bin(tokens)
            return tokens

        # per_feature
        B = x_numer.size(0)
        F = self.num_features
        BINS = self.bins_per_feature
        x = x_numer.view(B, F, BINS)                                 # [B, F, BINS]
        x = self.feature_tokenizer(x)                                 # [B, F, D]
        tokens = self.pos_feat(x)
        return tokens

    def forward(self, x_categ=None, x_numer=None, return_attn: bool = False):
        xs = []

        # categorical tokens (optional)
        if self.num_unique_categories > 0 and x_categ is not None:
            x_categ = x_categ + self.categories_offset
            x_categ = self.categorical_embeds(x_categ)      # [B, Nc, D]
            xs.append(x_categ)

        # numerical tokens (histogram)
        if self.num_continuous > 0 and x_numer is not None:
            if x_numer.dim() == 1:
                x_numer = x_numer.unsqueeze(0)
            num_tokens = self._tokens_from_numerical(x_numer)   # [B, N, D]
            xs.append(num_tokens)

        # concatenate categorical + numerical tokens
        if not xs:
            raise ValueError("FTTransformer received no inputs. Provide x_numer and/or x_categ.")
        x = torch.cat(xs, dim=1)                                # [B, N_all, D]

        # prepend CLS (always, even if we pool by mean, to keep attn shapes consistent)
        b = x.shape[0]
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)                   # [B, 1+N, D]

        # encode
        x, attns = self.transform
