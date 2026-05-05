import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_neighbor_mask(feature_size: int, neighbor_size: int) -> torch.Tensor:
    n = feature_size * feature_size
    coords = torch.stack(torch.meshgrid(
        torch.arange(feature_size),
        torch.arange(feature_size),
        indexing="ij",
    ), dim=-1).view(n, 2).float()
    diff = coords.unsqueeze(0) - coords.unsqueeze(1)
    half = neighbor_size // 2
    mask = (diff.abs().max(dim=-1).values <= half)
    return mask


def feature_jitter(x: torch.Tensor, scale: float, prob: float) -> torch.Tensor:
    if not (x.requires_grad or x.is_floating_point()):
        return x
    if prob <= 0.0 or scale <= 0.0:
        return x
    norm = x.norm(dim=-1, keepdim=True)
    noise = torch.randn_like(x)
    noise = noise * norm / (norm.shape[-1] ** 0.5) / scale
    keep = (torch.rand(x.shape[:-1], device=x.device) < prob).unsqueeze(-1).float()
    return x + noise * keep


class MultiHeadAttentionMasked(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, attn_mask: Optional[torch.Tensor] = None):
        b, nq, _ = q.shape
        nk = k.shape[1]
        Q = self.q_proj(q).view(b, nq, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(k).view(b, nk, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(v).view(b, nk, self.num_heads, self.head_dim).transpose(1, 2)
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = attn @ V
        out = out.transpose(1, 2).contiguous().view(b, nq, self.d_model)
        return self.out_proj(out)


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dim_ff: int, dropout: float):
        super().__init__()
        self.self_attn = MultiHeadAttentionMasked(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        x = x + self.drop(self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x), attn_mask))
        x = x + self.drop(self.ff(self.norm2(x)))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dim_ff: int, dropout: float):
        super().__init__()
        self.self_attn = MultiHeadAttentionMasked(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttentionMasked(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, q, memory, self_mask=None, cross_mask=None):
        q1 = self.norm1(q)
        q = q + self.drop(self.self_attn(q1, q1, q1, self_mask))
        q2 = self.norm2(q)
        m = self.norm2(memory)
        q = q + self.drop(self.cross_attn(q2, m, m, cross_mask))
        q = q + self.drop(self.ff(self.norm3(q)))
        return q


class UniADTransformer(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        feature_size: int = 14,
        neighbor_mask_size: int = 7,
        jitter_scale: float = 20.0,
        jitter_prob: float = 1.0,
        layer_wise_query: bool = True,
    ):
        super().__init__()
        self.feature_size = feature_size
        self.num_tokens = feature_size * feature_size
        self.hidden_dim = hidden_dim
        self.layer_wise_query = layer_wise_query
        self.jitter_scale = jitter_scale
        self.jitter_prob = jitter_prob

        self.input_proj = nn.Linear(feature_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, feature_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens, hidden_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        if layer_wise_query:
            self.queries = nn.Parameter(torch.zeros(num_decoder_layers, self.num_tokens, hidden_dim))
        else:
            self.queries = nn.Parameter(torch.zeros(1, self.num_tokens, hidden_dim))
        nn.init.trunc_normal_(self.queries, std=0.02)

        self.encoder = nn.ModuleList([
            EncoderLayer(hidden_dim, num_heads, dim_feedforward, dropout)
            for _ in range(num_encoder_layers)
        ])
        self.decoder = nn.ModuleList([
            DecoderLayer(hidden_dim, num_heads, dim_feedforward, dropout)
            for _ in range(num_decoder_layers)
        ])
        self.norm_enc = nn.LayerNorm(hidden_dim)
        self.norm_dec = nn.LayerNorm(hidden_dim)

        nbr = build_neighbor_mask(feature_size, neighbor_mask_size)
        self.register_buffer("neighbor_mask", nbr.unsqueeze(0).unsqueeze(0))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        b = tokens.shape[0]
        x = self.input_proj(tokens)
        if self.training:
            x = feature_jitter(x, self.jitter_scale, self.jitter_prob)
        x = x + self.pos_embed

        mask = self.neighbor_mask.expand(b, -1, -1, -1)
        for layer in self.encoder:
            x = layer(x, attn_mask=mask)
        memory = self.norm_enc(x)

        if self.layer_wise_query:
            out = None
            for i, layer in enumerate(self.decoder):
                q = self.queries[i].unsqueeze(0).expand(b, -1, -1) + self.pos_embed
                out = layer(q, memory, self_mask=mask, cross_mask=mask)
                memory = out
            dec_out = self.norm_dec(out)
        else:
            q = self.queries.expand(b, -1, -1) + self.pos_embed
            for layer in self.decoder:
                q = layer(q, memory, self_mask=mask, cross_mask=mask)
            dec_out = self.norm_dec(q)

        return self.output_proj(dec_out)
