import math
from typing import Optional

import torch
from torch import nn


def causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """Return a causal mask that blocks attention to future positions."""
    # True where future positions should be masked
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
    return mask


def local_causal_mask(
    seq_len: int, device: torch.device, local_window: Optional[int]
) -> torch.Tensor:
    """Return a causal mask optionally limited to a local attention window."""
    # True where positions should be masked (future or beyond local window).
    i = torch.arange(seq_len, device=device).unsqueeze(1)
    j = torch.arange(seq_len, device=device).unsqueeze(0)
    mask = j > i
    if local_window is not None and local_window > 0:
        mask = mask | ((i - j) >= local_window)
    return mask


class StandardMultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float):
        """Standard multi-head self-attention without relative positions."""
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        B, L, _ = x.shape
        qkv = self.qkv(x).view(B, L, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        q = q.transpose(1, 2)  # (B, H, L, Dh)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = causal_mask(L, x.device)
        scores = scores.masked_fill(mask, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)  # (B, H, L, Dh)
        out = out.transpose(1, 2).contiguous().view(B, L, self.d_model)
        return self.out(out)


class RelativeMultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float,
        max_relative_position: int = 64,
        local_window: Optional[int] = None,
    ):
        """Multi-head attention with relative positional logits and optional local mask."""
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.max_relative_position = max_relative_position
        self.local_window = local_window

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

        # Relative position embeddings for distances in [-max_rel, max_rel]
        self.rel_emb = nn.Embedding(2 * max_relative_position + 1, self.head_dim)

    def _relative_logits(self, q: torch.Tensor) -> torch.Tensor:
        """Compute relative positional logits using a skew-style indexing."""
        # Memory-efficient relative attention using skew-style indexing.
        # Computes Q * E_r^T (L x 2L-1) and then re-indexes to (L x L).
        B, H, L, _ = q.shape
        rel_positions = torch.arange(-(L - 1), L, device=q.device)
        rel_positions = rel_positions.clamp(
            -self.max_relative_position, self.max_relative_position
        )
        rel_positions = rel_positions + self.max_relative_position
        rel_emb = self.rel_emb(rel_positions)  # (2L-1, Dh)
        rel_logits = torch.matmul(q, rel_emb.t())  # (B, H, L, 2L-1)

        # Skew: map relative positions to absolute indices.
        idx = torch.arange(L, device=q.device)
        rel_idx = idx[None, :] - idx[:, None] + (L - 1)  # (L, L)
        rel_logits = rel_logits.gather(
            3, rel_idx.view(1, 1, L, L).expand(B, H, L, L)
        )
        return rel_logits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply relative attention with causal (and optional local) masking."""
        # x: (B, L, D)
        B, L, _ = x.shape
        qkv = self.qkv(x).view(B, L, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        q = q.transpose(1, 2)  # (B, H, L, Dh)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Standard content-based attention
        content_scores = torch.matmul(q, k.transpose(-2, -1))

        # Relative positional attention using skew-style indexing.
        rel_scores = self._relative_logits(q)

        scores = (content_scores + rel_scores) / math.sqrt(self.head_dim)
        mask = local_causal_mask(L, x.device, self.local_window)
        scores = scores.masked_fill(mask, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)  # (B, H, L, Dh)
        out = out.transpose(1, 2).contiguous().view(B, L, self.d_model)
        return self.out(out)
