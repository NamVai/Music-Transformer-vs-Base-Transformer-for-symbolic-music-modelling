import torch
from torch import nn


class EncoderBlock(nn.Module):
    def __init__(self, attention: nn.Module, d_model: int, d_ff: int, dropout: float):
        """Pre-norm Transformer encoder block with attention and feed-forward."""
        super().__init__()
        self.attn = attention
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=True),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=True),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply attention, feed-forward, and residual connections."""
        # Pre-norm Transformer block
        x = x + self.dropout(self.attn(self.ln1(x)))
        x = x + self.dropout(self.ff(self.ln2(x)))
        return x
