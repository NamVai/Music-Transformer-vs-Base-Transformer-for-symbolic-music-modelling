import torch
from torch import nn

from models.attention import StandardMultiHeadAttention
from models.transformer_layers import EncoderBlock
from utils.positional_encoding import SinusoidalPositionalEncoding


class TransformerModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        d_ff: int = 256,
        dropout: float = 0.1,
        max_len: int = 4096,
    ):
        """Transformer decoder with absolute sinusoidal positional encodings."""
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=max_len)
        self.dropout = nn.Dropout(dropout)

        layers = []
        for _ in range(num_layers):
            attn = StandardMultiHeadAttention(d_model, num_heads, dropout)
            layers.append(EncoderBlock(attn, d_model, d_ff, dropout))
        self.layers = nn.ModuleList(layers)
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute logits for next-token prediction."""
        # x: (B, L)
        h = self.token_emb(x)
        h = self.pos_enc(h)
        h = self.dropout(h)
        for layer in self.layers:
            h = layer(h)
        h = self.ln_f(h)
        return self.head(h)  # (B, L, V)
