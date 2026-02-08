import torch
from torch import nn

from models.attention import RelativeMultiHeadAttention
from models.transformer_layers import EncoderBlock


class MusicTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        d_ff: int = 256,
        dropout: float = 0.1,
        max_relative_position: int = 64,
        local_window: int | None = None,
    ):
        """Transformer decoder with relative positional self-attention."""
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.dropout = nn.Dropout(dropout)

        layers = []
        for _ in range(num_layers):
            attn = RelativeMultiHeadAttention(
                d_model, num_heads, dropout, max_relative_position, local_window
            )
            layers.append(EncoderBlock(attn, d_model, d_ff, dropout))
        self.layers = nn.ModuleList(layers)
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute logits for next-token prediction."""
        # x: (B, L)
        h = self.token_emb(x)
        h = self.dropout(h)
        for layer in self.layers:
            h = layer(h)
        h = self.ln_f(h)
        return self.head(h)
