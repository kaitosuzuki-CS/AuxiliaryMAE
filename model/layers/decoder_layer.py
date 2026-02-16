import torch
import torch.nn as nn

from model.components.layers import FactorizedAttentionLayer, LinearLayer


class DecoderLayer(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_heads, dropout):
        super(DecoderLayer, self).__init__()

        self._embed_dim = embed_dim
        self._hidden_dim = hidden_dim
        self._num_heads = num_heads
        self._dropout = dropout

        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn1 = LinearLayer(embed_dim, hidden_dim, dropout)
        self.dropout1 = nn.Dropout(dropout)

        self.attn = FactorizedAttentionLayer(embed_dim, num_heads, dropout)

        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn2 = LinearLayer(embed_dim, hidden_dim, dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        B, C, N, D = x.shape

        _x = self.norm1(x)
        _x = self.ffn1(_x)
        x = x + 0.5 * self.dropout1(_x)

        x = self.attn(x)

        _x = self.norm2(x)
        _x = self.ffn2(_x)
        x = x + 0.5 * self.dropout2(_x)

        return x
