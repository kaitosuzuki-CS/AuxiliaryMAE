import torch
import torch.nn as nn


class LinearLayer(nn.Module):
    def __init__(self, embed_dim, hidden_dim, dropout=0):
        super(LinearLayer, self).__init__()

        self._embed_dim = embed_dim
        self._hidden_dim = hidden_dim
        self._dropout = dropout

        self.layer = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x):
        return self.layer(x)


class FactorizedAttentionLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0):
        super(FactorizedAttentionLayer, self).__init__()

        self._embed_dim = embed_dim
        self._num_heads = num_heads
        self._dropout = dropout

        self.norm_pw = nn.LayerNorm(embed_dim)
        self.attn_pw = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.dropout_pw = nn.Dropout(dropout)

        self.norm_cw = nn.LayerNorm(embed_dim)
        self.attn_cw = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.dropout_cw = nn.Dropout(dropout)

    def forward(self, x):
        B, C, N, D = x.shape

        x = x.reshape(B * C, N, D)
        _x = self.norm_pw(x)
        _x, _ = self.attn_pw(_x, _x, _x)
        x = x + self.dropout_pw(_x)

        x = x.reshape(B, C, N, D)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.reshape(B * N, C, D)

        _x = self.norm_cw(x)
        _x, _ = self.attn_cw(_x, _x, _x)
        x = x + self.dropout_cw(_x)

        x = x.reshape(B, N, C, D)
        x = x.permute(0, 2, 1, 3).contiguous()

        return x  # B, C, N, D
