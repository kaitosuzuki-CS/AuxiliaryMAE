import torch
import torch.nn as nn

from model.layers.encoder_layer import EncoderLayer


class EncoderBlock(nn.Module):
    def __init__(self, hps):
        super(EncoderBlock, self).__init__()

        self._hps = hps

        self.layers = nn.ModuleList(
            [
                EncoderLayer(hps.embed_dim, hps.hidden_dim, hps.num_heads, hps.dropout)
                for _ in range(hps.num_layers)
            ]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x
