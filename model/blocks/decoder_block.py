import torch
import torch.nn as nn

from model.layers.decoder_layer import DecoderLayer


class DecoderBlock(nn.Module):
    def __init__(self, hps):
        super(DecoderBlock, self).__init__()

        self._hps = hps

        self.layers = nn.ModuleList(
            [
                DecoderLayer(hps.embed_dim, hps.hidden_dim, hps.num_heads, hps.dropout)
                for _ in range(hps.num_layers)
            ]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x
