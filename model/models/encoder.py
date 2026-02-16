import torch
import torch.nn as nn

from model.blocks.encoder_block import EncoderBlock
from model.components.embeddings import EncoderEmbedding


class Encoder(nn.Module):
    def __init__(self, hps):
        super(Encoder, self).__init__()

        self._hps = hps

        self.embedding = EncoderEmbedding(hps)
        self.block = EncoderBlock(hps)

    def add_cls_token(self, x):
        return self.embedding.add_cls_token(x)

    def embed(self, x):
        x, orig_shape, patch_count, x_patches = self.embedding.embed(x)
        return x, orig_shape, patch_count, x_patches

    def forward(self, x):
        return self.block(x)
