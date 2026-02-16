import torch
import torch.nn as nn

from model.blocks.decoder_block import DecoderBlock
from model.components.embeddings import DecoderEmbedding, ReverseEmbedding


class Decoder(nn.Module):
    def __init__(self, hps):
        super(Decoder, self).__init__()

        self._hps = hps

        self.embedding = DecoderEmbedding(hps)
        self.block = DecoderBlock(hps)
        self.reverse_embedding = ReverseEmbedding(hps)

    def forward(self, x, ids_restore, orig_shape, patch_count):
        x = self.embedding(x, ids_restore)
        x = self.block(x)
        x, x_patches = self.reverse_embedding(x, orig_shape, patch_count)

        return x, x_patches
