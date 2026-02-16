import torch
import torch.nn as nn

from model.models.decoder import Decoder
from model.models.encoder import Encoder


class FactorizedAttentionViT(nn.Module):
    def __init__(self, hps):
        super(FactorizedAttentionViT, self).__init__()

        self._hps = hps

        self.encoder = Encoder(hps.encoder)
        self.decoder = Decoder(hps.decoder)

        self.mask_ratio = hps.mask_ratio

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        print(f"Total Parameters: {sum(p.numel() for p in self.parameters())}")
        print(
            f"Trainable Parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad)}"
        )

    def random_masking(self, x):
        B, C, N, D = x.shape

        x = x.reshape(B * C, N, D)
        len_keep = int(N * (1 - self.mask_ratio))

        noise = torch.rand(B * C, N, device=x.device)

        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        mask = torch.ones([B * C, N], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        mask = mask.reshape(B, C, -1)

        x_masked = x_masked.reshape(B, C, -1, D)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x):
        x, orig_shape, patch_count, x_patches = self.encoder.embed(x)

        x_masked, mask, ids_restore = self.random_masking(x)

        x_masked = self.encoder.add_cls_token(x_masked)
        x_masked = self.encoder(x_masked)

        return x_masked, mask, ids_restore, orig_shape, patch_count, x_patches

    def forward_decoder(self, x, ids_restore, orig_shape, patch_count):
        x, x_patches = self.decoder(x, ids_restore, orig_shape, patch_count)

        return x, x_patches

    def forward(self, x):
        x_masked, mask, ids_restore, orig_shape, patch_count, x_patches = (
            self.forward_encoder(x)
        )
        x_pred, x_pred_patches = self.forward_decoder(
            x_masked, ids_restore, orig_shape, patch_count
        )

        return x_pred, x_pred_patches, x_patches, mask
