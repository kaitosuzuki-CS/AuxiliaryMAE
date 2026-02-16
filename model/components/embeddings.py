import numpy as np
import torch
import torch.nn as nn


class PositionalEmbedding(nn.Module):
    def __init__(self, embed_dim, max_len=128):
        super(PositionalEmbedding, self).__init__()

        self._embed_dim = embed_dim
        self._max_len = max_len

        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * (-np.log(10000.0) / embed_dim)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, start_idx, end_idx):
        return self.pe[:, start_idx:end_idx, :]  # type: ignore


class EncoderEmbedding(nn.Module):
    def __init__(self, hps):
        super(EncoderEmbedding, self).__init__()

        self._hps = hps

        self.patch_h, self.patch_w = hps.patch_size
        self.input_proj = nn.Sequential(
            nn.Linear(self.patch_h * self.patch_w, hps.embed_dim),
            nn.GELU(),
        )

        self.pad_value = nn.Parameter(torch.zeros(1))
        self.cls_token = nn.Parameter(torch.randn(1, hps.channels, hps.embed_dim))

        self.pw_pos_embed = PositionalEmbedding(hps.embed_dim, hps.max_len)
        self.cw_pos_embed = PositionalEmbedding(hps.embed_dim, hps.channels)

    def pad(self, x):
        B, C, H, W = x.shape

        h = np.ceil(H / self.patch_h) * self.patch_h
        w = np.ceil(W / self.patch_w) * self.patch_w

        pad_h = int(self.patch_w * h - H)
        pad_w = int(self.patch_h * w - W)

        if pad_h > 0:
            pad_tensor = self.pad_value.expand(B, C, pad_h, W + pad_w)
            x = torch.cat([x, pad_tensor], dim=2)

        if pad_w > 0:
            pad_tensor = self.pad_value.expand(B, C, H, pad_w)
            x = torch.cat([x, pad_tensor], dim=3)

        return x

    def patchify(self, x):
        B, C, H, W = x.shape

        num_h = H // self.patch_h
        num_w = W // self.patch_w

        N = num_h * num_w
        D = self.patch_w * self.patch_h

        x = x.view(B, C, num_h, self.patch_h, num_w, self.patch_w)
        x = x.permute(0, 1, 2, 4, 3, 5)
        x = x.reshape(B, C, N, D).contiguous()

        return x, (num_h, num_w)

    def add_pe(self, x):
        B, C, N, D = x.shape

        x = x.reshape(B * C, N, D)
        x = x + self.pw_pos_embed(1, N + 1)

        x = x.reshape(B, C, N, D)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.reshape(B * N, C, D)

        x = x + self.cw_pos_embed(0, C)

        x = x.reshape(B, N, C, D)
        x = x.permute(0, 2, 1, 3).contiguous()

        return x  # B, C, N, D

    def add_cls_token(self, x):
        B, C, N, D = x.shape

        cls_token = self.cls_token
        cls_token = cls_token + self.cw_pos_embed(0, C)
        cls_token = cls_token + self.pw_pos_embed(0, 1)
        cls_token = cls_token.unsqueeze(-2).repeat(B, 1, 1, 1)

        x = torch.cat([cls_token, x], dim=2)

        return x  # B, C, N+1, D

    def embed(self, x):
        orig_shape = x.shape

        x = self.pad(x)
        x_patches, patch_count = self.patchify(x)

        x = x_patches.clone()
        x = self.input_proj(x)
        x = self.add_pe(x)

        return x, orig_shape, patch_count, x_patches


class DecoderEmbedding(nn.Module):
    def __init__(self, hps):
        super(DecoderEmbedding, self).__init__()

        self._hps = hps

        self.mask_token = nn.Parameter(torch.randn(1, hps.channels, 1, hps.embed_dim))
        self.proj = nn.Linear(hps.embed_dim, hps.embed_dim)

        self.pw_pos_embed = PositionalEmbedding(hps.embed_dim, hps.max_len)
        self.cw_pos_embed = PositionalEmbedding(hps.embed_dim, hps.channels)

    def add_mask(self, x, ids_restore):
        B, C, N, D = x.shape

        x = x.reshape(B * C, N, D)
        mask_tokens = self.mask_token.repeat(
            x.shape[0], 1, ids_restore.shape[1] + 1 - x.shape[1], 1
        )
        mask_tokens = mask_tokens.reshape(B * C, -1, D)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_ = torch.gather(
            x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2])
        )
        x = torch.cat([x[:, :1, :], x_], dim=1)

        x = x.reshape(B, C, -1, D)

        return x

    def add_pe(self, x):
        B, C, N, D = x.shape

        x = x.reshape(B * C, N, D)
        x = x + self.pw_pos_embed(0, N)

        x = x.reshape(B, C, N, D)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.reshape(B * N, C, D)

        x = x + self.cw_pos_embed(0, C)

        x = x.reshape(B, N, C, D)
        x = x.permute(0, 2, 1, 3).contiguous()

        return x  # B, C, N, D

    def forward(self, x, ids_restore):
        x = self.proj(x)
        x = self.add_mask(x, ids_restore)
        x = self.add_pe(x)

        return x


class ReverseEmbedding(nn.Module):
    def __init__(self, hps):
        super(ReverseEmbedding, self).__init__()

        self._hps = hps

        self.patch_h, self.patch_w = hps.patch_size
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hps.embed_dim),
            nn.Linear(hps.embed_dim, self.patch_h * self.patch_w),
            nn.Sigmoid(),
        )

    def forward(self, x, orig_shape, patch_count):
        B, C, N, D = x.shape
        _, _, H, W = orig_shape
        num_h, num_w = patch_count

        x = x[:, :, 1:, :]
        x_patches = self.output_proj(x)

        x = x_patches.clone()
        x = x.reshape(B, C, num_h, num_w, self.patch_h, self.patch_w)
        x = x.permute(0, 1, 2, 4, 3, 5).contiguous()
        x = x.reshape(B, C, num_h * self.patch_h, num_w * self.patch_w)
        x = x[:, :, :H, :W]

        return x, x_patches
