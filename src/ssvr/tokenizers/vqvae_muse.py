import math
from functools import partial
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn


class Upsample(nn.Module):
    def __init__(self, in_channels: int, with_conv: bool):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, hidden_states):
        hidden_states = F.interpolate(hidden_states, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            hidden_states = self.conv(hidden_states)
        return hidden_states


class Downsample(nn.Module):
    def __init__(self, in_channels: int, with_conv: bool):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, hidden_states):
        if self.with_conv:
            pad = (0, 1, 0, 1)
            hidden_states = F.pad(hidden_states, pad, mode="constant", value=0)
            hidden_states = self.conv(hidden_states)
        else:
            hidden_states = F.avg_pool2d(hidden_states, kernel_size=2, stride=2)
        return hidden_states


class ResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = None, use_conv_shortcut: bool = False, dropout_prob: float = 0.0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.out_channels_ = self.in_channels if self.out_channels is None else self.out_channels
        self.use_conv_shortcut = use_conv_shortcut

        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.conv1 = nn.Conv2d(self.in_channels, self.out_channels_, kernel_size=3, stride=1, padding=1)

        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=self.out_channels_, eps=1e-6, affine=True)
        self.dropout = nn.Dropout(dropout_prob)
        self.conv2 = nn.Conv2d(self.out_channels_, self.out_channels_, kernel_size=3, stride=1, padding=1)

        if self.in_channels != self.out_channels_:
            if use_conv_shortcut:
                self.conv_shortcut = nn.Conv2d(self.in_channels, self.out_channels_, kernel_size=3, stride=1, padding=1)
            else:
                self.nin_shortcut = nn.Conv2d(self.in_channels, self.out_channels_, kernel_size=1, stride=1, padding=0)

    def forward(self, hidden_states):
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = F.silu(hidden_states)
        hidden_states = self.conv1(hidden_states)

        hidden_states = self.norm2(hidden_states)
        hidden_states = F.silu(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.in_channels != self.out_channels_:
            if self.use_conv_shortcut:
                residual = self.conv_shortcut(residual)
            else:
                residual = self.nin_shortcut(residual)
        return hidden_states + residual


class AttnBlock(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels
        conv = partial(nn.Conv2d, self.in_channels, self.in_channels, kernel_size=1, stride=1, padding=0)
        self.norm = nn.GroupNorm(num_groups=32, num_channels=self.in_channels, eps=1e-6, affine=True)
        self.q, self.k, self.v = conv(), conv(), conv()
        self.proj_out = conv()

    def forward(self, hidden_states):
        residual = hidden_states
        hidden_states = self.norm(hidden_states)
        query = self.q(hidden_states)
        key = self.k(hidden_states)
        value = self.v(hidden_states)

        batch, channels, height, width = query.shape
        query = query.reshape((batch, channels, height * width)).permute(0, 2, 1)
        key = key.reshape((batch, channels, height * width))

        attn_weights = torch.bmm(query, key) * (int(channels) ** -0.5)
        attn_weights = F.softmax(attn_weights, dim=2)

        value = value.reshape((batch, channels, height * width))
        attn_weights = attn_weights.permute(0, 2, 1)
        hidden_states = torch.bmm(value, attn_weights)
        hidden_states = hidden_states.reshape((batch, channels, height, width))
        hidden_states = self.proj_out(hidden_states)
        return hidden_states + residual


class MidBlock(nn.Module):
    def __init__(self, config, in_channels: int, no_attn: bool, dropout: float):
        super().__init__()
        self.block_1 = ResnetBlock(in_channels, in_channels, dropout_prob=dropout)
        if not no_attn:
            self.attn_1 = AttnBlock(in_channels)
        self.block_2 = ResnetBlock(in_channels, in_channels, dropout_prob=dropout)

    def forward(self, hidden_states):
        hidden_states = self.block_1(hidden_states)
        if hasattr(self, 'attn_1'):
            hidden_states = self.attn_1(hidden_states)
        hidden_states = self.block_2(hidden_states)
        return hidden_states


class DownsamplingBlock(nn.Module):
    def __init__(self, config, curr_res: int, block_idx: int):
        super().__init__()
        self.config = config
        self.curr_res = curr_res
        self.block_idx = block_idx

        in_channel_mult = (1,) + tuple(self.config.channel_mult)
        block_in = self.config.hidden_channels * in_channel_mult[self.block_idx]
        block_out = self.config.hidden_channels * self.config.channel_mult[self.block_idx]

        res_blocks = nn.ModuleList()
        attn_blocks = nn.ModuleList()
        for _ in range(self.config.num_res_blocks):
            res_blocks.append(ResnetBlock(block_in, block_out, dropout_prob=self.config.dropout))
            block_in = block_out
            if self.curr_res in self.config.attn_resolutions:
                attn_blocks.append(AttnBlock(block_in))

        self.block = res_blocks
        self.attn = attn_blocks
        self.downsample = None
        if self.block_idx != self.config.num_resolutions - 1:
            self.downsample = Downsample(block_in, self.config.resample_with_conv)

    def forward(self, hidden_states):
        for i, res_block in enumerate(self.block):
            hidden_states = res_block(hidden_states)
            if i < len(self.attn):
                hidden_states = self.attn[i](hidden_states)
        if self.downsample is not None:
            hidden_states = self.downsample(hidden_states)
        return hidden_states


class UpsamplingBlock(nn.Module):
    def __init__(self, config, curr_res: int, block_idx: int):
        super().__init__()
        self.config = config
        self.block_idx = block_idx
        self.curr_res = curr_res

        if self.block_idx == self.config.num_resolutions - 1:
            block_in = self.config.hidden_channels * self.config.channel_mult[-1]
        else:
            block_in = self.config.hidden_channels * self.config.channel_mult[self.block_idx + 1]

        block_out = self.config.hidden_channels * self.config.channel_mult[self.block_idx]

        res_blocks = []
        attn_blocks = []
        for _ in range(self.config.num_res_blocks + 1):
            res_blocks.append(ResnetBlock(block_in, block_out, dropout_prob=self.config.dropout))
            block_in = block_out
            if self.curr_res in self.config.attn_resolutions:
                attn_blocks.append(AttnBlock(block_in))

        self.block = nn.ModuleList(res_blocks)
        self.attn = nn.ModuleList(attn_blocks)
        self.upsample = None
        if self.block_idx != 0:
            self.upsample = Upsample(block_in, self.config.resample_with_conv)

    def forward(self, hidden_states):
        for i, res_block in enumerate(self.block):
            hidden_states = res_block(hidden_states)
            if i < len(self.attn):
                hidden_states = self.attn[i](hidden_states)
        if self.upsample is not None:
            hidden_states = self.upsample(hidden_states)
        return hidden_states


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.conv_in = nn.Conv2d(self.config.num_channels, self.config.hidden_channels, kernel_size=3, stride=1, padding=1)

        curr_res = self.config.resolution
        downsample_blocks = []
        for i_level in range(self.config.num_resolutions):
            downsample_blocks.append(DownsamplingBlock(self.config, curr_res, block_idx=i_level))
            if i_level != self.config.num_resolutions - 1:
                curr_res = curr_res // 2
        self.down = nn.ModuleList(downsample_blocks)

        mid_channels = self.config.hidden_channels * self.config.channel_mult[-1]
        self.mid = MidBlock(config, mid_channels, self.config.no_attn_mid_block, self.config.dropout)

        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=mid_channels, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(mid_channels, self.config.z_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, pixel_values):
        hidden_states = self.conv_in(pixel_values)
        for block in self.down:
            hidden_states = block(hidden_states)
        hidden_states = self.mid(hidden_states)
        hidden_states = self.norm_out(hidden_states)
        hidden_states = F.silu(hidden_states)
        hidden_states = self.conv_out(hidden_states)
        return hidden_states


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        block_in = self.config.hidden_channels * self.config.channel_mult[self.config.num_resolutions - 1]
        curr_res = self.config.resolution // 2 ** (self.config.num_resolutions - 1)
        self.z_shape = (1, self.config.z_channels, curr_res, curr_res)

        self.conv_in = nn.Conv2d(self.config.z_channels, block_in, kernel_size=3, stride=1, padding=1)
        self.mid = MidBlock(config, block_in, self.config.no_attn_mid_block, self.config.dropout)

        upsample_blocks = []
        for i_level in reversed(range(self.config.num_resolutions)):
            upsample_blocks.append(UpsamplingBlock(self.config, curr_res, block_idx=i_level))
            if i_level != 0:
                curr_res = curr_res * 2
        self.up = nn.ModuleList(list(reversed(upsample_blocks)))

        block_out = self.config.hidden_channels * self.config.channel_mult[0]
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_out, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(block_out, self.config.num_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, hidden_states):
        hidden_states = self.conv_in(hidden_states)
        hidden_states = self.mid(hidden_states)
        for block in reversed(self.up):
            hidden_states = block(hidden_states)
        hidden_states = self.norm_out(hidden_states)
        hidden_states = F.silu(hidden_states)
        hidden_states = self.conv_out(hidden_states)
        return hidden_states


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(self, hidden_states, return_loss=False):
        hidden_states = hidden_states.permute(0, 2, 3, 1).contiguous()
        distances = self.compute_distances(hidden_states)
        min_encoding_indices = torch.argmin(distances, axis=1).unsqueeze(1)
        min_encodings = torch.zeros(min_encoding_indices.shape[0], self.num_embeddings).to(hidden_states)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        z_q = torch.matmul(min_encodings, self.embedding.weight).view(hidden_states.shape)
        min_encoding_indices = min_encoding_indices.reshape(hidden_states.shape[0], -1)

        loss = None
        if return_loss:
            loss = torch.mean((z_q.detach() - hidden_states) ** 2) + self.commitment_cost * torch.mean(
                (z_q - hidden_states.detach()) ** 2
            )
            z_q = hidden_states + (z_q - hidden_states).detach()

        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        return z_q, min_encoding_indices, loss

    def compute_distances(self, hidden_states):
        hidden_states_flattended = hidden_states.reshape((-1, self.embedding_dim))
        emb_weights = self.embedding.weight.t()
        inputs_norm_sq = hidden_states_flattended.pow(2.0).sum(dim=1, keepdim=True)
        codebook_t_norm_sq = emb_weights.pow(2.0).sum(dim=0, keepdim=True)
        distances = torch.addmm(
            inputs_norm_sq + codebook_t_norm_sq,
            hidden_states_flattended,
            emb_weights,
            alpha=-2.0,
        )
        return distances

    def get_codebook_entry(self, indices):
        batch, num_tokens = indices.shape
        z_q = self.embedding(indices)
        z_q = z_q.reshape(batch, int(math.sqrt(num_tokens)), int(math.sqrt(num_tokens)), -1).permute(0, 3, 1, 2)
        return z_q

    def get_code(self, hidden_states):
        hidden_states = hidden_states.permute(0, 2, 3, 1).contiguous()
        distances = self.compute_distances(hidden_states)
        indices = torch.argmin(distances, axis=1).unsqueeze(1)
        indices = indices.reshape(hidden_states.shape[0], -1)
        return indices


class VQGANModel(nn.Module):

    def __init__(
        self,
        resolution: int = 256,
        num_channels: int = 3,
        hidden_channels: int = 128,
        channel_mult: Tuple = (1, 2, 2, 4, 6),
        num_res_blocks: int = 2,
        attn_resolutions: list = None,
        no_attn_mid_block: bool = True,
        z_channels: int = 64,
        num_embeddings: int = 8192,
        quantized_embed_dim: int = 64,
        dropout: float = 0.0,
        resample_with_conv: bool = True,
        commitment_cost: float = 0.25,
    ):
        super().__init__()


        self.config = type('Config', (), {
            'resolution': resolution,
            'num_channels': num_channels,
            'hidden_channels': hidden_channels,
            'channel_mult': list(channel_mult),
            'num_res_blocks': num_res_blocks,
            'attn_resolutions': attn_resolutions or [],
            'no_attn_mid_block': no_attn_mid_block,
            'z_channels': z_channels,
            'num_embeddings': num_embeddings,
            'quantized_embed_dim': quantized_embed_dim,
            'dropout': dropout,
            'resample_with_conv': resample_with_conv,
            'commitment_cost': commitment_cost,
            'num_resolutions': len(channel_mult),
            'reduction_factor': 2 ** (len(channel_mult) - 1),
            'latent_size': resolution // (2 ** (len(channel_mult) - 1)),
        })()

        self.encoder = Encoder(self.config)
        self.decoder = Decoder(self.config)
        self.quantize = VectorQuantizer(num_embeddings, quantized_embed_dim, commitment_cost)
        self.quant_conv = nn.Conv2d(z_channels, quantized_embed_dim, kernel_size=1)
        self.post_quant_conv = nn.Conv2d(quantized_embed_dim, z_channels, kernel_size=1)

    def encode(self, pixel_values, return_loss=False):
        hidden_states = self.encoder(pixel_values)
        hidden_states = self.quant_conv(hidden_states)
        quantized_states, codebook_indices, codebook_loss = self.quantize(hidden_states, return_loss)
        output = (quantized_states, codebook_indices)
        if return_loss:
            output = output + (codebook_loss,)
        return output

    def decode(self, quantized_states):
        hidden_states = self.post_quant_conv(quantized_states)
        return self.decoder(hidden_states)

    def decode_code(self, codebook_indices):
        quantized_states = self.quantize.get_codebook_entry(codebook_indices)
        return self.decode(quantized_states)

    def get_code(self, pixel_values):
        hidden_states = self.encoder(pixel_values)
        hidden_states = self.quant_conv(hidden_states)
        return self.quantize.get_code(hidden_states)

    def forward(self, pixel_values, return_loss=False):
        hidden_states = self.encoder(pixel_values)
        hidden_states = self.quant_conv(hidden_states)
        quantized_states, codebook_indices, codebook_loss = self.quantize(hidden_states, return_loss)
        reconstructed_pixel_values = self.decode(quantized_states)
        outputs = (reconstructed_pixel_values, quantized_states, codebook_indices)
        if return_loss:
            outputs = outputs + (codebook_loss,)
        return outputs


def get_tokenizer_muse():
    from huggingface_hub import hf_hub_download
    import json

    repo_id = "Emma02/vqvae_ckpts"


    config_path = hf_hub_download(repo_id, "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)


    config.pop("_class_name", None)
    config.pop("_version", None)


    model = VQGANModel(**config)


    weights_path = hf_hub_download(repo_id, "pytorch_model.bin")
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state_dict)

    model.eval()
    return model
