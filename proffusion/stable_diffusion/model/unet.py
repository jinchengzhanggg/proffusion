"""
---
title: U-Net for Stable Diffusion
summary: >
 Annotated PyTorch implementation/tutorial of the U-Net in stable diffusion.
---

#  U-Net for [Stable Diffusion](../index.html)

This implements the U-Net that
 gives $\epsilon_\text{cond}(x_t, c)$

We have kept to the model definition and naming unchanged from
[CompVis/stable-diffusion](https://github.com/CompVis/stable-diffusion)
so that we can load the checkpoints directly.
"""

import math
from typing import List

import numpy as np
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba
import sys
sys.path.append('proffusion/stable_diffusion/model/')
# from wavelet_block import LWN
from wavelet_block import WaveletBlock

from .unet_attention import SpatialTransformer

class Mamba2Conv(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2, channel_token=False):
        super().__init__()
        # print(f"MambaLayer: dim: {dim}")
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
            d_model=dim,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand,  # Block expansion factor
        )
        self.channel_token = channel_token  ## whether to use channel as tokens
        # TODO
        # self.cond_proj = nn.Linear(512, dim)

    def forward_patch_token(self, x):
        B, d_model = x.shape[:2]
        # assert d_model == self.dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, d_model, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm)
        # print(x_mamba.shape)
        out = x_mamba.transpose(-1, -2).reshape(B, d_model, *img_dims)

        return out

    # def forward_patch_token(self, x, cond):
        
    #     B, d_model = x.shape[:2]
    #     # assert d_model == self.dim
    #     n_tokens = x.shape[2:].numel()
    #     img_dims = x.shape[2:]
    #     x_flat = x.reshape(B, d_model, n_tokens).transpose(-1, -2)

    #     cond = self.cond_proj(cond).expand(-1, n_tokens, -1)
    #     x_flat = x_flat + cond
        
    #     x_norm = self.norm(x_flat)
    #     x_mamba = self.mamba(x_norm)
    #     # print(x_mamba.shape)
    #     out = x_mamba.transpose(-1, -2).reshape(B, d_model, *img_dims)

    #     return out

    def forward_channel_token(self, x):
        B, n_tokens = x.shape[:2]
        d_model = x.shape[2:].numel()
        assert d_model == self.dim, f"d_model: {d_model}, self.dim: {self.dim}"
        img_dims = x.shape[2:]
        x_flat = x.flatten(2)
        assert x_flat.shape[2] == d_model, f"x_flat.shape[2]: {x_flat.shape[2]}, d_model: {d_model}"
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm)
        out = x_mamba.reshape(B, n_tokens, *img_dims)

        return out

    # @autocast(enabled=False)
    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)

        input_x = x

        if self.channel_token:
            out = self.forward_channel_token(x)
        else:
            out = self.forward_patch_token(x)

        return out + input_x
        # return out


class UNetModel(nn.Module):
    """
    ## U-Net model
    """

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        channels: int,
        n_res_blocks: int,
        attention_levels: List[int],
        channel_multipliers: List[int],
        n_heads: int,
        tf_layers: int = 1,
        d_cond: int = 768,
    ):
        """
        :param in_channels: is the number of channels in the input feature map
        :param out_channels: is the number of channels in the output feature map
        :param channels: is the base channel count for the model
        :param n_res_blocks: number of residual blocks at each level
        :param attention_levels: are the levels at which attention should be performed
        :param channel_multipliers: are the multiplicative factors for number of channels for each level
        :param n_heads: the number of attention heads in the transformers
        """
        super().__init__()
        self.channels = channels

        # Number of levels
        levels = len(channel_multipliers)
        # Size time embeddings
        d_time_emb = channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(channels, d_time_emb),
            nn.SiLU(),
            nn.Linear(d_time_emb, d_time_emb),
        )

        # Input half of the U-Net
        self.input_blocks = nn.ModuleList()
        # Initial $3 \times 3$ convolution that maps the input to `channels`.
        # The blocks are wrapped in `TimestepEmbedSequential` module because
        # different modules have different forward function signatures;
        # for example, convolution only accepts the feature map and
        # residual blocks accept the feature map and time embedding.
        # `TimestepEmbedSequential` calls them accordingly.
        self.input_blocks.append(
            TimestepEmbedSequential(nn.Conv2d(in_channels, channels, 3, padding=1))
        )

        # Number of channels at each block in the input half of U-Net
        input_block_channels = [channels]
        lwn_channels_list =  [channels]
        # Number of channels at each level
        channels_list = [channels * m for m in channel_multipliers]

        # Prepare levels
        for i in range(levels):
            # Add the residual blocks and attentions
            for _ in range(n_res_blocks):
                # Residual block maps from previous number of channels to the number of
                # channels in the current level
                layers = [ResBlock(channels, d_time_emb, out_channels=channels_list[i])]
                channels = channels_list[i]
                # Add transformer
                if i in attention_levels:
                    layers.append(
                        SpatialTransformer(channels, n_heads, tf_layers, d_cond)
                    )
                    # TODO
                    layers.append(Mamba2Conv(channels))
                # Add them to the input half of the U-Net and keep track of the number of channels of
                # its output
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_channels.append(channels)
                lwn_channels_list.append(channels)
            # Down sample at all levels except last
            if i != levels - 1:
                self.input_blocks.append(TimestepEmbedSequential(DownSample(channels)))
                input_block_channels.append(channels)
                lwn_channels_list.append(channels)


        self.wave_blocks = nn.ModuleList()
        for i in range(levels):
            for j in range(n_res_blocks + 1):
                # ---------------------------------------------------------------------
                if i == levels-1:
                    self.wave_blocks.append(nn.Identity())
                else:
                    self.wave_blocks.append(WaveletBlock(c = lwn_channels_list.pop()))
                
                # self.wave_blocks.append(WaveletBlock(c = lwn_channels_list.pop()))
                # ---------------------------------------------------------------------
        self.levels = levels
        self.n_res_blocks = n_res_blocks

        # The middle of the U-Net
        self.middle_block = TimestepEmbedSequential(
            ResBlock(channels, d_time_emb),
            SpatialTransformer(channels, n_heads, tf_layers, d_cond),
            # TODO
            Mamba2Conv(channels),
            ResBlock(channels, d_time_emb),
        )

        # Second half of the U-Net
        self.output_blocks = nn.ModuleList([])
        # Prepare levels in reverse order
        for i in reversed(range(levels)):
            # Add the residual blocks and attentions
            for j in range(n_res_blocks + 1):
                # Residual block maps from previous number of channels plus the
                # skip connections from the input half of U-Net to the number of
                # channels in the current level.
                layers = [
                    ResBlock(
                        channels + input_block_channels.pop(),
                        d_time_emb,
                        out_channels=channels_list[i],
                    )
                ]
                channels = channels_list[i]
                # Add transformer
                if i in attention_levels:
                    layers.append(
                        SpatialTransformer(channels, n_heads, tf_layers, d_cond)
                    )
                    # TODO
                    layers.append(Mamba2Conv(channels))
                # Up-sample at every level after last residual block
                # except the last one.
                # Note that we are iterating in reverse; i.e. `i == 0` is the last.
                if i != 0 and j == n_res_blocks:
                    layers.append(UpSample(channels))
                # Add to the output half of the U-Net
                self.output_blocks.append(TimestepEmbedSequential(*layers))

        # Final normalization and $3 \times 3$ convolution
        self.out = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            nn.Conv2d(channels, out_channels, 3, padding=1),
        )

        # self.mamba_1 = Mamba2Conv(channels)

    def time_step_embedding(self, time_steps: torch.Tensor, max_period: int = 10000):
        """
        ## Create sinusoidal time step embeddings

        :param time_steps: are the time steps of shape `[batch_size]`
        :param max_period: controls the minimum frequency of the embeddings.
        """
        # $\frac{c}{2}$; half the channels are sin and the other half is cos,
        half = self.channels // 2
        # $\frac{1}{10000^{\frac{2i}{c}}}$
        frequencies = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=time_steps.device)
        # $\frac{t}{10000^{\frac{2i}{c}}}$
        args = time_steps[:, None].float() * frequencies[None]
        # $\cos\Bigg(\frac{t}{10000^{\frac{2i}{c}}}\Bigg)$ and $\sin\Bigg(\frac{t}{10000^{\frac{2i}{c}}}\Bigg)$
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    def forward(self, x: torch.Tensor, time_steps: torch.Tensor, cond: torch.Tensor):
        """
        :param x: is the input feature map of shape `[batch_size, channels, width, height]`
        :param time_steps: are the time steps of shape `[batch_size]`
        :param cond: conditioning of shape `[batch_size, n_cond, d_cond]`
        """
        # To store the input half outputs for skip connections
        x_input_block = []

        # Get time step embeddings
        t_emb = self.time_step_embedding(time_steps)
        t_emb = self.time_embed(t_emb)

        # Input half of the U-Net
        for module in self.input_blocks:
            x = module(x, t_emb, cond)
            # x = self.mamba_1(x)
            x_input_block.append(x)
            # print(x_input_block.shape)
        # Middle of the U-Net
        x = self.middle_block(x, t_emb, cond)

        # x_input_block = x_input_block[::-1]

        x_input_block_wave = []
        for module in self.wave_blocks:
            x_wave = module(x_input_block.pop())
            # print(x_wave.shape)
            x_input_block_wave.append(x_wave)

        x_input_block_wave = x_input_block_wave[::-1]

        x_input_block_wave_2 = x_input_block_wave

        # Output half of the U-Net
        for module in self.output_blocks:
            x = th.cat([x, x_input_block_wave.pop()], dim=1)
            x = module(x, t_emb, cond)

        # Final normalization and $3 \times 3$ convolution
        return self.out(x)

    def get_wavelet_loss(self):
        wavelet_loss = 0.
        n = 0

        for i in range(self.levels):
            for j in range(self.n_res_blocks + 1):
                # ---------------------------------------------------------------------
                if hasattr(self.wave_blocks[n], "get_wavelet_loss"):
                    wavelet_loss += self.wave_blocks[n].get_wavelet_loss()
                    
                # wavelet_loss += self.wave_blocks[n].get_wavelet_loss()
                # ---------------------------------------------------------------------
                n = n + 1

        # for index, _ in enumerate(self.num_blocks):
        #     for block in getattr(self, f'd{index+1}'):
        #         wavelet_loss += block.get_wavelet_loss()
        return wavelet_loss


class TimestepEmbedSequential(nn.Sequential):
    """
    ### Sequential block for modules with different inputs

    This sequential module can compose of different modules suck as `ResBlock`,
    `nn.Conv` and `SpatialTransformer` and calls them with the matching signatures
    """

    def forward(self, x, t_emb, cond=None):
        for layer in self:
            if isinstance(layer, ResBlock):
                x = layer(x, t_emb)
            elif isinstance(layer, SpatialTransformer):
                x = layer(x, cond)
            # ----------------------------------------------------------
            # elif isinstance(layer, Mamba2Conv):
            #     x = layer(x, cond)
            # ----------------------------------------------------------
            else:
                x = layer(x)
        return x


class UpSample(nn.Module):
    """
    ### Up-sampling layer
    """

    def __init__(self, channels: int):
        """
        :param channels: is the number of channels
        """
        super().__init__()
        # $3 \times 3$ convolution mapping
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor):
        """
        :param x: is the input feature map with shape `[batch_size, channels, height, width]`
        """
        # Up-sample by a factor of $2$
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        # Apply convolution
        return self.conv(x)


class DownSample(nn.Module):
    """
    ## Down-sampling layer
    """

    def __init__(self, channels: int):
        """
        :param channels: is the number of channels
        """
        super().__init__()
        # $3 \times 3$ convolution with stride length of $2$ to down-sample by a factor of $2$
        self.op = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor):
        """
        :param x: is the input feature map with shape `[batch_size, channels, height, width]`
        """
        # Apply convolution
        return self.op(x)


class ResBlock(nn.Module):
    """
    ## ResNet Block
    """

    def __init__(self, channels: int, d_t_emb: int, *, out_channels=None):
        """
        :param channels: the number of input channels
        :param d_t_emb: the size of timestep embeddings
        :param out_channels: is the number of out channels. defaults to `channels.
        """
        super().__init__()
        # `out_channels` not specified
        if out_channels is None:
            out_channels = channels

        # First normalization and convolution
        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            nn.Conv2d(channels, out_channels, 3, padding=1),
        )

        # Time step embeddings
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_t_emb, out_channels),
        )
        # Final convolution layer
        self.out_layers = nn.Sequential(
            normalization(out_channels),
            nn.SiLU(),
            nn.Dropout(0.0),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )

        # `channels` to `out_channels` mapping layer for residual connection
        if out_channels == channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Conv2d(channels, out_channels, 1)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor):
        """
        :param x: is the input feature map with shape `[batch_size, channels, height, width]`
        :param t_emb: is the time step embeddings of shape `[batch_size, d_t_emb]`
        """
        # Initial convolution
        h = self.in_layers(x)
        # Time step embeddings
        t_emb = self.emb_layers(t_emb).type(h.dtype)
        # Add time step embeddings
        h = h + t_emb[:, :, None, None]
        # Final convolution
        h = self.out_layers(h)
        # Add skip connection
        return self.skip_connection(x) + h


class GroupNorm32(nn.GroupNorm):
    """
    ### Group normalization with float32 casting
    """

    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


def normalization(channels):
    """
    ### Group normalization

    This is a helper function, with fixed number of groups..
    """
    return GroupNorm32(32, channels)


def _test_time_embeddings():
    """
    Test sinusoidal time step embeddings
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(15, 5))
    m = UNetModel(
        in_channels=1,
        out_channels=1,
        channels=320,
        n_res_blocks=1,
        attention_levels=[],
        channel_multipliers=[],
        n_heads=1,
        tf_layers=1,
        d_cond=1,
    )
    te = m.time_step_embedding(torch.arange(0, 1000))
    plt.plot(np.arange(1000), te[:, [50, 100, 190, 260]].numpy())
    plt.legend(["dim %d" % p for p in [50, 100, 190, 260]])
    plt.title("Time embeddings")
    plt.show()


#
if __name__ == "__main__":
    _test_time_embeddings()
