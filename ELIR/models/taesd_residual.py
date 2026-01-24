# =============================================================================
# Source: https://github.com/madebyollin/taesd/blob/main/taesd.py
# License: MIT License
#
# Attribution:
# This file was sourced from the repository "taesd",
# available at https://github.com/cszn/BSRGAN. Licensed under the MIT License: https://github.com/madebyollin/taesd/blob/main/LICENSE
# =============================================================================

#!/usr/bin/env python3
"""
Tiny AutoEncoder for Stable Diffusion
(DNN for encoding / decoding SD's latent space)
"""
from typing import Optional, List

import torch
import torch.nn as nn


def conv(n_in, n_out, **kwargs):
    return nn.Conv2d(n_in, n_out, 3, padding=1, **kwargs)

class Block(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.conv = nn.Sequential(conv(n_in, n_out), nn.ReLU(), conv(n_out, n_out), nn.ReLU(), conv(n_out, n_out))
        self.skip = nn.Conv2d(n_in, n_out, 1, bias=False) if n_in != n_out else nn.Identity()
        self.fuse = nn.ReLU()
    def forward(self, x):
        return self.fuse(self.conv(x) + self.skip(x))

# class Encoder(nn.Module):
#     def __init__(self, latent_channels):
#         """Initialize pretrained TAESD on the given device from the given checkpoints."""
#         super(Encoder, self).__init__()
#         self.layers = nn.Sequential(
#                         conv(3, 64), Block(64, 64),
#                         conv(64, 64, stride=2, bias=False), Block(64, 64), Block(64, 64), Block(64, 64),
#                         conv(64, 64, stride=2, bias=False), Block(64, 64), Block(64, 64), Block(64, 64),
#                         conv(64, 64, stride=2, bias=False), Block(64, 64), Block(64, 64), Block(64, 64),
#                         conv(64, latent_channels))
#     def forward(self, x):
#         x = self.layers(x)
#         return x
#
#     def load_weights(self, path):
#         if path is not None:
#             state_dict = torch.load(path, weights_only=True)
#             if path.endswith(".ckpt"):
#                 sd_enc = state_dict["state_dict_enc"]
#                 self.load_state_dict(sd_enc)
#             else:
#                 self.load_state_dict(state_dict)


class Clamp(nn.Module):
    @staticmethod
    def forward(x):
        return torch.tanh(x / 3) * 3

class SkipFusion(nn.Module):
    """
    Fuses encoder features with decoder features at the same resolution.

    Designed to be initialized to zero/identity so existing weights work unchanged.
    The fusion can then be gradually learned during fine-tuning.
    """

    def __init__(self, channels: int = 64, init_zero: bool = True):
        super().__init__()
        self.encoder_proj = nn.Sequential(
            conv(channels, channels),
            nn.ReLU(),
            conv(channels, channels)
        )
        self.fusion_weight = nn.Parameter(torch.zeros(1))

        if init_zero:
            self._init_zero()

    def _init_zero(self):
        for m in self.encoder_proj.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.zeros_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, decoder_feat: torch.Tensor, encoder_feat: Optional[torch.Tensor] = None) -> torch.Tensor:
        if encoder_feat is None:
            return decoder_feat
        enc_processed = self.encoder_proj(encoder_feat)
        fused = decoder_feat + self.fusion_weight * enc_processed
        return fused


class Decoder(nn.Module):
    """
    Decoder that can receive skip connections from encoder.
    Uses self.layers to match your existing weight structure.

    Architecture:
    layers[0]: conv(latent_channels, 64)
    layers[1]: nn.ReLU()  -- no weights
    layers[2]: Block(64, 64)
    layers[3]: Block(64, 64)
    layers[4]: Block(64, 64)
    layers[5]: nn.Upsample(scale_factor=2)  -- no weights
    layers[6]: conv(64, 64, bias=False)
    layers[7]: Block(64, 64)
    layers[8]: Block(64, 64)
    layers[9]: Block(64, 64)
    layers[10]: nn.Upsample(scale_factor=2)  -- no weights
    layers[11]: conv(64, 64, bias=False)
    layers[12]: Block(64, 64)
    layers[13]: Block(64, 64)
    layers[14]: Block(64, 64)
    layers[15]: nn.Upsample(scale_factor=2)  -- no weights
    layers[16]: conv(64, 64, bias=False)
    layers[17]: Block(64, 64)
    layers[18]: conv(64, 3)
    """

    def __init__(self, latent_channels: int = 4, enable_skip: bool = True, init_skip_zero: bool = True):
        super().__init__()

        self.enable_skip = enable_skip
        self.clamp = Clamp()

        # Use ModuleList with name 'layers' to match weight structure
        self.layers = nn.ModuleList([
            conv(latent_channels, 64),  # 0
            nn.ReLU(),  # 1 (no weights)
            Block(64, 64),  # 2
            Block(64, 64),  # 3
            Block(64, 64),  # 4
            nn.Upsample(scale_factor=2),  # 5 (no weights)
            conv(64, 64, bias=False),  # 6
            Block(64, 64),  # 7
            Block(64, 64),  # 8
            Block(64, 64),  # 9
            nn.Upsample(scale_factor=2),  # 10 (no weights)
            conv(64, 64, bias=False),  # 11
            Block(64, 64),  # 12
            Block(64, 64),  # 13
            Block(64, 64),  # 14
            nn.Upsample(scale_factor=2),  # 15 (no weights)
            conv(64, 64, bias=False),  # 16
            Block(64, 64),  # 17
            conv(64, 3),  # 18
        ])

        # Skip fusion modules (NEW - not in original weights)
        if enable_skip:
            self.skip_fusion_3 = SkipFusion(64, init_zero=init_skip_zero)  # at H/8
            self.skip_fusion_2 = SkipFusion(64, init_zero=init_skip_zero)  # at H/4
            self.skip_fusion_1 = SkipFusion(64, init_zero=init_skip_zero)  # at H/2

        # Indices where we apply skip fusion (after the block stages)
        self._skip_apply_indices = {
            'skip_3': 4,  # After blocks at H/8, before upsample
            'skip_2': 9,  # After blocks at H/4, before upsample
            'skip_1': 14,  # After blocks at H/2, before upsample
        }

    def forward(
            self,
            x: torch.Tensor,
            encoder_skips: Optional[List[torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Args:
            latent: Latent tensor (B, latent_channels, H/8, W/8)
            encoder_skips: Optional list [skip_1, skip_2, skip_3] at (H/2, H/4, H/8)
        """
        skip_1, skip_2, skip_3 = (None, None, None)
        if encoder_skips is not None and self.enable_skip:
            skip_1, skip_2, skip_3 = encoder_skips

        # x = self.clamp(latent)

        for i, layer in enumerate(self.layers):
            x = layer(x)

            # Apply skip connections after specific layers
            if self.enable_skip:
                if i == self._skip_apply_indices['skip_3']:
                    x = self.skip_fusion_3(x, skip_3)
                elif i == self._skip_apply_indices['skip_2']:
                    x = self.skip_fusion_2(x, skip_2)
                elif i == self._skip_apply_indices['skip_1']:
                    x = self.skip_fusion_1(x, skip_1)

        return x


class Encoder(nn.Module):
    """
    Encoder that returns intermediate features at multiple resolutions.
    Uses self.layers to match your existing weight structure.

    Architecture:
    layers[0]: conv(3, 64)
    layers[1]: Block(64, 64)
    layers[2]: conv(64, 64, stride=2)  # downsample to H/2
    layers[3]: Block(64, 64)
    layers[4]: Block(64, 64)
    layers[5]: Block(64, 64)
    layers[6]: conv(64, 64, stride=2)  # downsample to H/4
    layers[7]: Block(64, 64)
    layers[8]: Block(64, 64)
    layers[9]: Block(64, 64)
    layers[10]: conv(64, 64, stride=2)  # downsample to H/8
    layers[11]: Block(64, 64)
    layers[12]: Block(64, 64)
    layers[13]: Block(64, 64)
    layers[14]: conv(64, latent_channels)
    """

    def __init__(self, latent_channels: int = 4):
        super().__init__()

        # Use ModuleList with name 'layers' to match weight structure
        self.layers = nn.ModuleList([
            conv(3, 64),  # 0
            Block(64, 64),  # 1
            conv(64, 64, stride=2, bias=False),  # 2 - downsample
            Block(64, 64),  # 3
            Block(64, 64),  # 4
            Block(64, 64),  # 5
            conv(64, 64, stride=2, bias=False),  # 6 - downsample
            Block(64, 64),  # 7
            Block(64, 64),  # 8
            Block(64, 64),  # 9
            conv(64, 64, stride=2, bias=False),  # 10 - downsample
            Block(64, 64),  # 11
            Block(64, 64),  # 12
            Block(64, 64),  # 13
            conv(64, latent_channels),  # 14
        ])

        # Indices for tracking skip connection extraction points
        self._skip_indices = {
            'after_down1': 5,  # After stage at H/2
            'after_down2': 9,  # After stage at H/4
            'after_down3': 13,  # After stage at H/8 (before to_latent)
        }

    def forward(self, x: torch.Tensor, return_intermediates: bool = False):
        """
        Args:
            x: Input image (B, 3, H, W)
            return_intermediates: If True, return features for skip connections

        Returns:
            If return_intermediates=False: latent tensor
            If return_intermediates=True: (latent, [skip_1, skip_2, skip_3])
        """
        intermediates = {}

        for i, layer in enumerate(self.layers):
            x = layer(x)

            # Store intermediates at skip points
            if return_intermediates:
                if i == self._skip_indices['after_down1']:
                    intermediates['skip_1'] = x  # H/2
                elif i == self._skip_indices['after_down2']:
                    intermediates['skip_2'] = x  # H/4
                elif i == self._skip_indices['after_down3']:
                    intermediates['skip_3'] = x  # H/8

        if return_intermediates:
            return x, [intermediates['skip_1'], intermediates['skip_2'], intermediates['skip_3']]
        return x



    def load_weights(self, path):
        if path is not None:
            state_dict = torch.load(path, weights_only=True)
            if path.endswith(".ckpt"):
                sd_enc = state_dict["state_dict_enc"]
                self.load_state_dict(sd_enc)
            else:
                self.load_state_dict(state_dict)


# class Decoder(nn.Module):
#     def __init__(self, latent_channels, up_mode="nearest"):
#         """Initialize pretrained TAESD on the given device from the given checkpoints."""
#         super(Decoder, self).__init__()
#         self.layers = nn.Sequential(conv(latent_channels, 64), nn.ReLU(),
#                         Block(64, 64), Block(64, 64), Block(64, 64), nn.Upsample(scale_factor=2, mode=up_mode), conv(64, 64, bias=False),
#                         Block(64, 64), Block(64, 64), Block(64, 64), nn.Upsample(scale_factor=2, mode=up_mode), conv(64, 64, bias=False),
#                         Block(64, 64), Block(64, 64), Block(64, 64), nn.Upsample(scale_factor=2, mode=up_mode), conv(64, 64, bias=False),
#                         Block(64, 64), conv(64, 3))
#     def forward(self, x):
#         x = torch.tanh(x / 3) * 3
#         return self.layers(x)
#
#     def load_weights(self, path):
#         if path is not None:
#             state_dict = torch.load(path, weights_only=True)
#             if path.endswith(".ckpt"):
#                 sd_dec = state_dict["state_dict_dec"]
#                 self.load_state_dict(sd_dec)
#             else:
#                 self.load_state_dict(state_dict)

class TAESD(nn.Module):

    def __init__(self, pretrained=True, latent_channels=16, up_mode='nearest', enable_skip=True):
        """Initialize pretrained TAESD on the given device from the given checkpoints."""
        super().__init__()
        self.enable_skip = enable_skip
        self.encoder = Encoder(latent_channels)
        self.decoder = Decoder(latent_channels, up_mode)
        if pretrained:
            from diffusers import AutoencoderTiny
            pretrained = AutoencoderTiny.from_pretrained("madebyollin/taesd3")
            self.load_state_dict(pretrained.state_dict(), strict=not enable_skip)

    def load_weights(self, path):
        if path is not None:
            state_dict = torch.load(path, weights_only=True)
            if path.endswith(".ckpt"):
                sd_enc = state_dict["state_dict_enc"]
                self.encoder.load_state_dict(sd_enc)
                sd_dec = state_dict["state_dict_dec"]
                self.decoder.load_state_dict(sd_dec)
            else:
                self.load_state_dict(state_dict)

    def forward(self, x):
        '''
        Input in range [0,1]
        '''
        self.to(x.device)
        return self.decoder(*self.encoder(x, return_intermediates=self.enable_skip))