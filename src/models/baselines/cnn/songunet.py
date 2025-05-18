import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import List

# ----------------------------------------------------------------------------
# Simplified convolutional and normalization blocks
# ----------------------------------------------------------------------------
class Conv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size=3, stride=1, padding=1,
                 up: bool = False, down: bool = False):
        super().__init__()
        if up:
            self.layer = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        elif down:
            self.layer = nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2)
        else:
            self.layer = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)

class GroupNorm(nn.Module):
    def __init__(self, num_channels: int, num_groups: int = 8, eps: float = 1e-5):
        super().__init__()
        g = min(num_groups, num_channels)
        while g > 1 and (num_channels % g) != 0:
            g -= 1
        self.norm = nn.GroupNorm(g, num_channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)

# ----------------------------------------------------------------------------
# Core UNet block with optional self-attention
# ----------------------------------------------------------------------------
class UNetBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int,
                 up: bool = False, down: bool = False, attention: bool = False):
        super().__init__()
        self.down = down
        self.up = up
        self.attn = attention

        self.norm1 = GroupNorm(in_ch)
        self.conv1 = Conv2d(in_ch, out_ch, up=up, down=down)
        self.act   = nn.SiLU()
        self.norm2 = GroupNorm(out_ch)
        self.conv2 = Conv2d(out_ch, out_ch)

        if attention:
            self.attn_layer = nn.MultiheadAttention(out_ch, num_heads=4, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        h = self.act(self.norm2(h))
        h = self.conv2(h)

        if self.attn:
            b, c, hh, ww = h.shape
            seq = h.view(b, c, hh*ww).permute(0, 2, 1)
            attn_out, _ = self.attn_layer(seq, seq, seq)
            h = attn_out.permute(0, 2, 1).view(b, c, hh, ww)

        # only apply residual skip if no spatial shape change
        if not (self.down or self.up):
            return x + h
        return h

# ----------------------------------------------------------------------------
# Configuration dataclass for channel widths and attention
# ----------------------------------------------------------------------------
@dataclass
class ModelConfig:
    model_channels: int = 128
    channel_mult: List[int] = (1, 2, 2, 2)
    attn_resolutions: List[int] = ()

# ----------------------------------------------------------------------------
# Main UNet adapted for super-resolution / dense regression
# ----------------------------------------------------------------------------
class SongUNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        config: ModelConfig = ModelConfig(),
    ):
        super().__init__()
        mc = config.model_channels
        cm = config.channel_mult
        attn = config.attn_resolutions

        # Keep track of encoder channel sizes for skip connections
        self.enc_channels = [mc * mult for mult in cm]

        # Encoder path
        self.enc_blocks = nn.ModuleList()
        ch = in_channels
        for mult in cm:
            out_ch = mc * mult
            self.enc_blocks.append(
                UNetBlock(ch, out_ch, down=True, attention=(mult in attn))
            )
            ch = out_ch

        # Bottleneck
        self.mid = UNetBlock(ch, ch, attention=True)

        # Decoder path
        self.dec_blocks = nn.ModuleList()
        # ch holds bottleneck output channels
        for idx, mult in enumerate(reversed(cm)):
            out_ch = mc * mult
            if idx == 0:
                in_ch = ch
            else:
                # concat skip from corresponding encoder stage
                skip_ch = self.enc_channels[-idx-1]
                in_ch = ch + skip_ch
            self.dec_blocks.append(
                UNetBlock(in_ch, out_ch, up=True, attention=(mult in attn))
            )
            ch = out_ch

        # Final projection
        self.final_conv = nn.Conv2d(ch, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc_outputs = []
        h = x
        # encode
        for block in self.enc_blocks:
            h = block(h)
            enc_outputs.append(h)
        # bottleneck
        h = self.mid(h)
        # decode
        for idx, block in enumerate(self.dec_blocks):
            h = block(h)
            # skip concat for all but last decoder block
            if idx < len(self.dec_blocks) - 1:
                skip = enc_outputs[-idx-2]
                h = torch.cat([h, skip], dim=1)
        return self.final_conv(h)

# ----------------------------------------------------------------------------
# Wrapper to prepend a high-res zero map before events (optional)
# ----------------------------------------------------------------------------
class EventDepthUNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        config: ModelConfig = ModelConfig(),
    ):
        super().__init__()
        # one extra channel for the zero‐init high‐res map
        self.net = SongUNet(in_channels + 1, out_channels, config)

    def forward(self, event: torch.Tensor) -> torch.Tensor:
        # event: (B, C_event, H, W)
        hr = torch.zeros_like(event[:, :1, ...])
        x  = torch.cat([hr, event], dim=1)
        return self.net(x)