import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34
from typing import Tuple

class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, activation=nn.ReLU()):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.act   = activation
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        return x

class DecoderBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, activation=nn.ReLU()):
        super().__init__()
        self.upconv    = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.conv_block = ConvBlock(out_ch + skip_ch, out_ch, activation)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upconv(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv_block(x)

class ResNetUNet(nn.Module):
    """
    ResNet34‐backboned UNet with custom ConvBlocks, dropout,
    and a final upsample to restore full resolution.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        pretrained: bool = False,
        deep_start_filters: int = 64,
    ):
        super().__init__()
        # 1) Load ResNet34 and optionally swap in_channels
        backbone = resnet34(pretrained=pretrained)
        if in_channels != 3:
            backbone.conv1 = nn.Conv2d(
                in_channels, deep_start_filters,
                kernel_size=7, stride=2, padding=3, bias=False
            )

        # 2) Split off initial conv→bn→relu (skip0 at H/2) and maxpool (to H/4)
        self.initial = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
        )  # [B, deep_start_filters, H/2, W/2]
        self.pool0   = backbone.maxpool  # [B, deep_start_filters, H/4, W/4]

        # 3) Encoder stages (H/4→H/32)
        self.stage1 = backbone.layer1  # [B,  64, H/4, W/4]
        self.stage2 = backbone.layer2  # [B, 128, H/8, W/8]
        self.stage3 = backbone.layer3  # [B, 256, H/16, W/16]
        self.stage4 = backbone.layer4  # [B, 512, H/32, W/32]

        # 4) Bottleneck with dropout
        self.center = nn.Sequential(
            ConvBlock(512, 512),
            nn.Dropout2d(0.3),
        )

        # 5) Decoder blocks (mirror encoder)
        self.dec4 = DecoderBlock(512, 256, 256)      # H/32→H/16, cat stage3
        self.dec3 = DecoderBlock(256, 128, 128)      # H/16→H/8,  cat stage2
        self.dec2 = DecoderBlock(128,  64,  64)      # H/8 →H/4,  cat stage1
        self.dec1 = DecoderBlock( 64, deep_start_filters, deep_start_filters)
                                                     # H/4 →H/2,  cat initial

        # 6) Final conv + small dropout
        self.final_dropout = nn.Dropout2d(0.1)
        self.final_conv    = nn.Conv2d(deep_start_filters, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        skip0 = self.initial(x)        # H/2
        x1    = self.pool0(skip0)      # H/4
        s1    = self.stage1(x1)        # H/4
        s2    = self.stage2(s1)        # H/8
        s3    = self.stage3(s2)        # H/16
        s4    = self.stage4(s3)        # H/32

        # Bottleneck
        c = self.center(s4)

        # Decoder
        d4 = self.dec4(c,  s3)  # H/16
        d3 = self.dec3(d4, s2)  # H/8
        d2 = self.dec2(d3, s1)  # H/4
        d1 = self.dec1(d2, skip0)  # H/2

        # Final projection + upsample to input size
        out = self.final_dropout(d1)
        out = self.final_conv(out)   # [B, out_channels, H/2, W/2]
        # restore full H×W
        out = F.interpolate(out, size=x.shape[2:], mode="bilinear", align_corners=False)
        return out
