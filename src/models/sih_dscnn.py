"""
TinyML Asymmetric Depthwise Separable CNN (11,409 parameters) for SIH 5-class KWS.
Classes: GO, STOP, LEFT, RIGHT, UNKNOWN.
Target: ESP32 Xtensa Dual-Core 240 MHz (Log-Mel 40 + CMN features).
"""

import torch
import torch.nn as nn


class DepthwiseSeparableConv2d(nn.Module):
    """
    Standard MobileNet/TinyML Depthwise Separable 2D Convolution:
    Depthwise Conv (groups=in_c) + Pointwise Conv (1x1) + BatchNorm + ReLU6
    """

    def __init__(self, in_channels: int, out_channels: int, stride=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_channels,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU6(inplace=True)
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU6(inplace=True)

    def forward(self, x):
        return self.relu2(self.bn2(self.pointwise(self.relu1(self.bn1(self.depthwise(x))))))


class TinyML_DSCNN(nn.Module):
    """
    Research-grade Asymmetric Depthwise Separable CNN (Arm 'Hello Edge' SOTA):
    - Input: (B, 1, 49, 40)
    - Asymmetric Conv: 10x4 Phoneme Kernel with Stride (2, 2)
    - 4 DS Blocks: Progressive Channels (28 -> 32 -> 48 -> 64)
    - Global AvgPool + Linear Classifier (64 -> 5)
    - Total Parameters: 11,409
    """

    def __init__(self, num_classes=5):
        super().__init__()
        self.init_conv = nn.Sequential(
            nn.Conv2d(1, 28, kernel_size=(10, 4), stride=(2, 2), padding=(4, 1), bias=False),
            nn.BatchNorm2d(28),
            nn.ReLU6(inplace=True),
        )
        self.block1 = DepthwiseSeparableConv2d(28, 32, stride=1)
        self.block2 = DepthwiseSeparableConv2d(32, 48, stride=2)
        self.block3 = DepthwiseSeparableConv2d(48, 48, stride=1)
        self.block4 = DepthwiseSeparableConv2d(48, 64, stride=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.init_conv(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# Alias for compatibility with training script naming
TinyML_AsymDSCNN = TinyML_DSCNN
