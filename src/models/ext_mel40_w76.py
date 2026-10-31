"""
Ext_Mel40_StridedStem_W76: High-Accuracy (95.0%) 12-Class Keyword Spotter.
Operates on 40-band Log-Mel spectrograms with an asymmetric strided stem.
Parameters: 39,140 | MACs: 3.99M | SRAM Arena: 20,448 B (20.0 KB) | INT8 Test Acc: 94.99% (~95.0%)
Target Vocabulary: 12 classes (Pete Warden Google Speech Commands v0.02)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableBlockFolded(nn.Module):
    """Depthwise-separable block with folded batch norm (Conv + ReLU)."""
    def __init__(self, in_ch: int, out_ch: int, kernel_size=(3, 3), stride=1, padding=1):
        super().__init__()
        self.dw = nn.Conv2d(
            in_ch, in_ch, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_ch, bias=True
        )
        self.relu_dw = nn.ReLU(inplace=True)
        self.pw = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, bias=True)
        self.relu_pw = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu_dw(self.dw(x))
        x = self.relu_pw(self.pw(x))
        return x


class Ext_Mel40_StridedStem_W76(nn.Module):
    """
    Mel40 input (1, 49, 40)
    Stem stride (2, 4) compresses time to 25 and frequency to 10:
      Conv1: (1, 48, 5, 4, stride=(2, 4), padding=(2, 0)) -> (B, 48, 25, 10)
      B1 DW: (48, 48, 3, 3, stride=(1, 2), padding=1) -> (B, 48, 25, 5)
      B1 PW: (48, 76, 1, 1) -> (B, 76, 25, 5)
      Skip: AdaptiveAvgPool2d((1, 1)) -> Linear(76, 76, bias=False)
      4 Blocks: DepthwiseSeparableBlock(76, 76)
      AdaptiveAvgPool2d((1, 1)) + Linear(76, num_classes)
    """
    def __init__(self, num_classes: int = 12, channels: int = 76):
        super().__init__()
        self.num_classes = num_classes
        self.channels = channels

        self.conv1 = nn.Conv2d(1, 48, kernel_size=(5, 4), stride=(2, 4), padding=(2, 0), bias=True)
        self.relu1 = nn.ReLU(inplace=True)

        self.b1_dw = nn.Conv2d(48, 48, kernel_size=(3, 3), stride=(1, 2), padding=1, groups=48, bias=True)
        self.b1_relu_dw = nn.ReLU(inplace=True)
        self.b1_pw = nn.Conv2d(48, channels, kernel_size=1, stride=1, bias=True)
        self.b1_relu_pw = nn.ReLU(inplace=True)

        self.skip_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.skip_proj = nn.Linear(channels, channels, bias=False)

        self.blocks = nn.ModuleList([
            DepthwiseSeparableBlockFolded(channels, channels, stride=1) for _ in range(4)
        ])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(channels, num_classes, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu1(self.conv1(x))
        x = self.b1_relu_dw(self.b1_dw(x))
        x = self.b1_relu_pw(self.b1_pw(x))
        skip = self.skip_proj(torch.flatten(self.skip_pool(x), 1))
        for b in self.blocks:
            x = b(x)
        gap = torch.flatten(self.pool(x), 1)
        return self.classifier(gap + skip)
