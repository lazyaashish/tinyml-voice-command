"""
Official Arm "Hello Edge" Small Depthwise Separable CNN (DS-CNN-S) Baseline
Reference: "Hello Edge: Keyword Spotting on Microcontrollers" (arXiv:1711.07128)
Official Arm Code: https://github.com/ARM-software/ML-KWS-for-MCU/blob/master/models.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSeparableConvBlock(nn.Module):
    """Standard MobileNet-style Depthwise Separable Convolution block as used by Arm."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1):
        super().__init__()
        # Depthwise Conv (per-channel spatial filtering)
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size, stride=stride,
            padding=kernel_size // 2, groups=in_channels, bias=False
        )
        self.bn_dw = nn.BatchNorm2d(in_channels)
        self.relu_dw = nn.ReLU(inplace=True)
        
        # Pointwise Conv (1x1 channel linear combination)
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, bias=False
        )
        self.bn_pw = nn.BatchNorm2d(out_channels)
        self.relu_pw = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu_dw(self.bn_dw(self.depthwise(x)))
        x = self.relu_pw(self.bn_pw(self.pointwise(x)))
        return x


class ArmSmallDSCNN(nn.Module):
    """
    Arm Small DS-CNN (DS-CNN-S):
    - Input: (B, 1, 49, 10) for standard 10 MFCC features
      (also supports (B, 1, 49, 40) for fair same-input comparisons)
    - conv_1: Conv2d(1, 64, kernel=(10, 4), stride=(2, 2)) with SAME padding
    - 4 DS-Conv blocks: DSC(64, 3, 1) -> uniform 64 channels throughout
    - Global Average Pooling: (t_dim, f_dim) -> (1, 1)
    - Linear Classifier: (64 -> num_classes)
    """
    def __init__(self, num_classes: int = 12, in_channels: int = 1, freq_bins: int = 10):
        super().__init__()
        self.num_classes = num_classes
        self.freq_bins = freq_bins
        
        # In TensorFlow, padding='SAME' on (49, 10) with kernel (10, 4), stride (2, 2):
        # time output = ceil(49/2) = 25 -> pad 9 (4 top, 5 bottom)
        # freq output = ceil(10/2) = 5  -> pad 2 (1 left, 1 right)
        # For 40 bins: freq output = ceil(40/2) = 20 -> pad 2 (1 left, 1 right)
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(10, 4), stride=(2, 2), bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.blocks = nn.ModuleList([
            DepthwiseSeparableConvBlock(64, 64, kernel_size=3, stride=1)
            for _ in range(4)
        ])
        
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Replicate TensorFlow SAME padding: pad=(pad_left, pad_right, pad_top, pad_bottom)
        x = F.pad(x, (1, 1, 4, 5))
        x = self.relu1(self.bn1(self.conv1(x)))
        for block in self.blocks:
            x = block(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)
