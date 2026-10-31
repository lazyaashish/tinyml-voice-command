"""
Candidate Architectures for Low-SRAM Optimization with High Accuracy Recovery.
Target Constraints:
    - Test Accuracy >= 90.88%
    - MACs <= 2,100,000
    - Parameters <= 15,000
    - Peak SRAM <= 25.00 KB (<= 25,600 bytes)

Candidate Families:
1. LowSRAM_TimeFirst: Staged downsampling time-first (stride 2x1) then freq (stride 1x2) to preserve formant dynamics.
2. LowSRAM_SE: Candidate 1 augmented with ultra-lightweight Squeeze-and-Excitation channel attention (reduction=8).
3. LowSRAM_WideSemantic: Reallocating channels (24 -> 32 -> 52 -> 72) with identity residual in Block 3.
4. LowSRAM_TemporalKernel: Asymmetric 5x3 depthwise kernels in stages 1-3 to preserve phoneme duration cues.
5. LowSRAM_ProjectionResidual: 1x1 projection shortcuts across downsampling stages to prevent information loss.
6. LowSRAM_Synergy: Combining Wide Semantic (24 -> 32 -> 52 -> 72) + SE channel attention + Residuals.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBNReLU6(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, kernel_size=(3, 3), stride=1, padding=1, groups=1):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU6(inplace=True)
        )

class SEBlock(nn.Module):
    """Ultra-lightweight Squeeze-and-Excitation channel gating."""
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        mid_ch = max(8, channels // reduction)
        self.fc1 = nn.Linear(channels, mid_ch, bias=False)
        self.relu = nn.ReLU6(inplace=True)
        self.fc2 = nn.Linear(mid_ch, channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        w = x.mean(dim=(2, 3))
        w = self.fc1(w)
        w = self.relu(w)
        w = self.fc2(w)
        w = self.sigmoid(w).view(b, c, 1, 1)
        return x * w

class DepthwiseSeparableBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size=(3, 3), stride=1, padding=1,
                 use_residual: bool = False, use_se: bool = False):
        super().__init__()
        self.use_residual = use_residual and (in_ch == out_ch and stride == 1)
        self.dw = ConvBNReLU6(in_ch, in_ch, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_ch)
        self.pw = ConvBNReLU6(in_ch, out_ch, kernel_size=(1, 1), stride=1, padding=0)
        self.se = SEBlock(out_ch) if use_se else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.pw(self.dw(x))
        if self.se is not None:
            out = self.se(out)
        if self.use_residual:
            out = out + x
        return out

class DownsampleResidualBlock(nn.Module):
    """Depthwise separable block with 1x1 strided projection shortcut."""
    def __init__(self, in_ch: int, out_ch: int, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)):
        super().__init__()
        self.dw = ConvBNReLU6(in_ch, in_ch, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_ch)
        self.pw = ConvBNReLU6(in_ch, out_ch, kernel_size=(1, 1), stride=1, padding=0)
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=(1, 1), stride=stride, padding=0, bias=False),
            nn.BatchNorm2d(out_ch)
        )
        self.relu = nn.ReLU6(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.pw(self.dw(x)) + self.shortcut(x))


# ==============================================================================
# Model 1: LowSRAM_TimeFirst
# ==============================================================================
class LowSRAM_TimeFirst(nn.Module):
    """Downsamples time first (stride 2x1) in Block 1, then frequency (stride 1x2) in Block 2."""
    def __init__(self, num_classes: int = 12):
        super().__init__()
        self.init_conv = ConvBNReLU6(1, 24, kernel_size=(10, 4), stride=(2, 2), padding=(4, 1)) # -> (24, 24, 20)
        # Block 1: Time downsample -> (32, 12, 20)
        self.block1 = DepthwiseSeparableBlock(24, 32, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1))
        # Block 2: Frequency downsample -> (48, 12, 10)
        self.block2 = DepthwiseSeparableBlock(32, 48, kernel_size=(3, 3), stride=(1, 2), padding=(1, 1))
        # Block 3: Refinement on (12, 10) + residual
        self.block3 = DepthwiseSeparableBlock(48, 48, kernel_size=(3, 3), stride=1, padding=1, use_residual=True)
        # Block 4: Semantic expansion -> 64
        self.block4 = DepthwiseSeparableBlock(48, 64, kernel_size=(3, 3), stride=1, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.init_conv(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


# ==============================================================================
# Model 2: LowSRAM_SE
# ==============================================================================
class LowSRAM_SE(nn.Module):
    """Candidate 1 augmented with SE channel attention in Blocks 2, 3, 4."""
    def __init__(self, num_classes: int = 12):
        super().__init__()
        self.init_conv = ConvBNReLU6(1, 24, kernel_size=(10, 4), stride=(2, 2), padding=(4, 1))
        self.block1 = DepthwiseSeparableBlock(24, 32, kernel_size=(3, 3), stride=(1, 2), padding=(1, 1))
        self.block2 = DepthwiseSeparableBlock(32, 48, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1), use_se=True)
        self.block3 = DepthwiseSeparableBlock(48, 48, kernel_size=(3, 3), stride=1, padding=1, use_residual=True, use_se=True)
        self.block4 = DepthwiseSeparableBlock(48, 64, kernel_size=(3, 3), stride=1, padding=1, use_se=True)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.init_conv(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


# ==============================================================================
# Model 3: LowSRAM_WideSemantic
# ==============================================================================
class LowSRAM_WideSemantic(nn.Module):
    """Reallocates capacity to semantic layers: 24 -> 32 -> 52 -> 72 while maintaining 19.2 KB SRAM."""
    def __init__(self, num_classes: int = 12):
        super().__init__()
        self.init_conv = ConvBNReLU6(1, 24, kernel_size=(10, 4), stride=(2, 2), padding=(4, 1))
        self.block1 = DepthwiseSeparableBlock(24, 32, kernel_size=(3, 3), stride=(1, 2), padding=(1, 1))
        self.block2 = DepthwiseSeparableBlock(32, 52, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1))
        self.block3 = DepthwiseSeparableBlock(52, 52, kernel_size=(3, 3), stride=1, padding=1, use_residual=True)
        self.block4 = DepthwiseSeparableBlock(52, 72, kernel_size=(3, 3), stride=1, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(72, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.init_conv(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


# ==============================================================================
# Model 4: LowSRAM_TemporalKernel
# ==============================================================================
class LowSRAM_TemporalKernel(nn.Module):
    """Uses 5x3 asymmetric depthwise kernels in staged downsampling blocks."""
    def __init__(self, num_classes: int = 12):
        super().__init__()
        self.init_conv = ConvBNReLU6(1, 24, kernel_size=(10, 4), stride=(2, 2), padding=(4, 1))
        self.block1 = DepthwiseSeparableBlock(24, 32, kernel_size=(5, 3), stride=(1, 2), padding=(2, 1))
        self.block2 = DepthwiseSeparableBlock(32, 48, kernel_size=(5, 3), stride=(2, 1), padding=(2, 1))
        self.block3 = DepthwiseSeparableBlock(48, 48, kernel_size=(5, 3), stride=1, padding=(2, 1), use_residual=True)
        self.block4 = DepthwiseSeparableBlock(48, 64, kernel_size=(3, 3), stride=1, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.init_conv(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


# ==============================================================================
# Model 5: LowSRAM_ProjectionResidual
# ==============================================================================
class LowSRAM_ProjectionResidual(nn.Module):
    """Uses 1x1 projection shortcuts across downsampling blocks to retain gradient/feature flow."""
    def __init__(self, num_classes: int = 12):
        super().__init__()
        self.init_conv = ConvBNReLU6(1, 24, kernel_size=(10, 4), stride=(2, 2), padding=(4, 1))
        self.block1 = DownsampleResidualBlock(24, 32, kernel_size=(3, 3), stride=(1, 2), padding=(1, 1))
        self.block2 = DownsampleResidualBlock(32, 48, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1))
        self.block3 = DepthwiseSeparableBlock(48, 48, kernel_size=(3, 3), stride=1, padding=1, use_residual=True)
        self.block4 = DepthwiseSeparableBlock(48, 64, kernel_size=(3, 3), stride=1, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.init_conv(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


# ==============================================================================
# Model 6: LowSRAM_Synergy
# ==============================================================================
class LowSRAM_Synergy(nn.Module):
    """Synergy model: Wide Semantic (24 -> 32 -> 48 -> 68) + SE channel attention + Residuals."""
    def __init__(self, num_classes: int = 12):
        super().__init__()
        self.init_conv = ConvBNReLU6(1, 24, kernel_size=(10, 4), stride=(2, 2), padding=(4, 1))
        self.block1 = DepthwiseSeparableBlock(24, 32, kernel_size=(3, 3), stride=(1, 2), padding=(1, 1))
        self.block2 = DepthwiseSeparableBlock(32, 48, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1), use_se=True)
        self.block3 = DepthwiseSeparableBlock(48, 48, kernel_size=(3, 3), stride=1, padding=1, use_residual=True, use_se=True)
        self.block4 = DepthwiseSeparableBlock(48, 68, kernel_size=(3, 3), stride=1, padding=1, use_se=True)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(68, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.init_conv(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


# ==============================================================================
# Model 7: LowSRAM_SE_Temporal (Synergy: 5x3 Temporal Kernels + SE Channel Gating)
# ==============================================================================
class LowSRAM_SE_Temporal(nn.Module):
    """Synergy model: 5x3 asymmetric temporal depthwise kernels + SE channel gating."""
    def __init__(self, num_classes: int = 12):
        super().__init__()
        self.init_conv = ConvBNReLU6(1, 24, kernel_size=(10, 4), stride=(2, 2), padding=(4, 1))
        self.block1 = DepthwiseSeparableBlock(24, 32, kernel_size=(5, 3), stride=(1, 2), padding=(2, 1))
        self.block2 = DepthwiseSeparableBlock(32, 48, kernel_size=(5, 3), stride=(2, 1), padding=(2, 1), use_se=True)
        self.block3 = DepthwiseSeparableBlock(48, 48, kernel_size=(5, 3), stride=1, padding=(2, 1), use_residual=True, use_se=True)
        self.block4 = DepthwiseSeparableBlock(48, 64, kernel_size=(3, 3), stride=1, padding=1, use_se=True)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.init_conv(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)
