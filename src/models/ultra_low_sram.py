"""
Ultra-Low-SRAM Architecture Suite strictly satisfying:
    Peak SRAM <= 15.625 KB (<= 16,000 bytes)
    Total MACs < 2,656,768
    Parameters < 23,180

Architectures:
1. UltraLowSRAM_C16_Mel40: 40 Log-Mel CMN with 16 initial channels (Peak SRAM: 15.00 KB)
2. UltraLowSRAM_C16_Wide_Mel40: 40 Log-Mel CMN with 16 init channels, wider mid channels (24->52->64)
3. UltraLowSRAM_Mel20: 20 Log-Mel CMN with 24 init channels, stride (2, 2) (Peak SRAM: 15.00 KB)
4. UltraLowSRAM_MFCC10: 10 MFCC with 32 init channels, stride (2, 1) (Peak SRAM: 15.00 KB)
5. UltraLowSRAM_Mel30: 30 Log-Mel CMN with 18 init channels, stride (2, 2) (Peak SRAM: 12.66 KB)
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


# ==============================================================================
# Model 1: UltraLowSRAM_C16_Mel40 (Input: 40 Log-Mel CMN)
# ==============================================================================
class UltraLowSRAM_C16_Mel40(nn.Module):
    """
    Input: 1 x 49 x 40 Log-Mel
    Init Conv: 1 -> 16 (stride 2, 2) -> (16, 24, 20) = 7,680 bytes
    Peak SRAM: 15,360 bytes (15.00 KB) <= 15.62 KB
    """
    def __init__(self, num_classes: int = 12):
        super().__init__()
        self.init_conv = ConvBNReLU6(1, 16, kernel_size=(10, 4), stride=(2, 2), padding=(4, 1))
        self.block1 = DepthwiseSeparableBlock(16, 24, kernel_size=(3, 3), stride=(1, 2), padding=(1, 1))
        self.block2 = DepthwiseSeparableBlock(24, 48, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1), use_se=True)
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
# Model 2: UltraLowSRAM_C16_Wide_Mel40 (Input: 40 Log-Mel CMN)
# ==============================================================================
class UltraLowSRAM_C16_Wide_Mel40(nn.Module):
    """
    Input: 1 x 49 x 40 Log-Mel
    Wider intermediate layers: 16 -> 24 -> 52 -> 64
    Peak SRAM: 15,360 bytes (15.00 KB) <= 15.62 KB
    """
    def __init__(self, num_classes: int = 12):
        super().__init__()
        self.init_conv = ConvBNReLU6(1, 16, kernel_size=(10, 4), stride=(2, 2), padding=(4, 1))
        self.block1 = DepthwiseSeparableBlock(16, 24, kernel_size=(3, 3), stride=(1, 2), padding=(1, 1))
        self.block2 = DepthwiseSeparableBlock(24, 52, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1), use_se=True)
        self.block3 = DepthwiseSeparableBlock(52, 52, kernel_size=(3, 3), stride=1, padding=1, use_residual=True, use_se=True)
        self.block4 = DepthwiseSeparableBlock(52, 64, kernel_size=(3, 3), stride=1, padding=1, use_se=True)
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
# Model 3: UltraLowSRAM_Mel20 (Input: 20 Log-Mel CMN)
# ==============================================================================
class UltraLowSRAM_Mel20(nn.Module):
    """
    Input: 1 x 49 x 20 Log-Mel
    Init Conv: 1 -> 24 (stride 2, 2) -> (24, 24, 10) = 5,760 bytes
    Peak SRAM: 15,360 bytes (15.00 KB) <= 15.62 KB
    """
    def __init__(self, num_classes: int = 12):
        super().__init__()
        self.init_conv = ConvBNReLU6(1, 24, kernel_size=(10, 4), stride=(2, 2), padding=(4, 1))
        self.block1 = DepthwiseSeparableBlock(24, 32, kernel_size=(3, 3), stride=1, padding=1)
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
# Model 4: UltraLowSRAM_MFCC10 (Input: 10 MFCC — Direct Arm Baseline Comparison)
# ==============================================================================
class UltraLowSRAM_MFCC10(nn.Module):
    """
    Input: 1 x 49 x 10 MFCC (Identical feature representation to Arm baseline)
    Init Conv: 1 -> 32 (stride 2, 1) -> (32, 24, 10) = 7,680 bytes
    Peak SRAM: 15,360 bytes (15.00 KB) <= 15.62 KB
    """
    def __init__(self, num_classes: int = 12):
        super().__init__()
        self.init_conv = ConvBNReLU6(1, 32, kernel_size=(10, 4), stride=(2, 1), padding=(4, 1))
        self.block1 = DepthwiseSeparableBlock(32, 32, kernel_size=(3, 3), stride=1, padding=1)
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
# Model 5: UltraLowSRAM_Mel30 (Input: 30 Log-Mel CMN)
# ==============================================================================
class UltraLowSRAM_Mel30(nn.Module):
    """
    Input: 1 x 49 x 30 Log-Mel
    Init Conv: 1 -> 18 (stride 2, 2) -> (18, 24, 15) = 6,480 bytes
    Peak SRAM: 12,960 bytes (12.66 KB) <= 15.62 KB
    """
    def __init__(self, num_classes: int = 12):
        super().__init__()
        self.init_conv = ConvBNReLU6(1, 18, kernel_size=(10, 4), stride=(2, 2), padding=(4, 1))
        self.block1 = DepthwiseSeparableBlock(18, 28, kernel_size=(3, 3), stride=(1, 2), padding=(1, 1))
        self.block2 = DepthwiseSeparableBlock(28, 48, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1), use_se=True)
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
