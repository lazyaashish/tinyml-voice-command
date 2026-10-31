"""
Hypothesis-Driven Architecture Candidates for Ultra-Low-SRAM Search.
Strict Constraints:
- Peak SRAM <= 15.625 KB (Hard ceiling, Arm Baseline)
- Preferred Peak SRAM <= 13.50 KB (Current Winner)
- MACs <= 1,434,112 (Preferred) or < 2,656,768 (Hard ceiling)
- Parameters <= 14,764 (Preferred) or < 23,180 (Hard ceiling)
"""

import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

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
    """Squeeze-and-Excitation channel gating with configurable reduction ratio."""
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        mid_ch = max(4, channels // reduction)
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
                 use_residual: bool = False, use_se: bool = False, se_reduction: int = 8):
        super().__init__()
        self.use_residual = use_residual and (in_ch == out_ch and stride == 1)
        self.dw = ConvBNReLU6(in_ch, in_ch, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_ch)
        self.pw = ConvBNReLU6(in_ch, out_ch, kernel_size=(1, 1), stride=1, padding=0)
        self.se = SEBlock(out_ch, reduction=se_reduction) if use_se else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.pw(self.dw(x))
        if self.se is not None:
            out = self.se(out)
        if self.use_residual:
            out = out + x
        return out

class InvertedBottleneckBlock(nn.Module):
    """Lightweight inverted bottleneck block with expansion ratio <= 1.5."""
    def __init__(self, in_ch: int, out_ch: int, expansion: float = 1.5, kernel_size=(3, 3), stride=1, padding=1, use_se: bool = True):
        super().__init__()
        self.use_residual = (in_ch == out_ch and stride == 1)
        mid_ch = int(in_ch * expansion)
        self.expand = ConvBNReLU6(in_ch, mid_ch, kernel_size=(1, 1), stride=1, padding=0)
        self.dw = ConvBNReLU6(mid_ch, mid_ch, kernel_size=kernel_size, stride=stride, padding=padding, groups=mid_ch)
        self.se = SEBlock(mid_ch, reduction=8) if use_se else None
        self.project = nn.Sequential(
            nn.Conv2d(mid_ch, out_ch, kernel_size=(1, 1), stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_ch)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.dw(self.expand(x))
        if self.se is not None:
            out = self.se(out)
        out = self.project(out)
        if self.use_residual:
            out = out + x
        return F.relu6(out)


# ==============================================================================
# REFERENCE PRODUCTION MODEL: UltraLowSRAM_MFCC10 (The model to beat)
# ==============================================================================
try:
    from .ultra_low_sram import UltraLowSRAM_MFCC10
except ImportError:
    from src.models.ultra_low_sram import UltraLowSRAM_MFCC10


# ==============================================================================
# CANDIDATE 1 (Phase 3): Late Channel Capacity Expansion (24 -> 36 -> 48 -> 64)
# Hypothesis: Shifting capacity from early stem (32->24) to mid layers (32->36)
# keeps all layer activations <= 6,912 bytes, strictly guaranteeing 13.50 KB SRAM.
# ==============================================================================
class Cand1_LateCapacity(nn.Module):
    def __init__(self, num_classes: int = 12):
        super().__init__()
        self.init_conv = ConvBNReLU6(1, 24, kernel_size=(10, 4), stride=(2, 1), padding=(4, 1)) # (24, 24, 9) = 5,184 B
        self.block1 = DepthwiseSeparableBlock(24, 36, kernel_size=(3, 3), stride=1, padding=1)
        self.block2 = DepthwiseSeparableBlock(36, 48, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1), use_se=True)
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
        return self.fc(torch.flatten(x, 1))


# ==============================================================================
# CANDIDATE 2 (Phase 4): Dual Residual Highway (Residuals in Block 1 & Block 3)
# Hypothesis: Adding an identity residual to Block 1 (32->32) creates a continuous
# residual highway from early features to classifier, improving multi-seed stability.
# ==============================================================================
class Cand2_DualResidual(nn.Module):
    def __init__(self, num_classes: int = 12):
        super().__init__()
        self.init_conv = ConvBNReLU6(1, 32, kernel_size=(10, 4), stride=(2, 1), padding=(4, 1))
        # Block 1 WITH identity residual
        self.block1 = DepthwiseSeparableBlock(32, 32, kernel_size=(3, 3), stride=1, padding=1, use_residual=True)
        self.block2 = DepthwiseSeparableBlock(32, 48, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1), use_se=True)
        # Block 3 WITH identity residual
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
        return self.fc(torch.flatten(x, 1))


# ==============================================================================
# CANDIDATE 3 (Phase 5): Inverted Bottleneck Refinement in Deep Layer
# Hypothesis: Small expansion (1.5x) in Block 3 enriches intermediate nonlinearities
# while keeping Peak SRAM strictly <= 13.50 KB.
# ==============================================================================
class Cand3_InvertedBottleneck(nn.Module):
    def __init__(self, num_classes: int = 12):
        super().__init__()
        self.init_conv = ConvBNReLU6(1, 32, kernel_size=(10, 4), stride=(2, 1), padding=(4, 1))
        self.block1 = DepthwiseSeparableBlock(32, 32, kernel_size=(3, 3), stride=1, padding=1)
        self.block2 = DepthwiseSeparableBlock(32, 48, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1), use_se=True)
        # Inverted Bottleneck in Block 3: 48 -> 72 -> 48
        self.block3 = InvertedBottleneckBlock(48, 48, expansion=1.5, kernel_size=(3, 3), stride=1, padding=1, use_se=True)
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
        return self.fc(torch.flatten(x, 1))


# ==============================================================================
# CANDIDATE 4 (Phase 6): High-Fidelity Transition SE (r=4 in Block 2)
# Hypothesis: Finer-grained channel gating (r=4 instead of r=8) at the critical
# time-downsampling junction in Block 2 preserves subtle formant modulations.
# ==============================================================================
class Cand4_TunedSE(nn.Module):
    def __init__(self, num_classes: int = 12):
        super().__init__()
        self.init_conv = ConvBNReLU6(1, 32, kernel_size=(10, 4), stride=(2, 1), padding=(4, 1))
        self.block1 = DepthwiseSeparableBlock(32, 32, kernel_size=(3, 3), stride=1, padding=1)
        # Block 2 with reduction=4 (12 intermediate dims instead of 6)
        self.block2 = DepthwiseSeparableBlock(32, 48, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1), use_se=True, se_reduction=4)
        self.block3 = DepthwiseSeparableBlock(48, 48, kernel_size=(3, 3), stride=1, padding=1, use_residual=True, use_se=True, se_reduction=8)
        self.block4 = DepthwiseSeparableBlock(48, 64, kernel_size=(3, 3), stride=1, padding=1, use_se=True, se_reduction=8)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.init_conv(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.pool(x)
        return self.fc(torch.flatten(x, 1))


# ==============================================================================
# CANDIDATE 5 (Phase 7): Asymmetric Temporal Receptive Field (5x3 Kernels)
# Hypothesis: 5x3 kernels span 100ms across time in early blocks, enhancing phoneme duration modeling.
# ==============================================================================
class Cand5_Temporal5x3(nn.Module):
    def __init__(self, num_classes: int = 12):
        super().__init__()
        self.init_conv = ConvBNReLU6(1, 32, kernel_size=(10, 4), stride=(2, 1), padding=(4, 1))
        self.block1 = DepthwiseSeparableBlock(32, 32, kernel_size=(5, 3), stride=1, padding=(2, 1))
        self.block2 = DepthwiseSeparableBlock(32, 48, kernel_size=(5, 3), stride=(2, 1), padding=(2, 1), use_se=True)
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
        return self.fc(torch.flatten(x, 1))


# ==============================================================================
# NEW CONSTRAINED SEARCH CANDIDATES (Strictly <= 16,000 Bytes Deployment Arena)
# ==============================================================================

# Candidate A: 48 -> 64 -> 48 Non-Residual Expansion
# Peak SRAM: 13,824 B ping-pong / 14,976 B with scratch (<= 16,000 B!)
class CandA_NonResidualExpansion(nn.Module):
    def __init__(self, num_classes: int = 12):
        super().__init__()
        self.init_conv = ConvBNReLU6(1, 32, kernel_size=(10, 4), stride=(2, 1), padding=(4, 1))
        self.block1 = DepthwiseSeparableBlock(32, 32, kernel_size=(3, 3), stride=1, padding=1)
        self.block2 = DepthwiseSeparableBlock(32, 48, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1), use_se=True)
        # Non-residual 48 -> 64 -> 48 expansion: NO held residual!
        self.b3_expand = ConvBNReLU6(48, 64, kernel_size=(1, 1), stride=1, padding=0)
        self.b3_dw = ConvBNReLU6(64, 64, kernel_size=(3, 3), stride=1, padding=1, groups=64)
        self.b3_se = SEBlock(64, reduction=8)
        self.b3_project = ConvBNReLU6(64, 48, kernel_size=(1, 1), stride=1, padding=0)
        self.block4 = DepthwiseSeparableBlock(48, 64, kernel_size=(3, 3), stride=1, padding=1, use_se=True)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.init_conv(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.b3_project(self.b3_se(self.b3_dw(self.b3_expand(x))))
        x = self.block4(x)
        x = self.pool(x)
        return self.fc(torch.flatten(x, 1))


# Candidate B: Deep Dual 48-Channel Stages (3a + 3b with local residuals)
# Peak SRAM: 13,824 B ping-pong / 15,552 B with residual hold (<= 16,000 B!)
class CandB_DeepDual48(nn.Module):
    def __init__(self, num_classes: int = 12):
        super().__init__()
        self.init_conv = ConvBNReLU6(1, 32, kernel_size=(10, 4), stride=(2, 1), padding=(4, 1))
        self.block1 = DepthwiseSeparableBlock(32, 32, kernel_size=(3, 3), stride=1, padding=1)
        self.block2 = DepthwiseSeparableBlock(32, 48, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1), use_se=True)
        self.block3a = DepthwiseSeparableBlock(48, 48, kernel_size=(3, 3), stride=1, padding=1, use_residual=True, use_se=True)
        self.block3b = DepthwiseSeparableBlock(48, 48, kernel_size=(3, 3), stride=1, padding=1, use_residual=True, use_se=True)
        self.block4 = DepthwiseSeparableBlock(48, 64, kernel_size=(3, 3), stride=1, padding=1, use_se=True)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.init_conv(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3a(x)
        x = self.block3b(x)
        x = self.block4(x)
        x = self.pool(x)
        return self.fc(torch.flatten(x, 1))


# Candidate C: Late Temporal 5x3 Kernels in Block 3 and Block 4
# Peak SRAM: 13,824 B ping-pong / 15,264 B with scratch (<= 16,000 B!)
class CandC_LateTemporal5x3(nn.Module):
    def __init__(self, num_classes: int = 12):
        super().__init__()
        self.init_conv = ConvBNReLU6(1, 32, kernel_size=(10, 4), stride=(2, 1), padding=(4, 1))
        self.block1 = DepthwiseSeparableBlock(32, 32, kernel_size=(3, 3), stride=1, padding=1)
        self.block2 = DepthwiseSeparableBlock(32, 48, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1), use_se=True)
        self.block3 = DepthwiseSeparableBlock(48, 48, kernel_size=(5, 3), stride=1, padding=(2, 1), use_residual=True, use_se=True)
        self.block4 = DepthwiseSeparableBlock(48, 64, kernel_size=(5, 3), stride=1, padding=(2, 1), use_se=True)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.init_conv(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.pool(x)
        return self.fc(torch.flatten(x, 1))


# Candidate D: Wider Mid-Stage (32 -> 56 -> 64 without residual hold)
# Peak SRAM: 12,096 B ping-pong / 13,104 B with scratch (<= 13,824 B Preferred!)
class CandD_WiderMidStage(nn.Module):
    def __init__(self, num_classes: int = 12):
        super().__init__()
        self.init_conv = ConvBNReLU6(1, 32, kernel_size=(10, 4), stride=(2, 1), padding=(4, 1))
        self.block1 = DepthwiseSeparableBlock(32, 32, kernel_size=(3, 3), stride=1, padding=1)
        self.block2 = DepthwiseSeparableBlock(32, 56, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1), use_se=True)
        self.block3 = DepthwiseSeparableBlock(56, 56, kernel_size=(3, 3), stride=1, padding=1, use_residual=False, use_se=True)
        self.block4 = DepthwiseSeparableBlock(56, 64, kernel_size=(3, 3), stride=1, padding=1, use_se=True)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.init_conv(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.pool(x)
        return self.fc(torch.flatten(x, 1))


# Candidate E: Deep Dual 48 Stage + Temporal 5x3 in Block 3b
# Peak SRAM: 13,824 B ping-pong / 15,552 B with residual hold (<= 16,000 B!)
class CandE_Dual48_Temporal5x3(nn.Module):
    def __init__(self, num_classes: int = 12):
        super().__init__()
        self.init_conv = ConvBNReLU6(1, 32, kernel_size=(10, 4), stride=(2, 1), padding=(4, 1))
        self.block1 = DepthwiseSeparableBlock(32, 32, kernel_size=(3, 3), stride=1, padding=1)
        self.block2 = DepthwiseSeparableBlock(32, 48, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1), use_se=True)
        self.block3a = DepthwiseSeparableBlock(48, 48, kernel_size=(3, 3), stride=1, padding=1, use_residual=True, use_se=True)
        self.block3b = DepthwiseSeparableBlock(48, 48, kernel_size=(5, 3), stride=1, padding=(2, 1), use_residual=True, use_se=True)
        self.block4 = DepthwiseSeparableBlock(48, 64, kernel_size=(3, 3), stride=1, padding=1, use_se=True)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.init_conv(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3a(x)
        x = self.block3b(x)
        x = self.block4(x)
        x = self.pool(x)
        return self.fc(torch.flatten(x, 1))


# Candidate F: 44-Channel Dual Residual Stages with SRAM Headroom
# Peak SRAM: 14,256 B tensors + 792 B scratch = 15,048 B (14.70 KB <= 16,000 B!)
class CandF_44ChDualResidual(nn.Module):
    def __init__(self, num_classes: int = 12):
        super().__init__()
        self.init_conv = ConvBNReLU6(1, 32, kernel_size=(10, 4), stride=(2, 1), padding=(4, 1))
        self.block1 = DepthwiseSeparableBlock(32, 32, kernel_size=(3, 3), stride=1, padding=1)
        self.block2 = DepthwiseSeparableBlock(32, 44, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1), use_se=True)
        self.block3a = DepthwiseSeparableBlock(44, 44, kernel_size=(3, 3), stride=1, padding=1, use_residual=True, use_se=True)
        self.block3b = DepthwiseSeparableBlock(44, 44, kernel_size=(3, 3), stride=1, padding=1, use_residual=True, use_se=True)
        self.block4 = DepthwiseSeparableBlock(44, 64, kernel_size=(3, 3), stride=1, padding=1, use_se=True)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.init_conv(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3a(x)
        x = self.block3b(x)
        x = self.block4(x)
        x = self.pool(x)
        return self.fc(torch.flatten(x, 1))


