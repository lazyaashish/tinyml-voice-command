"""
Model architectures for SIH ESP32 Keyword Spotting and TinyML SRAM Optimization.
"""

from .sih_dscnn import TinyML_DSCNN, TinyML_AsymDSCNN, DepthwiseSeparableConv2d
from .cand_f import (
    CandF_44ChDualResidual,
    CandB_DeepDual48,
    Cand3_InvertedBottleneck,
    Cand1_LateCapacity,
    Cand2_DualResidual,
)
from .arm_baseline import ArmSmallDSCNN
from .ext_mel40_w76 import Ext_Mel40_StridedStem_W76
from .low_sram_candidates import LowSRAM_SE, LowSRAM_TimeFirst
from .ultra_low_sram import UltraLowSRAM_MFCC10

__all__ = [
    "TinyML_DSCNN",
    "TinyML_AsymDSCNN",
    "DepthwiseSeparableConv2d",
    "CandF_44ChDualResidual",
    "CandB_DeepDual48",
    "Cand3_InvertedBottleneck",
    "Cand1_LateCapacity",
    "Cand2_DualResidual",
    "ArmSmallDSCNN",
    "Ext_Mel40_StridedStem_W76",
    "LowSRAM_SE",
    "LowSRAM_TimeFirst",
    "UltraLowSRAM_MFCC10",
]
