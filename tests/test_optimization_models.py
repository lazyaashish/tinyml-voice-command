"""
Unit test suite for academic KWS optimization and SRAM benchmark models.
Tests CandF FP32, CandF QAT INT8, Arm Small DS-CNN baseline, and comparison architectures.
"""

import os
import sys
import unittest
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.models import (
    CandF_44ChDualResidual,
    CandB_DeepDual48,
    Cand3_InvertedBottleneck,
    ArmSmallDSCNN,
    Ext_Mel40_StridedStem_W76,
)
from src.quantization import CandF_QAT

CKPT_DIR = os.path.join(REPO_ROOT, "checkpoints")


def _load_sd(path):
    data = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(data, dict) and "model_state_dict" in data:
        return data["model_state_dict"]
    return data


class TestOptimizationModels(unittest.TestCase):
    def test_cand_f_fp32(self):
        ckpt = os.path.join(CKPT_DIR, "cand_f_44ch_champion_fp32.pt")
        self.assertTrue(os.path.exists(ckpt), f"Missing {ckpt}")
        model = CandF_44ChDualResidual(num_classes=12)
        model.load_state_dict(_load_sd(ckpt))
        model.eval()

        # Input is 10-band MFCC: (B, 1, 49, 10)
        x = torch.randn(2, 1, 49, 10)
        out = model(x)
        self.assertEqual(out.shape, (2, 12))

    def test_cand_f_qat(self):
        ckpt = os.path.join(CKPT_DIR, "cand_f_qat_champion_int8.pt")
        self.assertTrue(os.path.exists(ckpt), f"Missing {ckpt}")
        model = CandF_QAT(num_classes=12)
        model.load_state_dict(_load_sd(ckpt))
        model.eval()

        x = torch.randn(2, 1, 49, 10)
        out = model(x)
        self.assertEqual(out.shape, (2, 12))

    def test_ext_mel40_w76(self):
        ckpt = os.path.join(CKPT_DIR, "ext_mel40_w76_champion_95.pt")
        self.assertTrue(os.path.exists(ckpt), f"Missing {ckpt}")
        model = Ext_Mel40_StridedStem_W76(num_classes=12)
        raw_sd = _load_sd(ckpt)
        filtered_sd = {k: v for k, v in raw_sd.items() if not any(x in k for x in ['quant_', 'running_max', 'initialized'])}
        model.load_state_dict(filtered_sd, strict=False)
        model.eval()

        # Input is 40-band Log-Mel: (B, 1, 49, 40)
        x = torch.randn(2, 1, 49, 40)
        out = model(x)
        self.assertEqual(out.shape, (2, 12))

    def test_cand_b(self):
        ckpt = os.path.join(CKPT_DIR, "cand_b_deep_dual48.pt")
        self.assertTrue(os.path.exists(ckpt), f"Missing {ckpt}")
        model = CandB_DeepDual48(num_classes=12)
        model.load_state_dict(_load_sd(ckpt))
        model.eval()

        x = torch.randn(2, 1, 49, 10)
        out = model(x)
        self.assertEqual(out.shape, (2, 12))

    def test_cand_3(self):
        ckpt = os.path.join(CKPT_DIR, "cand3_inverted_bottleneck.pt")
        self.assertTrue(os.path.exists(ckpt), f"Missing {ckpt}")
        model = Cand3_InvertedBottleneck(num_classes=12)
        model.load_state_dict(_load_sd(ckpt))
        model.eval()

        x = torch.randn(2, 1, 49, 10)
        out = model(x)
        self.assertEqual(out.shape, (2, 12))

    def test_arm_small_ds_cnn(self):
        ckpt = os.path.join(CKPT_DIR, "arm_small_dscnn_baseline.pt")
        self.assertTrue(os.path.exists(ckpt), f"Missing {ckpt}")
        model = ArmSmallDSCNN(num_classes=12)
        model.load_state_dict(_load_sd(ckpt))
        model.eval()

        x = torch.randn(2, 1, 49, 10)
        out = model(x)
        self.assertEqual(out.shape, (2, 12))


if __name__ == "__main__":
    unittest.main()

