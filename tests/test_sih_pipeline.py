"""
Unit test suite for the canonical SIH 5-Class Embedded Model.
Target: ESP32 Xtensa Dual-Core 240 MHz (Log-Mel 40 + CMN features).
"""

import os
import sys
import unittest
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.models import TinyML_DSCNN, TinyML_AsymDSCNN


class TestSIHPipeline(unittest.TestCase):
    def test_sih_model_load_and_forward(self):
        ckpt_path = os.path.join(REPO_ROOT, "checkpoints", "sih_final_5class.pt")
        self.assertTrue(os.path.exists(ckpt_path), f"Checkpoint missing: {ckpt_path}")

        model = TinyML_DSCNN(num_classes=5)
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if isinstance(sd, dict) and "model_state_dict" in sd:
            sd = sd["model_state_dict"]
        model.load_state_dict(sd)
        model.eval()

        # Check parameter count matches documented 11,409
        n_params = sum(p.numel() for p in model.parameters())
        self.assertEqual(n_params, 11409)

        # Forward pass with (B, 1, 49, 40) Log-Mel spectrogram input
        x = torch.randn(4, 1, 49, 40)
        out = model(x)
        self.assertEqual(out.shape, (4, 5))

    def test_sih_model_alias(self):
        self.assertIs(TinyML_AsymDSCNN, TinyML_DSCNN)


if __name__ == "__main__":
    unittest.main()

