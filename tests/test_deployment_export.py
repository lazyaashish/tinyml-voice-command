"""
Unit tests for ESP32 model exporter and deployment header generation.
"""

import os
import sys
import tempfile
import unittest
import numpy as np
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.models import TinyML_DSCNN
from src.preprocessing import AudioFeatureExtractor, LogMelSpectrogramExtractor
from scripts.export_esp32 import export_model_data_header, export_model_config_header


class TestDeploymentExport(unittest.TestCase):
    def setUp(self):
        self.model = TinyML_DSCNN(num_classes=5)
        self.model.eval()

    def test_parameter_count(self):
        total_params = sum(p.numel() for p in self.model.parameters())
        self.assertEqual(total_params, 11409, f"Expected 11,409 parameters, got {total_params}")

    def test_audio_feature_extractor(self):
        extractor = AudioFeatureExtractor(feature_type="mel40_cmn", sample_rate=16000)
        dummy_audio = np.random.randn(16000).astype(np.float32)
        features = extractor.extract(dummy_audio)
        self.assertEqual(features.shape, (1, 49, 40), f"Expected (1, 49, 40), got {features.shape}")

    def test_log_mel_alias(self):
        self.assertIs(LogMelSpectrogramExtractor, AudioFeatureExtractor)

    def test_header_export(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_header = os.path.join(tmpdir, "model_data.h")
            config_header = os.path.join(tmpdir, "model_config.h")

            elems, nbytes = export_model_data_header(self.model, data_header, fmt="int8")
            self.assertEqual(elems, 11409)
            self.assertGreater(nbytes, 10000)
            self.assertTrue(os.path.isfile(data_header))

            export_model_config_header(
                model_name="TinyML_DSCNN",
                classes=["GO", "STOP", "LEFT", "RIGHT", "UNKNOWN"],
                param_count=11409,
                macs=2056640,
                arena_bytes=28800,
                mel_bins=40,
                time_frames=49,
                output_file=config_header,
            )
            self.assertTrue(os.path.isfile(config_header))

            with open(config_header, "r") as f:
                content = f.read()
            self.assertIn("TinyML_DSCNN", content)
            self.assertIn("KWS_NUM_CLASSES         5", content)


if __name__ == "__main__":
    unittest.main()

