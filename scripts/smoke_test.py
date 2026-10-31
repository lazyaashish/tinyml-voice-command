#!/usr/bin/env python3
"""
scripts/smoke_test.py
---------------------
Fast end-to-end smoke test validating the complete runtime pipeline:
  Audio (16 kHz) -> Log-Mel (40 bands) -> TinyML_DSCNN -> Prediction -> C Header Verification.

Usage:
    python scripts/smoke_test.py
"""

import os
import sys
import time
import numpy as np
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.models import TinyML_DSCNN
from src.preprocessing import AudioFeatureExtractor


def main():
    print("==================================================")
    print("  ESP32 KWS End-to-End Pipeline Smoke Test")
    print("==================================================")
    start_time = time.time()

    # 1. Model Loading
    checkpoint_path = os.path.join(REPO_ROOT, "checkpoints", "sih_final_5class.pt")
    if not os.path.isfile(checkpoint_path):
        print(f"[FAIL] Missing checkpoint: {checkpoint_path}")
        sys.exit(1)

    print(f"[1/5] Loading recommended model: TinyML_DSCNN...")
    model = TinyML_DSCNN(num_classes=5)
    sd = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    model.load_state_dict(sd)
    model.eval()
    param_count = sum(p.numel() for p in model.parameters())
    print(f"      Parameters loaded: {param_count:,}")

    # 2. Synthetic Audio Input Generation
    print("[2/5] Synthesizing 1.0-second 16 kHz audio waveform...")
    t = np.linspace(0, 1.0, 16000, endpoint=False)
    # Synthetic speech-like formant frequencies (F1=300 Hz, F2=800 Hz) + noise
    audio = 0.5 * np.sin(2 * np.pi * 300 * t) + 0.3 * np.sin(2 * np.pi * 800 * t) + 0.05 * np.random.normal(0, 0.1, 16000)
    audio = audio.astype(np.float32)

    # 3. Audio Preprocessing
    print("[3/5] Extracting 40-band Log-Mel Spectrogram with CMN...")
    extractor = AudioFeatureExtractor(feature_type="mel40_cmn", sample_rate=16000)
    features = extractor.extract(audio)  # (1, 49, 40)
    tensor_in = torch.from_numpy(features).unsqueeze(0)  # (1, 1, 49, 40)
    print(f"      Feature tensor shape: {tuple(tensor_in.shape)}")
    assert tensor_in.shape == (1, 1, 49, 40), f"Unexpected shape {tensor_in.shape}"

    # 4. Neural Network Inference
    print("[4/5] Running forward inference pass...")
    with torch.no_grad():
        logits = model(tensor_in)
        probs = torch.softmax(logits, dim=-1).numpy()[0]

    classes = ["GO", "STOP", "LEFT", "RIGHT", "UNKNOWN"]
    top_idx = int(np.argmax(probs))
    top_class = classes[top_idx]
    top_conf = probs[top_idx] * 100.0

    print(f"      Output Distribution:")
    for cls, p in zip(classes, probs):
        print(f"        - {cls:<8}: {p * 100.0:5.2f}%")
    print(f"      Top Prediction: {top_class} ({top_conf:.2f}%)")

    # 5. Header Verification
    print("[5/5] Verifying ESP32 C deployment headers...")
    data_header = os.path.join(REPO_ROOT, "deployment", "esp32", "model", "model_data.h")
    config_header = os.path.join(REPO_ROOT, "deployment", "esp32", "model", "model_config.h")
    assert os.path.isfile(data_header), f"Missing {data_header}"
    assert os.path.isfile(config_header), f"Missing {config_header}"

    with open(config_header, "r") as f:
        config_text = f.read()
    assert "tinyml_dscnn" in config_text.lower(), "model_config.h does not reference TinyML_DSCNN"
    assert "11409" in config_text, "model_config.h parameter mismatch"

    elapsed_ms = (time.time() - start_time) * 1000.0
    print(f"\n==================================================")
    print(f"  [SMOKE TEST PASSED] End-to-end pipeline verified in {elapsed_ms:.1f} ms!")
    print(f"==================================================")


if __name__ == "__main__":
    main()
