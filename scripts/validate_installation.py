#!/usr/bin/env python3
"""
scripts/validate_installation.py
--------------------------------
Comprehensive system and environment health check.
Validates dependencies, model definitions, pretrained checkpoints,
and ESP32 deployment artifacts.

Usage:
    python scripts/validate_installation.py
"""

import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def test_section(name):
    print(f"\n--- [CHECK] {name} ---")


def main():
    print("==================================================")
    print("  ESP32 KWS System & Installation Validator")
    print("==================================================")

    all_passed = True

    # 1. Python Environment Check
    test_section("Python Environment")
    py_version = sys.version.split()[0]
    print(f"  Python Version: {py_version}")
    if sys.version_info >= (3, 8):
        print("  -> Status: OK (Python >= 3.8)")
    else:
        print("  -> [FAIL]: Python >= 3.8 required.")
        all_passed = False

    # 2. Dependencies
    test_section("Core Dependencies")
    dependencies = [
        ("torch", "PyTorch"),
        ("numpy", "NumPy"),
        ("scipy", "SciPy"),
    ]
    for mod_name, label in dependencies:
        try:
            mod = __import__(mod_name)
            ver = getattr(mod, "__version__", "installed")
            print(f"  {label:<12}: {ver} (OK)")
        except ImportError as e:
            print(f"  {label:<12}: NOT FOUND ({e})")
            all_passed = False

    # Optional torchaudio / sounddevice
    for mod_name, label in [("torchaudio", "TorchAudio"), ("sounddevice", "SoundDevice")]:
        try:
            mod = __import__(mod_name)
            ver = getattr(mod, "__version__", "installed")
            print(f"  {label:<12}: {ver} (Optional / Available)")
        except ImportError:
            print(f"  {label:<12}: Not installed (Optional)")

    # 3. Model Architecture Imports
    test_section("Model Architecture Registry")
    try:
        from src.models import (
            TinyML_DSCNN,
            CandF_44ChDualResidual,
            ArmSmallDSCNN,
            Ext_Mel40_StridedStem_W76,
            UltraLowSRAM_MFCC10,
        )
        print("  -> TinyML_DSCNN          : Imported Successfully")
        print("  -> CandF_44ChDualResidual: Imported Successfully")
        print("  -> ArmSmallDSCNN         : Imported Successfully")
        print("  -> Ext_Mel40_StridedStem_W76: Imported Successfully")
        print("  -> UltraLowSRAM_MFCC10   : Imported Successfully")
    except Exception as e:
        print(f"  -> [FAIL] Model import error: {e}")
        all_passed = False

    # 4. Preprocessing Modules
    test_section("Preprocessing Pipeline")
    try:
        from src.preprocessing import LogMelSpectrogramExtractor
        extractor = LogMelSpectrogramExtractor()
        print("  -> LogMelSpectrogramExtractor: Initialized Successfully (40 Mel bins, 16 kHz)")
    except Exception as e:
        print(f"  -> [FAIL] Preprocessing error: {e}")
        all_passed = False

    # 5. Checkpoints Integrity
    test_section("Pretrained Model Checkpoints")
    checkpoints = [
        ("checkpoints/sih_final_5class.pt", TinyML_DSCNN, 5, "TinyML_DSCNN (5-Class Navigation)"),
        ("checkpoints/ext_mel40_w76_champion_95.pt", Ext_Mel40_StridedStem_W76, 12, "Ext_Mel40_W76 (12-Class 95% Champion)"),
        ("checkpoints/cand_f_44ch_champion_fp32.pt", CandF_44ChDualResidual, 12, "CandF_44Ch Dual Residual (12-Class FP32)"),
        ("checkpoints/arm_small_dscnn_baseline.pt", ArmSmallDSCNN, 12, "Arm Small DS-CNN (Baseline)"),
    ]

    import torch

    for rel_path, model_cls, num_classes, desc in checkpoints:
        full_path = os.path.join(REPO_ROOT, rel_path)
        if not os.path.isfile(full_path):
            print(f"  -> [FAIL] Missing checkpoint: {rel_path}")
            all_passed = False
            continue

        try:
            model = model_cls(num_classes=num_classes)
            sd = torch.load(full_path, map_location="cpu", weights_only=False)
            if isinstance(sd, dict) and "model_state_dict" in sd:
                sd = sd["model_state_dict"]
            filtered_sd = {k: v for k, v in sd.items() if not any(x in k for x in ['quant_', 'running_max', 'initialized'])}
            model.load_state_dict(filtered_sd, strict=False)
            model.eval()
            param_count = sum(p.numel() for p in model.parameters())
            file_size_kb = os.path.getsize(full_path) / 1024
            print(f"  -> {desc:<40}: OK ({param_count:,} params, {file_size_kb:.1f} KB)")
        except Exception as e:
            print(f"  -> [FAIL] Could not load {rel_path}: {e}")
            all_passed = False

    # 6. ESP32 Deployment Package
    test_section("ESP32 Deployment Package")
    deployment_files = [
        "deployment/esp32/firmware/esp32_kws.ino",
        "deployment/esp32/model/model_config.h",
        "deployment/esp32/model/model_data.h",
        "deployment/esp32/platformio/platformio.ini",
        "deployment/esp32/README.md",
    ]
    for rel_path in deployment_files:
        full_path = os.path.join(REPO_ROOT, rel_path)
        if os.path.isfile(full_path):
            size_kb = os.path.getsize(full_path) / 1024
            print(f"  -> {rel_path:<45}: OK ({size_kb:.1f} KB)")
        else:
            print(f"  -> [FAIL] Missing file: {rel_path}")
            all_passed = False

    print("\n==================================================")
    if all_passed:
        print("  [SUCCESS] All system validation checks passed!")
        print("  The repository is ready for development & ESP32 flashing.")
        print("==================================================")
        sys.exit(0)
    else:
        print("  [WARNING] One or more checks failed. Review errors above.")
        print("==================================================")
        sys.exit(1)


if __name__ == "__main__":
    main()

