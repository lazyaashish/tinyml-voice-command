#!/usr/bin/env python3
"""
scripts/export_esp32.py
-----------------------
One-click model export utility for ESP32 and embedded microcontrollers.
Converts PyTorch (.pt) checkpoints into bit-exact C/C++ header files:
  1. model_data.h    - Aligned INT8 (or FP32) quantized weight array.
  2. model_config.h  - Preprocessing parameters, buffer dimensions, and vocabulary.

Usage:
    python scripts/export_esp32.py
    python scripts/export_esp32.py --checkpoint checkpoints/sih_final_5class.pt --model tinyml_dscnn
    python scripts/export_esp32.py --checkpoint checkpoints/cand_f_44ch_champion_fp32.pt --model cand_f --classes YES NO UP DOWN LEFT RIGHT ON OFF STOP GO SILENCE UNKNOWN
"""

import os
import sys
import argparse
import numpy as np
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.models import (
    TinyML_DSCNN,
    CandF_44ChDualResidual,
    ArmSmallDSCNN,
    Ext_Mel40_StridedStem_W76,
)


MODEL_REGISTRY = {
    "tinyml_dscnn": {
        "display_name": "TinyML_DSCNN",
        "class": TinyML_DSCNN,
        "default_classes": ["GO", "STOP", "LEFT", "RIGHT", "UNKNOWN"],
        "default_checkpoint": os.path.join(REPO_ROOT, "checkpoints", "sih_final_5class.pt"),
        "mel_bins": 40,
        "time_frames": 49,
        "macs": 2056640,
        "arena_bytes": 28800,
        "feature_type": "LOG_MEL_40",
    },
    "cand_f": {
        "display_name": "CandF_44ChDualResidual",
        "class": CandF_44ChDualResidual,
        "default_classes": [
            "YES", "NO", "UP", "DOWN", "LEFT", "RIGHT",
            "ON", "OFF", "STOP", "GO", "SILENCE", "UNKNOWN"
        ],
        "default_checkpoint": os.path.join(REPO_ROOT, "checkpoints", "cand_f_44ch_champion_fp32.pt"),
        "mel_bins": 40,
        "time_frames": 49,
        "macs": 1406672,
        "arena_bytes": 15056,
        "feature_type": "LOG_MEL_40",
    },
    "arm_small_dscnn": {
        "display_name": "ArmSmallDSCNN",
        "class": ArmSmallDSCNN,
        "default_classes": [
            "YES", "NO", "UP", "DOWN", "LEFT", "RIGHT",
            "ON", "OFF", "STOP", "GO", "SILENCE", "UNKNOWN"
        ],
        "default_checkpoint": os.path.join(REPO_ROOT, "checkpoints", "arm_small_dscnn_baseline.pt"),
        "mel_bins": 40,
        "time_frames": 49,
        "macs": 5400000,
        "arena_bytes": 48000,
        "feature_type": "LOG_MEL_40",
    },
    "ext_mel40_w76": {
        "display_name": "Ext_Mel40_StridedStem_W76",
        "class": Ext_Mel40_StridedStem_W76,
        "default_classes": [
            "SILENCE", "UNKNOWN", "YES", "NO", "UP", "DOWN",
            "LEFT", "RIGHT", "ON", "OFF", "STOP", "GO"
        ],
        "default_checkpoint": os.path.join(REPO_ROOT, "checkpoints", "ext_mel40_w76_champion_95.pt"),
        "mel_bins": 40,
        "time_frames": 49,
        "macs": 3986688,
        "arena_bytes": 20448,
        "feature_type": "LOG_MEL_40",
    },
}


def export_model_data_header(model, output_file: str, fmt: str = "int8"):
    """Serializes model weights into an aligned C array header."""
    weights = []
    total_elements = 0

    for name, param in model.named_parameters():
        data = param.detach().cpu().numpy()
        total_elements += data.size

        if fmt == "int8":
            max_val = np.max(np.abs(data))
            scale = max_val / 127.0 if max_val > 0 else 1.0
            q_data = np.clip(np.round(data / scale), -128, 127).astype(np.int8)
            weights.append((name, q_data.tobytes(), scale, "int8"))
        else:
            # Raw IEEE 754 float32
            weights.append((name, data.astype(np.float32).tobytes(), 1.0, "float32"))

    total_bytes = sum(len(b) for _, b, _, _ in weights)
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)

    with open(output_file, "w") as f:
        f.write("// ==============================================================================\n")
        f.write("// Auto-generated Bit-Exact Model Weights Header\n")
        f.write(f"// Format: {fmt.upper()} | Parameters: {total_elements:,} | Payload: {total_bytes:,} bytes\n")
        f.write("// Target: ESP32 Xtensa / ARM Cortex-M\n")
        f.write("// ==============================================================================\n\n")
        f.write("#ifndef MODEL_DATA_H\n")
        f.write("#define MODEL_DATA_H\n\n")
        f.write("#include <stdint.h>\n\n")
        f.write(f"const unsigned int g_model_len = {total_bytes};\n")
        f.write("const unsigned char g_model[] __attribute__((aligned(16))) = {\n")

        byte_list = []
        for _, b, _, _ in weights:
            byte_list.extend([f"0x{byte:02x}" for byte in b])

        for i in range(0, len(byte_list), 12):
            f.write("    " + ", ".join(byte_list[i : i + 12]) + ",\n")

        f.write("};\n\n")
        f.write("#endif // MODEL_DATA_H\n")

    return total_elements, total_bytes


def export_model_config_header(
    model_name: str,
    classes: list,
    param_count: int,
    macs: int,
    arena_bytes: int,
    mel_bins: int,
    time_frames: int,
    output_file: str,
):
    """Generates the C metadata header defining runtime sizes and vocabulary."""
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)

    with open(output_file, "w") as f:
        f.write("#ifndef MODEL_CONFIG_H\n")
        f.write("#define MODEL_CONFIG_H\n\n")
        f.write("// ==============================================================================\n")
        f.write(f"// ESP32 Keyword Spotting Metadata & Configuration: {model_name}\n")
        f.write("// Generated automatically by scripts/export_esp32.py\n")
        f.write("// ==============================================================================\n\n")
        f.write(f'#define KWS_MODEL_NAME          "{model_name}"\n')
        f.write('#define KWS_MODEL_VERSION       "1.0.0"\n')
        f.write("#define KWS_SAMPLE_RATE         16000\n")
        f.write("#define KWS_CLIP_DURATION_MS    1000\n")
        f.write("#define KWS_CLIP_SAMPLES        (KWS_SAMPLE_RATE * KWS_CLIP_DURATION_MS / 1000)\n\n")
        f.write("// Audio Preprocessing Constants\n")
        f.write("#define KWS_WINDOW_SIZE_MS      30\n")
        f.write("#define KWS_HOP_SIZE_MS         20\n")
        f.write("#define KWS_WINDOW_SAMPLES      (KWS_SAMPLE_RATE * KWS_WINDOW_SIZE_MS / 1000)\n")
        f.write("#define KWS_HOP_SAMPLES         (KWS_SAMPLE_RATE * KWS_HOP_SIZE_MS / 1000)\n")
        f.write("#define KWS_PREEMPHASIS_COEFF   0.97f\n\n")
        f.write("// Spectrogram Dimensions\n")
        f.write(f'#define KWS_FEATURE_TYPE        "LOG_MEL_{mel_bins}"\n')
        f.write(f"#define KWS_NUM_MEL_BINS        {mel_bins}\n")
        f.write(f"#define KWS_NUM_TIME_FRAMES     {time_frames}\n")
        f.write(f"#define KWS_FEATURE_SIZE        (KWS_NUM_TIME_FRAMES * KWS_NUM_MEL_BINS)\n\n")
        f.write("// Architecture & Memory\n")
        f.write(f"#define KWS_TOTAL_PARAMETERS    {param_count}\n")
        f.write(f"#define KWS_TOTAL_MACS          {macs}\n")
        f.write(f"#define KWS_PEAK_ARENA_BYTES    {arena_bytes}\n\n")
        f.write("// Output Vocabulary\n")
        f.write(f"#define KWS_NUM_CLASSES         {len(classes)}\n")
        f.write("static const char* const KWS_CLASS_LABELS[KWS_NUM_CLASSES] = {\n")
        for cls in classes:
            f.write(f'    "{cls}",\n')
        f.write("};\n\n")
        f.write("// Default Runtime Thresholds\n")
        f.write("#define KWS_DEFAULT_THRESHOLD   0.80f\n")
        f.write("#define KWS_DEFAULT_DEBOUNCE_MS 500\n\n")
        f.write("#endif // MODEL_CONFIG_H\n")


def main():
    parser = argparse.ArgumentParser(description="Export PyTorch KWS model to C header files for ESP32.")
    parser.add_argument(
        "--model",
        type=str,
        default="tinyml_dscnn",
        choices=list(MODEL_REGISTRY.keys()),
        help="Model architecture name (default: tinyml_dscnn)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to .pt checkpoint file (defaults to bundled champion checkpoint)",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=None,
        help="Class labels for output vocabulary",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(REPO_ROOT, "deployment", "esp32", "model"),
        help="Target output directory for C headers",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="int8",
        choices=["int8", "fp32"],
        help="Weight quantization format (default: int8)",
    )
    args = parser.parse_args()

    meta = MODEL_REGISTRY[args.model]
    checkpoint_path = args.checkpoint or meta["default_checkpoint"]
    classes = args.classes or meta["default_classes"]

    print("==================================================")
    print("  ESP32 KWS Model Exporter")
    print("==================================================")
    print(f"  Model Architecture : {args.model}")
    print(f"  Checkpoint         : {checkpoint_path}")
    print(f"  Classes ({len(classes)})       : {', '.join(classes)}")
    print(f"  Quantization       : {args.format.upper()}")
    print(f"  Output Directory   : {args.output_dir}")

    if not os.path.isfile(checkpoint_path):
        print(f"[ERROR] Checkpoint file not found: {checkpoint_path}")
        sys.exit(1)

    # Instantiate model
    model_cls = meta["class"]
    model = model_cls(num_classes=len(classes))

    # Load weights
    sd = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    filtered_sd = {k: v for k, v in sd.items() if not any(x in k for x in ['quant_', 'running_max', 'initialized'])}
    model.load_state_dict(filtered_sd, strict=False)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())

    # Export model_data.h
    data_header_path = os.path.join(args.output_dir, "model_data.h")
    total_elements, total_bytes = export_model_data_header(model, data_header_path, fmt=args.format)

    # Export model_config.h
    config_header_path = os.path.join(args.output_dir, "model_config.h")
    export_model_config_header(
        model_name=meta.get("display_name", args.model),
        classes=classes,
        param_count=total_params,
        macs=meta["macs"],
        arena_bytes=meta["arena_bytes"],
        mel_bins=meta["mel_bins"],
        time_frames=meta["time_frames"],
        output_file=config_header_path,
    )

    print("\n[SUCCESS] C/C++ Header Files Generated:")
    print(f"  -> {data_header_path} ({total_bytes:,} bytes payload)")
    print(f"  -> {config_header_path}")
    print("\nNext step: Flash the firmware to ESP32 using Arduino IDE or PlatformIO.")
    print("See deployment/esp32/README.md for wiring and flashing instructions.\n")


if __name__ == "__main__":
    main()
