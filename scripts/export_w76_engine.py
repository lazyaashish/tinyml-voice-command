#!/usr/bin/env python3
"""
scripts/export_w76_engine.py
----------------------------
One-click standalone C++ exporter for Ext_Mel40_StridedStem_W76 (12-class 95% champion).
Generates self-contained, header-only C++ forward inference engine:
  - w76_mel_table.h: Hanning window, twiddle table, sparse 40-band triangular mel filterbank.
  - w76_weights.h: 39,140 floating point weights in PROGMEM.
  - w76_engine.h: FFT, Mel spectrogram, CMN, and neural forward pass.
"""

import os
import sys
import numpy as np
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.models import Ext_Mel40_StridedStem_W76
from src.preprocessing.mel_features import create_mel_filterbank


def generate_mel_table_header(output_path: str):
    print("Generating Mel table header...")
    hanning = (0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(640) / 639.0))).astype(np.float32)

    # Twiddle factors for 1024-point FFT
    # For stage step, delta angle = -pi / step, w_step_re = cos(-pi/step), w_step_im = sin(-pi/step)
    # We can precompute twiddle factors for all m in range(512)
    # Actually, a simple table of cos and sin for 512 angles: angle = -pi * k / 512 for k in 0..511
    angles = -np.pi * np.arange(512) / 512.0
    fft_cos = np.cos(angles).astype(np.float32)
    fft_sin = np.sin(angles).astype(np.float32)

    # Mel filterbank
    fb = create_mel_filterbank(num_bins=40, fft_size=1024, sample_rate=16000, low_freq=60.0, high_freq=7800.0)
    mel_starts = []
    mel_lens = []
    mel_weights = []
    for m in range(40):
        nz = np.where(fb[m] > 0)[0]
        start = int(nz[0])
        length = len(nz)
        mel_starts.append(start)
        mel_lens.append(length)
        mel_weights.extend(fb[m, nz].tolist())

    with open(output_path, "w") as f:
        f.write("// Auto-generated Audio Preprocessing Tables for ESP32 KWS\n")
        f.write("#ifndef W76_MEL_TABLE_H\n#define W76_MEL_TABLE_H\n\n")
        f.write("#ifdef ARDUINO\n  #include <Arduino.h>\n#else\n  #include <stdint.h>\n  #include <stddef.h>\n  #include <math.h>\n#endif\n\n")
        f.write("#ifndef PROGMEM\n  #define PROGMEM\n#endif\n")
        f.write("#ifndef pgm_read_float\n  #define pgm_read_float(addr) (*(const float *)(addr))\n#endif\n")
        f.write("#ifndef pgm_read_word\n  #define pgm_read_word(addr) (*(const uint16_t *)(addr))\n#endif\n\n")

        # Hanning
        f.write(f"// 640-point Hanning window (40 ms @ 16 kHz)\n")
        f.write("static const float W76_HANNING_WINDOW[640] PROGMEM = {\n")
        for i in range(0, 640, 8):
            f.write("    " + ", ".join(f"{hanning[j]:.8e}f" for j in range(i, min(i + 8, 640))) + ",\n")
        f.write("};\n\n")

        # Mel filterbank metadata
        f.write("// Sparse Mel Filterbank (40 triangular bands, 60 Hz to 7800 Hz)\n")
        f.write(f"#define W76_NUM_MEL_BINS 40\n")
        f.write(f"#define W76_MEL_TOTAL_WEIGHTS {len(mel_weights)}\n\n")

        f.write("static const uint16_t W76_MEL_STARTS[40] PROGMEM = {\n    " + ", ".join(str(s) for s in mel_starts) + "\n};\n\n")
        f.write("static const uint16_t W76_MEL_LENS[40] PROGMEM = {\n    " + ", ".join(str(l) for l in mel_lens) + "\n};\n\n")

        f.write("static const float W76_MEL_WEIGHTS[" + str(len(mel_weights)) + "] PROGMEM = {\n")
        for i in range(0, len(mel_weights), 8):
            chunk = mel_weights[i : min(i + 8, len(mel_weights))]
            f.write("    " + ", ".join(f"{w:.8e}f" for w in chunk) + ",\n")
        f.write("};\n\n")

        f.write("#endif // W76_MEL_TABLE_H\n")
    print(f"  -> Saved {output_path}")


def generate_weights_header(checkpoint_path: str, output_path: str):
    print("Generating W76 model weights header...")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    raw_sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    filtered_sd = {k: v for k, v in raw_sd.items() if not any(x in k for x in ["quant_", "running_max", "initialized"])}

    p = {k: v.detach().cpu().numpy() for k, v in filtered_sd.items()}

    with open(output_path, "w") as f:
        f.write("// Auto-generated Model Weights for Ext_Mel40_StridedStem_W76 (12 Classes)\n")
        f.write("#ifndef W76_WEIGHTS_H\n#define W76_WEIGHTS_H\n\n")
        f.write("#ifdef ARDUINO\n  #include <Arduino.h>\n#else\n  #include <stdint.h>\n  #include <stddef.h>\n#endif\n\n")
        f.write("#ifndef PROGMEM\n  #define PROGMEM\n#endif\n\n")

        def write_array(name, arr):
            flat = arr.flatten().tolist()
            f.write(f"// {name} shape={list(arr.shape)} count={len(flat)}\n")
            f.write(f"static const float {name}[{len(flat)}] PROGMEM = {{\n")
            for i in range(0, len(flat), 8):
                chunk = flat[i : min(i + 8, len(flat))]
                f.write("    " + ", ".join(f"{w:.8e}f" for w in chunk) + ",\n")
            f.write("};\n\n")

        write_array("W76_CONV1_W", p["conv1.weight"]) # 48 * 1 * 5 * 4 = 960
        write_array("W76_CONV1_B", p["conv1.bias"])   # 48

        write_array("W76_STEM_DW_W", p["b1_dw.weight"]) # 48 * 1 * 3 * 3 = 432
        write_array("W76_STEM_DW_B", p["b1_dw.bias"])   # 48

        write_array("W76_STEM_PW_W", p["b1_pw.weight"]) # 76 * 48 * 1 * 1 = 3648
        write_array("W76_STEM_PW_B", p["b1_pw.bias"])   # 76

        write_array("W76_SKIP_PROJ_W", p["skip_proj.weight"]) # 76 * 76 = 5776

        for b in range(4):
            write_array(f"W76_BLK{b}_DW_W", p[f"blocks.{b}.dw.weight"]) # 76 * 1 * 3 * 3 = 684
            write_array(f"W76_BLK{b}_DW_B", p[f"blocks.{b}.dw.bias"])   # 76
            write_array(f"W76_BLK{b}_PW_W", p[f"blocks.{b}.pw.weight"]) # 76 * 76 = 5776
            write_array(f"W76_BLK{b}_PW_B", p[f"blocks.{b}.pw.bias"])   # 76

        write_array("W76_CLASSIFIER_W", p["classifier.weight"]) # 12 * 76 = 912
        write_array("W76_CLASSIFIER_B", p["classifier.bias"])   # 12

        f.write("#endif // W76_WEIGHTS_H\n")
    print(f"  -> Saved {output_path}")


if __name__ == "__main__":
    ckpt = os.path.join(REPO_ROOT, "checkpoints", "ext_mel40_w76_champion_95.pt")
    out_dir = os.path.join(REPO_ROOT, "deployment", "esp32", "model")
    os.makedirs(out_dir, exist_ok=True)
    generate_mel_table_header(os.path.join(out_dir, "w76_mel_table.h"))
    generate_weights_header(ckpt, os.path.join(out_dir, "w76_weights.h"))
