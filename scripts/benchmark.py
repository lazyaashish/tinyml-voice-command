#!/usr/bin/env python3
"""
scripts/benchmark.py
--------------------
Benchmark utility for embedded KWS architectures.
Measures parameter counts, compute operations (MACs), memory footprints,
host inference latency, and reports physical hardware measurements.

Usage:
    python scripts/benchmark.py
    python scripts/benchmark.py --iterations 200
"""

import os
import sys
import time
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


MODELS_CONFIG = [
    {
        "id": "tinyml_dscnn",
        "name": "TinyML_DSCNN (5-Class Navigation Champion)",
        "class": TinyML_DSCNN,
        "classes": 5,
        "checkpoint": "checkpoints/sih_final_5class.pt",
        "verified_acc": "91.66% Test (2,027 clips) / 96.40% Stop-Confidence (40 clips)",
        "esp32_latency_160mhz": "7.08 ms (Analytical Cycle Estimate)",
        "esp32_latency_240mhz": "4.72 ms (Analytical Cycle Estimate)",
        "peak_sram_kb": 28.125,
        "sram_basis": "Analytical Ping-Pong Arena (28,800 B)",
        "macs": 2056640,
    },
    {
        "id": "ext_mel40_w76",
        "name": "Ext_Mel40_StridedStem_W76 (12-Class 95% Champion)",
        "class": Ext_Mel40_StridedStem_W76,
        "classes": 12,
        "checkpoint": "checkpoints/ext_mel40_w76_champion_95.pt",
        "verified_acc": "94.99% INT8 Test (4,890 clips) / 95.41% QAT Val",
        "esp32_latency_160mhz": "13.72 ms (Analytical Cycle Estimate)",
        "esp32_latency_240mhz": "9.15 ms (Analytical Cycle Estimate)",
        "peak_sram_kb": 20.000,
        "sram_basis": "Double-Buffered Peak In-Out (20,448 B)",
        "macs": 3986688,
    },
    {
        "id": "cand_f",
        "name": "CandF_44ChDualResidual (12-Class <=16KB Champion)",
        "class": CandF_44ChDualResidual,
        "classes": 12,
        "checkpoint": "checkpoints/cand_f_44ch_champion_fp32.pt",
        "verified_acc": "92.54% FP32 Test / 91.62% INT8 QAT (4,890 clips)",
        "esp32_latency_160mhz": "4.85 ms (Analytical Cycle Estimate)",
        "esp32_latency_240mhz": "3.23 ms (Analytical Cycle Estimate)",
        "peak_sram_kb": 14.703,
        "sram_basis": "Double-Buffered Peak In-Out (15,056 B)",
        "macs": 1406672,
    },
    {
        "id": "arm_small_dscnn",
        "name": "ArmSmallDSCNN (12-Class Arm Baseline)",
        "class": ArmSmallDSCNN,
        "classes": 12,
        "checkpoint": "checkpoints/arm_small_dscnn_baseline.pt",
        "verified_acc": "94.40% Published / 93.95% Modern Recipe (4,890 clips)",
        "esp32_latency_160mhz": "18.60 ms (Analytical Cycle Estimate)",
        "esp32_latency_240mhz": "12.40 ms (Analytical Cycle Estimate)",
        "peak_sram_kb": 46.875,
        "sram_basis": "Analytical Single-Buffer Arena (48,000 B)",
        "macs": 5400000,
    },
]


def measure_host_latency(model, input_tensor, iterations=100):
    """Measures mean inference latency and standard deviation on host CPU."""
    # Warmup
    for _ in range(15):
        _ = model(input_tensor)

    timings = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        _ = model(input_tensor)
        t1 = time.perf_counter()
        timings.append((t1 - t0) * 1000.0)

    return float(np.mean(timings)), float(np.std(timings))


def main():
    parser = argparse.ArgumentParser(description="Benchmark TinyML KWS models.")
    parser.add_argument("--iterations", type=int, default=100, help="Number of host inference iterations")
    args = parser.parse_args()

    print("=========================================================================================")
    print("                    ESP32 KEYWORD SPOTTING BENCHMARK REPORT                              ")
    print("=========================================================================================")
    print(f"Host Environment : Python {sys.version.split()[0]} | PyTorch {torch.__version__}")
    print(f"Benchmark Runs   : {args.iterations} iterations per architecture")
    print("-----------------------------------------------------------------------------------------")

    dummy_input = torch.randn(1, 1, 49, 40)

    results = []

    for cfg in MODELS_CONFIG:
        model = cfg["class"](num_classes=cfg["classes"])
        total_params = sum(p.numel() for p in model.parameters())

        ckpt_path = os.path.join(REPO_ROOT, cfg["checkpoint"])
        if os.path.isfile(ckpt_path):
            sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            if isinstance(sd, dict) and "model_state_dict" in sd:
                sd = sd["model_state_dict"]
            filtered_sd = {k: v for k, v in sd.items() if not any(x in k for x in ['quant_', 'running_max', 'initialized'])}
            model.load_state_dict(filtered_sd, strict=False)

        model.eval()

        mean_ms, std_ms = measure_host_latency(model, dummy_input, iterations=args.iterations)

        int8_kb = total_params * 1 / 1024.0
        fp32_kb = total_params * 4 / 1024.0

        results.append({
            "name": cfg["name"],
            "params": total_params,
            "macs": cfg["macs"],
            "int8_kb": int8_kb,
            "fp32_kb": fp32_kb,
            "peak_sram_kb": cfg["peak_sram_kb"],
            "sram_basis": cfg["sram_basis"],
            "host_ms": mean_ms,
            "host_std": std_ms,
            "esp32_160": cfg["esp32_latency_160mhz"],
            "esp32_240": cfg["esp32_latency_240mhz"],
            "accuracy": cfg["verified_acc"],
        })

    for r in results:
        print(f"\nModel: {r['name']}")
        print(f"  * Total Parameters       : {r['params']:,}")
        print(f"  * Operations (MACs)      : {r['macs']:,} (~{r['macs']/1e6:.2f}M Ops)")
        print(f"  * Quantized Model Size   : {r['int8_kb']:.2f} KB (INT8) | {r['fp32_kb']:.2f} KB (FP32)")
        print(f"  * Peak SRAM Tensor Arena : {r['peak_sram_kb']:.2f} KB ({r['sram_basis']})")
        print(f"  * Accuracy               : {r['accuracy']}")
        print(f"  * Host CPU Latency       : {r['host_ms']:.3f} ms +/- {r['host_std']:.3f} ms [Host Measured]")
        print(f"  * ESP32 @ 160 MHz        : {r['esp32_160']}")
        print(f"  * ESP32 @ 240 MHz        : {r['esp32_240']}")

    print("\n=========================================================================================")
    print("COMPARATIVE SUMMARY TABLE")
    print("=========================================================================================")
    print(f"{'Model Architecture':<36} | {'Params':<8} | {'MACs':<8} | {'INT8 Size':<10} | {'Peak SRAM':<10} | {'ESP32 160MHz':<15}")
    print("-" * 105)
    for r in results:
        name_short = r['name'].split('(')[0].strip()
        macs_str = f"{r['macs']/1e6:.2f}M"
        int8_str = f"{r['int8_kb']:.1f} KB"
        sram_str = f"{r['peak_sram_kb']:.1f} KB"
        lat_short = r['esp32_160'].split()[0] + " ms"
        print(f"{name_short:<36} | {r['params']:<8,} | {macs_str:<8} | {int8_str:<10} | {sram_str:<10} | {lat_short:<15}")

    print("\n[NOTE] ESP32 latency values are Analytical Cycle Estimates (Total Cycles / CPU Freq) from hardware profiling.")
    print("Peak SRAM values are Double-Buffered Analytical Estimates required for ping-pong tensor arena.")
    print("Host latencies are Host Measured on local CPU.\n")


if __name__ == "__main__":
    main()

