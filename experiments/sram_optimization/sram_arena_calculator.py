"""
Analytical Static SRAM Tensor Arena Calculator for Microcontroller KWS.
Computes ping-pong double buffering memory footprints across layer boundaries.
Target: Strictly verify <= 16 KB static SRAM arena for microcontroller deployment.
"""

import math


def calculate_sram_arena(input_shape, layers):
    """
    Simulates CMSIS-NN / TFLM static memory arena allocation:
    Peak SRAM = Max(layer_i_output + layer_{i+1}_output + scratch_buffers)
    """
    current_h, current_w, current_c = input_shape
    prev_bytes = current_h * current_w * current_c  # INT8 = 1 byte per element
    peak_arena = prev_bytes

    print(f"Input: {input_shape} -> {prev_bytes:,} Bytes")

    for i, layer in enumerate(layers):
        ltype = layer.get("type")
        out_c = layer.get("out_c", current_c)
        stride_h, stride_w = layer.get("stride", (1, 1))

        out_h = math.ceil(current_h / stride_h)
        out_w = math.ceil(current_w / stride_w)
        out_bytes = out_h * out_w * out_c

        # Scratch buffer for im2col / depthwise packing
        scratch = layer.get("scratch_bytes", 0)
        active_memory = prev_bytes + out_bytes + scratch

        if active_memory > peak_arena:
            peak_arena = active_memory

        print(
            f"Layer {i+1:02d} [{ltype:12s}]: ({current_h:2d}x{current_w:2d}x{current_c:2d}) -> "
            f"({out_h:2d}x{out_w:2d}x{out_c:2d}) | Arena: {active_memory:,} Bytes"
        )

        current_h, current_w, current_c = out_h, out_w, out_c
        prev_bytes = out_bytes

    print("-" * 60)
    print(f"Total Peak Static SRAM Arena: {peak_arena:,} Bytes ({peak_arena / 1024:.2f} KB)")
    fits_16k = peak_arena <= 16384
    print(f"Strict <= 16 KB Ceiling Compliance: {'PASS [VERIFIED]' if fits_16k else 'FAIL'}")
    return peak_arena


if __name__ == "__main__":
    # CandF 44-channel Dual Residual architecture profile
    cand_f_layers = [
        {"type": "Init Conv", "out_c": 44, "stride": (2, 1), "scratch_bytes": 44 * 10 * 4},
        {"type": "DS Block 1", "out_c": 44, "stride": (1, 1), "scratch_bytes": 44 * 3 * 3},
        {"type": "DS Block 2", "out_c": 44, "stride": (2, 2), "scratch_bytes": 44 * 3 * 3},
        {"type": "DS Block 3a", "out_c": 44, "stride": (1, 1), "scratch_bytes": 44 * 3 * 3},
        {"type": "DS Block 3b", "out_c": 44, "stride": (1, 1), "scratch_bytes": 44 * 3 * 3},
        {"type": "DS Block 4", "out_c": 44, "stride": (1, 1), "scratch_bytes": 44 * 3 * 3},
    ]

    print("=== CandF_44ChDualResidual SRAM Memory Audit ===")
    calculate_sram_arena(input_shape=(49, 10, 1), layers=cand_f_layers)

