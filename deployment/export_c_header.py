"""
C Header Export Utility for Embedded Microcontroller Deployment.
Quantizes floating point PyTorch state dictionaries to symmetric 8-bit integer C arrays.
Target: ESP32 Xtensa / ARM Cortex-M microcontrollers.
"""

import os
import sys
import argparse
import numpy as np
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.models import TinyML_DSCNN, CandF_44ChDualResidual


def export_model_to_c_header(model, array_name: str, output_path: str):
    """Exports model weights as an aligned uint8/int8 C array."""
    weights = []
    for name, param in model.named_parameters():
        data = param.detach().cpu().numpy()
        # Symmetric INT8 quantization
        scale = np.max(np.abs(data)) / 127.0 if np.max(np.abs(data)) > 0 else 1.0
        q_data = np.clip(np.round(data / scale), -128, 127).astype(np.int8)
        weights.append((name, q_data.tobytes(), scale))

    total_bytes = sum(len(b) for _, b, _ in weights)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    with open(output_path, "w") as f:
        f.write(f"// Auto-generated bit-exact model data header: {array_name}\n")
        f.write(f"// Total weights payload: {total_bytes} bytes\n\n")
        f.write("#ifndef " + array_name.upper() + "_DATA_H\n")
        f.write("#define " + array_name.upper() + "_DATA_H\n\n")
        f.write('#include <stdint.h>\n\n')
        f.write(f"const unsigned int g_{array_name}_len = {total_bytes};\n")
        f.write(f"const unsigned char g_{array_name}[] __attribute__((aligned(16))) = {{\n")

        byte_list = []
        for _, b, _ in weights:
            byte_list.extend([f"0x{byte:02x}" for byte in b])

        for i in range(0, len(byte_list), 12):
            f.write("    " + ", ".join(byte_list[i : i + 12]) + ",\n")

        f.write("};\n\n")
        f.write("#endif // " + array_name.upper() + "_DATA_H\n")

    print(f"Successfully exported {array_name} ({total_bytes} bytes) to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export PyTorch model weights to C header.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint")
    parser.add_argument("--model-type", type=str, default="sih", choices=["sih", "cand_f"])
    parser.add_argument("--output", type=str, required=True, help="Output .h path")
    args = parser.parse_args()

    if args.model_type == "sih":
        m = TinyML_DSCNN(num_classes=5)
    else:
        m = CandF_44ChDualResidual(num_classes=12)

    sd = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    m.load_state_dict(sd)
    m.eval()

    export_model_to_c_header(m, f"{args.model_type}_model_data", args.output)
