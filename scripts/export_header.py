"""
Top-level script to export PyTorch checkpoints to C/C++ static byte array headers.
"""

import os
import sys
import argparse

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from deployment.export_c_header import export_model_to_c_header, _load_sd
from src.models import TinyML_DSCNN, CandF_44ChDualResidual

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export C header for ESP32 deployment.")
    parser.add_argument("--model-type", type=str, default="sih", choices=["sih", "cand_f"])
    parser.add_argument("--checkpoint", type=str, default="checkpoints/sih_final_5class.pt")
    parser.add_argument("--output", type=str, default="deployment/headers/sih_model_data.h")
    args = parser.parse_args()

    import torch
    if args.model_type == "sih":
        model = TinyML_DSCNN(num_classes=5)
    else:
        model = CandF_44ChDualResidual(num_classes=12)

    sd = _load_sd(args.checkpoint)
    model.load_state_dict(sd)
    model.eval()

    export_model_to_c_header(model, f"{args.model_type}_model_data", args.output)

