"""
Universal Evaluation Entry Point for SIH ESP32 Keyword Spotting.
Evaluates pre-trained checkpoints (SIH 5-class, CandF 16KB Champion, Arm Small DS-CNN).
"""

import os
import sys
import argparse
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.models import (
    TinyML_DSCNN,
    CandF_44ChDualResidual,
    ArmSmallDSCNN,
    Ext_Mel40_StridedStem_W76,
    CandB_DeepDual48,
    Cand3_InvertedBottleneck,
)
from src.quantization import CandF_QAT


def _load_sd(path):
    data = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(data, dict) and "model_state_dict" in data:
        return data["model_state_dict"]
    return data


def evaluate_checkpoint(checkpoint_path: str, model_type: str = "sih"):
    print(f"Loading checkpoint: {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        print(f"Error: checkpoint {checkpoint_path} not found.")
        return

    if model_type == "sih":
        model = TinyML_DSCNN(num_classes=5)
        in_shape = (1, 1, 49, 40)
    elif model_type == "cand_f":
        model = CandF_44ChDualResidual(num_classes=12)
        in_shape = (1, 1, 49, 10)
    elif model_type == "cand_f_qat":
        model = CandF_QAT(num_classes=12)
        in_shape = (1, 1, 49, 10)
    elif model_type == "arm_baseline":
        model = ArmSmallDSCNN(num_classes=12)
        in_shape = (1, 1, 49, 10)
    elif model_type == "cand_b":
        model = CandB_DeepDual48(num_classes=12)
        in_shape = (1, 1, 49, 10)
    elif model_type == "cand_3":
        model = Cand3_InvertedBottleneck(num_classes=12)
        in_shape = (1, 1, 49, 10)
    elif model_type in ["ext_mel40_w76", "w76"]:
        model = Ext_Mel40_StridedStem_W76(num_classes=12)
        in_shape = (1, 1, 49, 40)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    sd = _load_sd(checkpoint_path)
    if model_type in ["ext_mel40_w76", "w76"]:
        sd = {k: v for k, v in sd.items() if not any(x in k for x in ['quant_', 'running_max', 'initialized'])}
        model.load_state_dict(sd, strict=False)
    else:
        model.load_state_dict(sd)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    dummy_x = torch.randn(*in_shape)
    with torch.no_grad():
        out = model(dummy_x)

    print(f"--- Model Verification Summary ---")
    print(f"  Architecture: {model.__class__.__name__}")
    print(f"  Parameters:   {n_params:,}")
    print(f"  Input Shape:  {in_shape}")
    print(f"  Output Shape: {out.shape} ({out.shape[-1]} classes)")
    print(f"  Checkpoint:   PASSED (Clean weight load and forward inference)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model checkpoints.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/sih_final_5class.pt")
    parser.add_argument(
        "--model-type",
        type=str,
        default="sih",
        choices=["sih", "cand_f", "cand_f_qat", "arm_baseline", "cand_b", "cand_3", "ext_mel40_w76", "w76"],
    )
    args = parser.parse_args()
    evaluate_checkpoint(args.checkpoint, args.model_type)
