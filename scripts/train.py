"""
Universal Training Entry Point for SIH ESP32 Keyword Spotting.
Supports training the 5-class SIH edge model or 12-class TinyML benchmark models.
"""

import os
import sys
import argparse
import yaml
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.models import TinyML_DSCNN, CandF_44ChDualResidual


def run_smoke_train(model_type: str = "sih"):
    print(f"Running smoke training pass for {model_type.upper()} architecture...")
    if model_type == "sih":
        model = TinyML_DSCNN(num_classes=5)
        dummy_in = torch.randn(4, 1, 49, 40)
        target = torch.tensor([0, 1, 2, 3])
    else:
        model = CandF_44ChDualResidual(num_classes=12)
        dummy_in = torch.randn(4, 1, 49, 10)
        target = torch.tensor([0, 1, 2, 3])

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()
    optimizer.zero_grad()
    out = model(dummy_in)
    loss = criterion(out, target)
    loss.backward()
    optimizer.step()

    print(f"Smoke pass successful! Initial Loss: {loss.item():.4f}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SIH KWS or Benchmark models.")
    parser.add_argument("--config", type=str, default="configs/sih_5class.yaml", help="Path to config yaml")
    parser.add_argument("--smoke-test", action="store_true", help="Run a fast 1-batch synthetic test")
    parser.add_argument("--model-type", type=str, default="sih", choices=["sih", "benchmark"])
    args = parser.parse_args()

    if args.smoke_test:
        run_smoke_train(args.model_type)
    else:
        print(f"Loading config from {args.config}...")
        if os.path.exists(args.config):
            with open(args.config, "r") as f:
                cfg = yaml.safe_load(f)
            print(f"Config loaded for project: {cfg.get('project', {}).get('name')}")
        print("To run full training on Speech Commands, download dataset first via scripts/download_dataset.py")
        run_smoke_train(args.model_type)
