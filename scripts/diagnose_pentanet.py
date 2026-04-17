#!/usr/bin/env python3
"""
PentaNet Scaling Diagnostic
Transposed from ShiftQuant to analyze failure at 345M params.

Hypotheses:
  H1 - Layer-wise scale vs block-wise scale
  H2 - Per-layer bucket occupancy (do outer states collapse?)
  H3 - Gradient norm per depth position
"""

import torch
import argparse
import json
from pathlib import Path
from glob import glob


def mse(a, b):
    return (a.float() - b.float()).pow(2).mean().item()


def analyze_layer_buckets(weight: torch.Tensor, name: str) -> dict:
    """Analyze bucket occupancy for a single PentaLinear layer."""
    scale = weight.abs().mean().clamp(min=1e-8)
    w_norm = weight / scale
    w_quant = torch.clamp(torch.round(w_norm), -2, 2)

    buckets = {
        -2: 0.0,
        -1: 0.0,
        0: 0.0,
        1: 0.0,
        2: 0.0,
    }
    total = w_quant.numel()
    for val in [-2, -1, 0, 1, 2]:
        buckets[val] = (w_quant == val).sum().item() / total * 100

    return {
        "name": name,
        "shape": tuple(weight.shape),
        "scale": scale.item(),
        "buckets": buckets,
        "±2_occupancy": buckets[-2] + buckets[2],
    }


def compare_scales(weight: torch.Tensor, block_sizes=(32, 64, 128)) -> dict:
    """Compare layer-wise vs block-wise scaling MSE."""
    w = weight.float()
    out_f, in_f = w.shape

    results = {}
    # Full layer absmean (current impl)
    scale_layer = w.abs().mean()
    w_q = torch.clamp(torch.round(w / scale_layer), -2, 2)
    w_hat = w_q * scale_layer
    results["layer_wise"] = {
        "scale": scale_layer.item(),
        "mse": mse(w, w_hat),
    }

    # Block-wise absmean (like ShiftQuant)
    for bs in block_sizes:
        pad = (bs - in_f % bs) % bs
        wp = torch.nn.functional.pad(w, (0, pad)) if pad else w
        blocks = wp.reshape(out_f, -1, bs)  # [out_f, n_blocks, bs]
        scales = blocks.abs().mean(dim=-1).clamp(min=1e-8)  # [out_f, n_blocks]
        scales = scales.unsqueeze(-1).expand_as(blocks)

        w_scaled = blocks / scales
        w_q = torch.clamp(torch.round(w_scaled), -2, 2)
        w_hat = (w_q * scales).reshape(out_f, -1)[:, :in_f]
        results[f"block_{bs}"] = {
            "scale_mean": scales.mean().item(),
            "mse": mse(w, w_hat),
        }

    return results


def analyze_gradients(model_path: str) -> dict:
    """Analyze recorded gradients from training if available."""
    # Look for saved checkpoints with gradient info
    results = {}
    # This would need checkpoint data - placeholder for now
    return results


def load_model_checkpoints(pattern: str = "models/*345M*.json"):
    """Load any available checkpoints for 345M analysis."""
    files = glob(pattern)
    return files


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PentaNet scaling diagnostic")
    parser.add_argument(
        "--weight-file", type=str, default=None, help="Path to saved weights .pt file"
    )
    parser.add_argument(
        "--json-log",
        type=str,
        default=None,
        help="Path to training log JSON with bucket stats",
    )
    args = parser.parse_args()

    if args.json_log:
        # Analyze from training logs (if bucket stats were saved)
        with open(args.json_log) as f:
            data = json.load(f)
        print("Loaded training log with bucket stats")
        # Would aggregate per-layer data here
        print(json.dumps(data, indent=2))

    elif args.weight_file:
        # Load weights directly
        w = torch.load(args.weight_file)
        print(f"Analyzing weights from {args.weight_file}")

        if isinstance(w, dict) and "weight" in w:
            w = w["weight"]

        layer_analysis = analyze_layer_buckets(w, "input_layer")
        print(f"\n{layer_analysis['name']}:")
        print(f"  Shape: {layer_analysis['shape']}")
        print(f"  Scale: {layer_analysis['scale']:.6f}")
        print(f"  Buckets: {layer_analysis['buckets']}")
        print(f"  ±2 occupancy: {layer_analysis['±2_occupancy']:.1f}%")

        print("\n--- Scale comparison ---")
        scale_comp = compare_scales(w)
        for method, data in scale_comp.items():
            print(f"  {method}: MSE={data['mse']:.6f}")

    else:
        # Show what would be needed
        print("PentaNet Scaling Diagnostic")
        print("=" * 50)
        print("\nUsage:")
        print("  python scripts/diagnose_pentanet.py --weight-file path/to/weights.pt")
        print("  python scripts/diagnose_pentanet.py --json-log path/to/log.json")
        print("\nTo analyze failure at 345M params, need to:")
        print("  1. Save model weights during/after training")
        print("  2. Track bucket occupancy per layer (not just aggregate)")
        print("  3. Save gradient norms per depth position")
        print("\nOr adapt training loop to save layer-wise stats.")
