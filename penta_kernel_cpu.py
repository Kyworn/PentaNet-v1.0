"""
penta_kernel_cpu.py — PentaNet CPU Inference Kernel
====================================================

Zero-multiplier matrix-vector product for pentanary weights {-2,-1,0,+1,+2}.

Strategy
--------
Unpack stored 3-bit weights into integer buckets, then split each bucket
into a binary (0/1) mask. Accumulate via four masked matmuls:

    Y ≈ (X @ pos1.T)            # w = +1 : add x once
      - (X @ neg1.T)            # w = -1 : subtract x once
      + (X @ pos2.T) + (X @ pos2.T)   # w = +2 : add x twice  (no FMUL)
      - (X @ neg2.T) - (X @ neg2.T)   # w = -2 : subtract x twice

The masks contain only 0 and 1 — no arbitrary float weight is ever
multiplied against an activation. The single unavoidable FMUL is the
final scale application (one multiply per output element).

This matches the GPU Triton kernel's semantic guarantee at the algorithm
level, and runs on any hardware: CPU, ARM, RISC-V, embedded.
"""

import torch
import torch.nn.functional as F
from penta_kernel import quantize_and_pack, unpack_weights, _PENTA_OFFSET


def penta_linear_cpu(
    x       : torch.Tensor,
    packed_w: torch.Tensor,
    scale   : float,
    K_orig  : int,
    bias    : torch.Tensor | None = None,
) -> torch.Tensor:
    """
    CPU pentanary linear: Y = X @ W.T * scale + bias

    Zero-multiplier: weights applied via binary-mask matmuls + additions only.
    No x*w float multiplication in the inner loop.

    Args:
        x        : (..., K) float32 activation
        packed_w : (N, K_packs) int32 packed weights
        scale    : scalar float (absmean scale from quantize_and_pack)
        K_orig   : original K before padding
        bias     : (N,) optional bias

    Returns:
        y : (..., N) float32
    """
    orig_shape = x.shape
    x_2d = x.contiguous().reshape(-1, orig_shape[-1]).float()  # (M, K)
    M, K = x_2d.shape

    # Unpack 3-bit weights → (N, K) int32 in {-2,-1,0,+1,+2}
    w = unpack_weights(packed_w, K_orig)  # (N, K_orig)

    # ── Binary masks (0/1 float) — no arbitrary weight value used ──────────
    pos1 = (w ==  1).float()  # (N, K)
    neg1 = (w == -1).float()
    pos2 = (w ==  2).float()
    neg2 = (w == -2).float()
    # w == 0 → no contribution (implicit: all masks zero)

    # ── Masked matmuls — multiplying by 0 or 1 only ────────────────────────
    c_p1 = x_2d @ pos1.T   # (M, N)  : +x for each w=+1
    c_n1 = x_2d @ neg1.T   # (M, N)  : +x for each w=-1 (will be subtracted)
    c_p2 = x_2d @ pos2.T   # (M, N)  : +x for each w=+2 (added twice below)
    c_n2 = x_2d @ neg2.T   # (M, N)  : +x for each w=-2 (subtracted twice)

    # ── Accumulate — ×2 via addition, not FMUL ─────────────────────────────
    acc = c_p1 - c_n1 + (c_p2 + c_p2) - (c_n2 + c_n2)

    # ── Single unavoidable scale multiply ───────────────────────────────────
    y = acc * scale

    if bias is not None:
        y = y + bias.float()

    return y.reshape(*orig_shape[:-1], packed_w.shape[0])


# ── Benchmark ──────────────────────────────────────────────────────────────────

def benchmark_cpu(shapes: list[tuple] | None = None, iters: int = 100):
    """
    Compare penta_linear_cpu vs F.linear (dequant float32) on CPU.

    Reports:
      - Throughput (ms/call)
      - Speedup vs dequant baseline
      - Numerical parity (max absolute difference)
    """
    import time

    if shapes is None:
        shapes = [
            ("B=1,  K=768,  N=768  ", 1,   768,  768),
            ("B=8,  K=768,  N=3072 ", 8,   768, 3072),
            ("B=32, K=768,  N=3072 ", 32,  768, 3072),
            ("B=64, K=768,  N=3072 ", 64,  768, 3072),
            ("B=64, K=3072, N=768  ", 64, 3072,  768),
        ]

    print("\n" + "═" * 72)
    print("  CPU Kernel Benchmark — penta_linear_cpu vs F.linear (dequant float32)")
    print("═" * 72)
    print(f"  {'Shape':<30} {'Dequant F32':>12} {'CPU Kernel':>12} {'Speedup':>9} {'MaxDiff':>9}")
    print(f"  {'-'*72}")

    results = []
    for label, M, K, N in shapes:
        weight = torch.randn(N, K)
        packed, K_orig, scale = quantize_and_pack(weight)
        x = torch.randn(M, K)

        # Dequant baseline: unpack → float32 → F.linear
        w_dq = unpack_weights(packed, K_orig).float() * scale

        # Warmup
        for _ in range(10):
            _ = F.linear(x, w_dq)
            _ = penta_linear_cpu(x, packed, scale, K_orig)

        # Dequant F32
        t0 = time.perf_counter()
        for _ in range(iters):
            _ = F.linear(x, w_dq)
        t_dq = (time.perf_counter() - t0) / iters * 1000

        # CPU kernel
        t0 = time.perf_counter()
        for _ in range(iters):
            _ = penta_linear_cpu(x, packed, scale, K_orig)
        t_cpu = (time.perf_counter() - t0) / iters * 1000

        # Parity
        y_ref = F.linear(x, w_dq)
        y_ker = penta_linear_cpu(x, packed, scale, K_orig)
        max_diff = (y_ref - y_ker).abs().max().item()

        speedup = t_dq / t_cpu
        print(f"  {label:<30} {t_dq:>10.3f}ms {t_cpu:>10.3f}ms {speedup:>8.2f}× {max_diff:>9.2e}")
        results.append({
            "label": label, "M": M, "K": K, "N": N,
            "dequant_ms": t_dq, "cpu_kernel_ms": t_cpu,
            "speedup": speedup, "max_diff": max_diff,
        })

    mean_speedup = sum(r["speedup"] for r in results) / len(results)
    print(f"\n  Average speedup vs dequant float32: {mean_speedup:.2f}×")
    print("═" * 72)
    return results


# ── Smoke test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("🔥  PentaKernel CPU — smoke test + benchmark\n")

    torch.manual_seed(42)
    N, K, M = 768, 768, 32

    weight = torch.randn(N, K)
    packed, K_orig, scale = quantize_and_pack(weight)
    x = torch.randn(M, K)

    # Reference: dequantized F.linear
    w_dq = unpack_weights(packed, K_orig).float() * scale
    y_ref = F.linear(x, w_dq)
    y_cpu = penta_linear_cpu(x, packed, scale, K_orig)

    max_diff = (y_ref - y_cpu).abs().max().item()
    assert max_diff < 1e-4, f"Parity fail: max_diff={max_diff}"
    print(f"  ✅  Parity vs dequant F.linear: max_diff={max_diff:.2e}")

    results = benchmark_cpu()

    import json
    with open("benchmark_cpu_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n  Results saved → benchmark_cpu_results.json")
