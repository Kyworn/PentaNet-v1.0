"""
penta_kernel_cpu.py — PentaNet CPU Inference Kernel (Numba JIT)
================================================================

Zero-multiplier CPU kernel for pentanary weights {-2,-1,0,+1,+2}.

Explicit 5-case inner loop: for each (n, m, k), branches on w and
accumulates using only add/subtract. The ×2 for |w|=2 is x+x (FADD),
not x*2 (FMUL). Compiled by LLVM via Numba — no fmul in inner loop.

Parallelized over N (output neurons) — for N=768 or N=3072 this gives
full multi-core utilization regardless of batch size.

Related to DeepShift (Elhoushi et al., 2019): same shift+sign principle
applied to LLM-scale matrices with native pentanary training.
"""

import torch
import torch.nn.functional as F
import numpy as np
from penta_kernel import quantize_and_pack, unpack_weights

try:
    from numba import njit, prange
    _NUMBA_AVAILABLE = True
except ImportError:
    _NUMBA_AVAILABLE = False


# ══════════════════════════════════════════════════════════════════════════════
# Numba JIT kernel — parallelized over N (output neurons)
# ══════════════════════════════════════════════════════════════════════════════

if _NUMBA_AVAILABLE:
    @njit(parallel=True, fastmath=True, cache=True)
    def _penta_matmul_numba(x: np.ndarray, w: np.ndarray, out: np.ndarray):
        """
        Pentanary matmul: out[m, n] = sum_k(x[m,k] * w[n,k]) for w ∈ {-2,-1,0,+1,+2}.

        Parallelized over N (output neurons) — gives full multi-core utilization
        even at batch size 1. Inner loop uses add/subtract only, no FMUL.
        """
        N, K = w.shape
        M = x.shape[0]
        for n in prange(N):                    # parallel over neurons
            for m in range(M):
                acc = np.float32(0.0)
                for k in range(K):
                    wk = w[n, k]
                    xk = x[m, k]
                    if wk == 1:
                        acc += xk
                    elif wk == -1:
                        acc -= xk
                    elif wk == 2:
                        acc += xk + xk         # ×2 via FADD, not FMUL
                    elif wk == -2:
                        acc -= xk + xk         # ×2 via FADD, not FMUL
                    # wk == 0: skip
                out[m, n] = acc


def penta_linear_cpu(
    x       : torch.Tensor,
    packed_w: torch.Tensor,
    scale   : float,
    K_orig  : int,
    bias    : torch.Tensor | None = None,
) -> torch.Tensor:
    """
    CPU pentanary linear: Y = X @ W.T * scale + bias.
    Zero-multiplier: explicit 5-case add/subtract loop, no x*w FMUL.
    Compiled to native SIMD via Numba LLVM backend.

    Args:
        x        : (..., K) float32
        packed_w : (N, K_packs) int32
        scale    : absmean scale
        K_orig   : original K
        bias     : (N,) optional

    Returns:
        y : (..., N) float32
    """
    assert _NUMBA_AVAILABLE, "numba required — pip install numba"

    orig_shape = x.shape
    x_np = x.contiguous().reshape(-1, orig_shape[-1]).float().numpy().astype(np.float32)
    w_np = unpack_weights(packed_w, K_orig).numpy().astype(np.int8)

    M = x_np.shape[0]
    N = w_np.shape[0]
    out = np.empty((M, N), dtype=np.float32)

    _penta_matmul_numba(x_np, w_np, out)

    y = torch.from_numpy(out) * scale
    if bias is not None:
        y = y + bias.float()

    return y.reshape(*orig_shape[:-1], N)


# ══════════════════════════════════════════════════════════════════════════════
# Benchmark
# ══════════════════════════════════════════════════════════════════════════════

def benchmark_cpu(shapes: list[tuple] | None = None, iters: int = 100):
    """
    Compare penta_linear_cpu vs F.linear (dequant float32) on CPU.
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

    print("\n" + "═" * 68)
    print("  CPU Kernel Benchmark — penta_linear_cpu vs F.linear (dequant F32)")
    print("═" * 68)
    print(f"  {'Shape':<30} {'Dequant F32':>12} {'CPU Kernel':>12} {'Speedup':>9}")
    print(f"  {'-'*66}")

    results = []
    for label, M, K, N in shapes:
        weight = torch.randn(N, K)
        packed, K_orig, scale = quantize_and_pack(weight)
        x = torch.randn(M, K)
        w_dq = unpack_weights(packed, K_orig).float() * scale

        # Warmup + JIT trigger
        for _ in range(5):
            _ = F.linear(x, w_dq)
        for _ in range(3):
            _ = penta_linear_cpu(x, packed, scale, K_orig)

        # Dequant baseline
        t0 = time.perf_counter()
        for _ in range(iters):
            _ = F.linear(x, w_dq)
        t_dq = (time.perf_counter() - t0) / iters * 1000

        # Pentanary CPU kernel
        t0 = time.perf_counter()
        for _ in range(iters):
            _ = penta_linear_cpu(x, packed, scale, K_orig)
        t_cpu = (time.perf_counter() - t0) / iters * 1000

        # Parity
        y_ref = F.linear(x, w_dq)
        y_k   = penta_linear_cpu(x, packed, scale, K_orig)
        diff  = (y_ref - y_k).abs().max().item()
        speedup = t_dq / t_cpu

        print(f"  {label:<30} {t_dq:>10.3f}ms {t_cpu:>10.3f}ms {speedup:>8.2f}×  diff={diff:.1e}")
        results.append({
            "label": label, "M": M, "K": K, "N": N,
            "dequant_ms": t_dq, "cpu_kernel_ms": t_cpu,
            "speedup": speedup, "max_diff": diff,
        })

    mean_speedup = sum(r["speedup"] for r in results) / len(results)
    print(f"\n  Average speedup: {mean_speedup:.2f}×")
    print("═" * 68)
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Smoke test + entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("🔥  PentaKernel CPU — smoke test + benchmark\n")
    torch.manual_seed(42)

    N, K, M = 768, 768, 32
    weight = torch.randn(N, K)
    packed, K_orig, scale = quantize_and_pack(weight)
    x = torch.randn(M, K)
    w_dq = unpack_weights(packed, K_orig).float() * scale
    y_ref = F.linear(x, w_dq)

    print("  ⏳  Compiling Numba kernel (first run)...")
    y_cpu = penta_linear_cpu(x, packed, scale, K_orig)
    diff = (y_ref - y_cpu).abs().max().item()
    assert diff < 1e-4, f"Parity fail: {diff}"
    print(f"  ✅  Parity vs dequant F.linear: max_diff={diff:.2e}\n")

    results = benchmark_cpu()

    import json
    with open("benchmark_cpu_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n  Results saved → benchmark_cpu_results.json")
