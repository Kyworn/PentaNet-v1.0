"""
penta_avx2_wrapper.py — Python wrapper for the AVX2 zero-multiplier kernel.

Loads penta_avx2.so via ctypes and exposes:
  - penta_unpack()       : 3-bit packed int32 → int8
  - penta_linear_avx2()  : zero-multiplier matmul (no _mm256_mul_ps in inner loop)
"""

import ctypes
import os
import numpy as np
import torch

# ── Load shared library ──────────────────────────────────────────────────────
_so_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "penta_avx2.so")
_lib = ctypes.CDLL(_so_path)

# void penta_unpack_3bit(int32*, int8*, int N, int K_packs, int K_orig)
_lib.penta_unpack_3bit.restype  = None
_lib.penta_unpack_3bit.argtypes = [
    ctypes.POINTER(ctypes.c_int32),
    ctypes.POINTER(ctypes.c_int8),
    ctypes.c_int, ctypes.c_int, ctypes.c_int,
]

# void penta_matmul_avx2(float* x, int8* w, float* out, int M,N,K, float scale)
_lib.penta_matmul_avx2.restype  = None
_lib.penta_matmul_avx2.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_int8),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_float,
]


def penta_unpack(packed: torch.Tensor, K_orig: int) -> np.ndarray:
    """
    Unpack (N, K_packs) int32 → (N, K_orig) int8  via C kernel.
    Returns a numpy int8 array ready for penta_matmul_avx2.
    """
    packed_np = packed.numpy().astype(np.int32)
    N, K_packs = packed_np.shape
    w_int8 = np.empty((N, K_orig), dtype=np.int8)

    _lib.penta_unpack_3bit(
        packed_np.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        w_int8.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
        ctypes.c_int(N),
        ctypes.c_int(K_packs),
        ctypes.c_int(K_orig),
    )
    return w_int8


def penta_linear_avx2(
    x      : torch.Tensor,
    w_int8 : np.ndarray,
    scale  : float,
    bias   : torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Zero-multiplier AVX2 pentanary linear layer.

    Args:
        x      : (..., K) float32 on CPU
        w_int8 : (N, K) int8 numpy array (from penta_unpack)
        scale  : absmean scale factor
        bias   : (N,) optional

    Returns:
        y : (..., N) float32
    """
    orig_shape = x.shape
    x_np = x.contiguous().reshape(-1, orig_shape[-1]).float().numpy()
    M, K = x_np.shape
    N    = w_int8.shape[0]

    out_np = np.empty((M, N), dtype=np.float32)

    _lib.penta_matmul_avx2(
        x_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        w_int8.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
        out_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int(M),
        ctypes.c_int(N),
        ctypes.c_int(K),
        ctypes.c_float(scale),
    )

    y = torch.from_numpy(out_np)
    if bias is not None:
        y = y + bias.float()
    return y.reshape(*orig_shape[:-1], N)


# ── Benchmark ────────────────────────────────────────────────────────────────

def benchmark(iters: int = 300):
    import time
    import torch.nn.functional as F
    from penta_kernel import quantize_and_pack, unpack_weights
    from penta_kernel_cpu import penta_linear_cpu

    shapes = [
        ("B=1,  K=768,  N=768  ", 1,   768,  768),
        ("B=8,  K=768,  N=3072 ", 8,   768, 3072),
        ("B=32, K=768,  N=3072 ", 32,  768, 3072),
        ("B=64, K=768,  N=3072 ", 64,  768, 3072),
        ("B=64, K=3072, N=768  ", 64, 3072,  768),
    ]

    print("\n" + "═" * 85)
    print("  CPU Kernel Comparison (AMD Ryzen 9800X3D)")
    print("═" * 85)
    print(f"  {'Shape':<30} {'Dequant F32':>12} {'INT8 _mm':>10} {'Numba':>10} {'AVX2':>10} {'AVX2 vs F32':>12}")
    print(f"  {'-'*83}")

    torch.manual_seed(42)
    results = []

    for label, M, K, N in shapes:
        weight = torch.randn(N, K)
        packed, K_orig, scale = quantize_and_pack(weight)
        x = torch.randn(M, K)

        # Prepare baselines
        w_dq    = unpack_weights(packed, K_orig).float() * scale
        w_int8n = unpack_weights(packed, K_orig).to(torch.int8)
        x_scale = x.float().abs().max() / 127.0
        x_int8  = (x / x_scale).round().clamp(-128, 127).to(torch.int8)
        w_avx2  = penta_unpack(packed, K_orig)   # int8 numpy array

        # Warmup
        for _ in range(20):
            _ = F.linear(x, w_dq)
            _ = torch._int_mm(x_int8, w_int8n.T)
            _ = penta_linear_cpu(x, packed, scale, K_orig)
            _ = penta_linear_avx2(x, w_avx2, scale)

        def time_fn(fn, n=iters):
            t0 = time.perf_counter()
            for _ in range(n): fn()
            return (time.perf_counter() - t0) / n * 1000

        t_dq    = time_fn(lambda: F.linear(x, w_dq))
        t_int8  = time_fn(lambda: torch._int_mm(x_int8, w_int8n.T).float() * (scale * x_scale))
        t_numba = time_fn(lambda: penta_linear_cpu(x, packed, scale, K_orig))
        t_avx2  = time_fn(lambda: penta_linear_avx2(x, w_avx2, scale))

        # Parity check
        y_ref  = F.linear(x, w_dq)
        y_avx2 = penta_linear_avx2(x, w_avx2, scale)
        diff   = (y_ref - y_avx2).abs().max().item()

        sp_avx2 = t_dq / t_avx2
        print(f"  {label:<30} {t_dq:>10.3f}ms {t_int8:>8.3f}ms {t_numba:>8.3f}ms "
              f"{t_avx2:>8.3f}ms {sp_avx2:>10.2f}×  diff={diff:.1e}")
        results.append({
            "label": label, "M": M, "K": K, "N": N,
            "dequant_ms": t_dq, "int8_ms": t_int8,
            "numba_ms": t_numba, "avx2_ms": t_avx2,
            "speedup_vs_dequant": sp_avx2, "max_diff": diff,
        })

    print("═" * 85)
    return results


if __name__ == "__main__":
    import torch
    import torch.nn.functional as F
    from penta_kernel import quantize_and_pack, unpack_weights

    print("🔥  PentaNet AVX2 kernel — smoke test\n")
    torch.manual_seed(42)

    N, K, M = 768, 768, 32
    weight = torch.randn(N, K)
    packed, K_orig, scale = quantize_and_pack(weight)
    x = torch.randn(M, K)

    w_dq   = unpack_weights(packed, K_orig).float() * scale
    y_ref  = F.linear(x, w_dq)
    w_avx2 = penta_unpack(packed, K_orig)
    y_avx2 = penta_linear_avx2(x, w_avx2, scale)

    diff = (y_ref - y_avx2).abs().max().item()
    assert diff < 1e-3, f"Parity fail: max_diff={diff}"
    print(f"  ✅  Parity vs dequant F.linear: max_diff={diff:.2e}\n")

    results = benchmark()

    import json
    with open("benchmark_avx2_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n  Results saved → benchmark_avx2_results.json")
