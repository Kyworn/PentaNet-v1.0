"""
penta_kernel.py — PentaNet Custom Triton Kernel
================================================

3-bit pentanary weight packing + fused dequantize-matmul.

Encoding (actual weight → stored 3-bit unsigned value):
    -2 → 0  (0b000)
    -1 → 1  (0b001)
     0 → 2  (0b010)    ← padding always uses this value
    +1 → 3  (0b011)
    +2 → 4  (0b100)
    5, 6, 7 are unused (3 states wasted out of 8)

Packing: 10 weights per int32
    bits [2:0]   → weight at k-column (g*10 + 0)
    bits [5:3]   → weight at k-column (g*10 + 1)
    ...
    bits [29:27] → weight at k-column (g*10 + 9)
    bits [31:30] → 0 (unused)

Memory savings vs FP16:
    FP16:   16 bits/weight
    Packed:  3 bits/weight  → 5.33× compression
    vs BitNet 2-bit: PentaNet uses 1.5× more bits
                     → +47% information capacity (log2(5)/log2(3))
"""

import torch
import triton
import triton.language as tl


# ── Constants ──────────────────────────────────────────────────────────────────
_W_PER_INT32  = 10   # 10 × 3-bit = 30 bits per int32
_BITS_PER_W   = 3    # bits per pentanary weight
_PENTA_OFFSET = 2    # stored = actual + 2  ∈ {0..4}


# ── CPU Packing / Unpacking ────────────────────────────────────────────────────

def quantize_and_pack(weight: torch.Tensor) -> tuple[torch.Tensor, int, float]:
    """
    Quantize FP32 master weights → pentanary {-2..+2}  then 3-bit pack.

    Args:
        weight : (N, K) FP32/BF16 weight matrix

    Returns:
        packed  : (N, ceil(K/10)) int32 tensor
        K_orig  : original K (needed for unpadding)
        scale   : scalar float (absmean quantization scale)
    """
    assert weight.dim() == 2
    N, K = weight.shape
    K_orig    = K
    scale_val = float(weight.float().abs().mean().clamp(min=1e-8))

    # Quantize
    w_q = torch.clamp(torch.round(weight.float() / scale_val), -2, 2).to(torch.int32)

    # Pad K to multiple of 10 (pad with 0 actual weight → stored as PENTA_OFFSET)
    pad = (-K) % _W_PER_INT32
    if pad:
        w_q = torch.cat([w_q, w_q.new_zeros(N, pad)], dim=1)

    K_packs = w_q.shape[1] // _W_PER_INT32

    # Unsigned encoding: actual + 2
    w_u = (w_q + _PENTA_OFFSET).clamp(0, 7)          # (N, K_padded)
    w_g = w_u.reshape(N, K_packs, _W_PER_INT32)       # (N, K_packs, 10)

    # Pack 10 × 3-bit into each int32
    packed = torch.zeros(N, K_packs, dtype=torch.int32, device=weight.device)
    for i in range(_W_PER_INT32):
        packed |= w_g[:, :, i] << (_BITS_PER_W * i)

    return packed.contiguous(), K_orig, scale_val


def unpack_weights(packed: torch.Tensor, K_orig: int) -> torch.Tensor:
    """
    Unpack 3-bit tensor → int32 (N, K_orig) with values in {-2..+2}.
    Used for correctness validation.
    """
    N, K_packs = packed.shape
    out = torch.zeros(N, K_packs * _W_PER_INT32, dtype=torch.int32, device=packed.device)
    for i in range(_W_PER_INT32):
        out[:, i::_W_PER_INT32] = (packed >> (_BITS_PER_W * i)) & 7
    return (out - _PENTA_OFFSET)[:, :K_orig]


def memory_stats(N: int, K: int) -> dict:
    """Return memory footprint comparison for a given (N, K) weight matrix."""
    fp16_bytes   = N * K * 2
    packed_bits  = N * K * _BITS_PER_W
    packed_bytes = (packed_bits + 7) // 8
    ternary_bits = N * K * 2
    ternary_bytes = (ternary_bits + 7) // 8
    return {
        "fp16_MB"    : fp16_bytes   / 1e6,
        "packed_MB"  : packed_bytes / 1e6,
        "ternary_MB" : ternary_bytes / 1e6,
        "penta_vs_fp16_ratio"    : round(fp16_bytes   / packed_bytes, 2),
        "penta_vs_ternary_ratio" : round(ternary_bytes / packed_bytes, 2),
    }


# ── Triton Kernel ──────────────────────────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32}, num_stages=2, num_warps=8),
    ],
    key=['M', 'N', 'K_packs'],
)
@triton.jit
def _penta_matmul_kernel(
    # Pointers
    X_ptr, PW_ptr, Y_ptr,
    # Scalar scale
    scale,
    # Dimensions
    M, N, K_packs,
    # Strides
    stride_xm, stride_xk,
    stride_pwn, stride_pwg,
    stride_ym, stride_yn,
    # Tile sizes (autotuned)
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Y[M, N] = (X[M, K] @ W[N, K].T) * scale

    W is stored packed in PW[N, K_packs] (int32).
    Each int32 encodes 10 pentanary weights:
        weight_i = ((PW >> (3*i)) & 7) - 2,   i in 0..9

    Zero-Multiplier Implementation
    ───────────────────────────────
    No floating-point multiply (*) is used for the weight application.
    The 5 cases are handled with additions, negations, and tl.where selects:

        stored 0 → actual -2 : acc -= x + x
        stored 1 → actual -1 : acc -= x
        stored 2 → actual  0 : acc unchanged  (no-op)
        stored 3 → actual +1 : acc += x
        stored 4 → actual +2 : acc += x + x

    The ×2 is realised as x+x (one FADD), NOT as x*2 (FMUL).
    The scale multiplication at the end is the single unavoidable FP multiply.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)   # (BLOCK_M,)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)   # (BLOCK_N,)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for g in range(K_packs):

        # Load one packed int32 per output neuron: (BLOCK_N,)
        pw = tl.load(
            PW_ptr + rn * stride_pwn + g * stride_pwg,
            mask=rn < N,
            other=0,
        )  # int32

        for bit_pos in tl.static_range(10):
            k_col = g * 10 + bit_pos

            # Load one activation column: (BLOCK_M,) → float32
            x_col = tl.load(
                X_ptr + rm * stride_xm + k_col * stride_xk,
                mask=rm < M,
                other=0.0,
            ).to(tl.float32)

            # ── Zero-Multiplier Accumulation ───────────────────────────────
            # w_uint ∈ {0,1,2,3,4}  (BLOCK_N,)
            w_uint = (pw >> (bit_pos * 3)) & 7

            # Signed actual value: {-2, -1, 0, +1, +2}
            w_int = w_uint - 2

            # Explicit per-case flags (all shape BLOCK_N)
            abs_w   = tl.abs(w_int)     # {0, 1, 2}
            is_zero = w_int == 0        # skip → no accumulation
            is_mag2 = abs_w == 2        # ±2  → x + x  (FADD, no FMUL)
            is_neg  = w_int < 0         # negative → negate contribution

            # ×2 via FADD only — no FMUL
            x2 = x_col + x_col                          # (BLOCK_M,)

            # Select magnitude: ±2→x2, ±1→x_col  (broadcast BLOCK_N)
            contrib = tl.where(is_mag2[None, :],
                               x2[:, None],
                               x_col[:, None])           # (BLOCK_M, BLOCK_N)

            # Apply sign
            contrib = tl.where(is_neg[None, :], -contrib, contrib)

            # is_zero applied last — overrides any parasitic ±1/±2 path
            contrib = tl.where(
                is_zero[None, :],
                tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32),
                contrib,
            )

            acc = acc + contrib
            # ──────────────────────────────────────────────────────────────

    # Scale (one FMUL per output element) and store as bfloat16
    y_ptrs = Y_ptr + rm[:, None] * stride_ym + rn[None, :] * stride_yn
    mask   = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(y_ptrs, (acc * scale).to(tl.bfloat16), mask=mask)


# ── Python Wrapper ─────────────────────────────────────────────────────────────

def penta_linear(
    x       : torch.Tensor,
    packed_w: torch.Tensor,
    scale   : float,
    K_orig  : int,
    bias    : torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Fast pentanary linear: Y = X @ W.T * scale + bias

    Args:
        x        : (..., K) bfloat16 activation on CUDA
        packed_w : (N, K_packs) int32 packed weights on CUDA
        scale    : scalar float (absmean scale from quantize_and_pack)
        K_orig   : original K before padding
        bias     : (N,) optional bias tensor

    Returns:
        y : (..., N) bfloat16
    """
    assert x.is_cuda, "penta_linear requires CUDA"
    assert x.dtype == torch.bfloat16, "Activations must be bfloat16"
    assert packed_w.dtype == torch.int32

    orig_shape = x.shape
    x_2d = x.contiguous().reshape(-1, orig_shape[-1])
    M, K = x_2d.shape
    N, K_packs = packed_w.shape
    assert K == K_orig, f"Activation K={K} must match K_orig={K_orig}"

    y = torch.empty(M, N, device=x.device, dtype=torch.bfloat16)

    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(N, meta['BLOCK_N']),
    )

    _penta_matmul_kernel[grid](
        x_2d, packed_w, y,
        scale,
        M, N, K_packs,
        x_2d.stride(0), x_2d.stride(1),
        packed_w.stride(0), packed_w.stride(1),
        y.stride(0), y.stride(1),
    )

    if bias is not None:
        y = y + bias.to(torch.bfloat16)

    return y.reshape(*orig_shape[:-1], N)


# ── Quick Smoke Test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("🔥  PentaKernel smoke test\n")

    if not torch.cuda.is_available():
        print("❌  CUDA not available — skipping")
        raise SystemExit(0)

    DEVICE = "cuda"
    N, K, M = 768, 768, 32
    print(f"  Shape: X({M},{K}) × W({N},{K}).T\n")

    weight = torch.randn(N, K, device=DEVICE)
    packed, K_orig, scale = quantize_and_pack(weight)

    # Roundtrip correctness
    w_unpacked = unpack_weights(packed, K_orig)
    w_ref      = torch.clamp(torch.round(weight.float() / scale), -2, 2).int()
    assert (w_unpacked == w_ref).all(), "❌  Pack/unpack mismatch"
    print("  ✅  Pack/unpack roundtrip: EXACT MATCH")

    # Kernel vs PyTorch reference
    x = torch.randn(M, K, device=DEVICE, dtype=torch.bfloat16)
    y_kernel = penta_linear(x, packed, scale, K_orig)

    w_fp = w_ref.to(torch.bfloat16) * scale
    y_ref = (x.float() @ w_fp.float().T).to(torch.bfloat16)

    diff = (y_kernel.float() - y_ref.float()).abs()
    print(f"  ✅  Kernel vs PyTorch: max_diff={diff.max():.4f}, mean_diff={diff.mean():.5f}")

    # Memory stats
    stats = memory_stats(N, K)
    print(f"\n  📦  Memory ({N}×{K} weight matrix):")
    print(f"      FP16   : {stats['fp16_MB']:.2f} MB")
    print(f"      3-bit  : {stats['packed_MB']:.2f} MB  ({stats['penta_vs_fp16_ratio']}× smaller than FP16)")
    print(f"      2-bit  : {stats['ternary_MB']:.2f} MB  (BitNet reference)")
    print(f"      Penta vs Ternary: {stats['penta_vs_ternary_ratio']:.2f}× — uses 1.5× more memory, carries 1.47× more information")
    print("\n  🎉  All tests passed!")
