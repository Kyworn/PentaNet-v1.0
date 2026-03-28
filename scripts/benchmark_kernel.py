"""
scripts/benchmark_kernel.py — PentaKernel vs PyTorch Baseline
=============================================================

Validates:
  1. Exact pack/unpack roundtrip
  2. Numerical parity (kernel ≈ PyTorch STE reference)
  3. Throughput (ms/call) and speedup
  4. VRAM footprint comparison
"""

import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from penta_kernel import quantize_and_pack, unpack_weights, penta_linear, memory_stats
from pentanet_layer import PentaLinear

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WARMUP = 20
ITERS  = 200


# ── 1. Pack / Unpack Roundtrip ────────────────────────────────────────────────
def test_roundtrip():
    print("\n" + "═" * 60)
    print("  TEST 1 — Pack/Unpack Exact Roundtrip")
    print("═" * 60)

    cases = [(768, 768), (768, 3072), (3072, 768), (512, 512)]
    all_ok = True
    for N, K in cases:
        # Random pentanary ground truth
        w_true = torch.randint(-2, 3, (N, K), dtype=torch.int32, device=DEVICE)
        packed, K_orig, _ = quantize_and_pack(w_true.float())
        w_back = unpack_weights(packed, K_orig)
        ok = (w_true == w_back).all().item()
        print(f"  {'✅' if ok else '❌'}  ({N}×{K})  {'PASS' if ok else 'FAIL'}")
        all_ok = all_ok and ok
    return all_ok


# ── 2. Numerical Parity ───────────────────────────────────────────────────────
def test_parity():
    print("\n" + "═" * 60)
    print("  TEST 2 — Numerical Parity (kernel vs PyTorch STE)")
    print("═" * 60)

    cases = [
        ("Attn-QKV  ", 16,  768, 2304),
        ("Attn-Proj ", 16,  768,  768),
        ("FFN-Up    ", 16,  768, 3072),
        ("FFN-Down  ", 16, 3072,  768),
    ]
    all_ok = True
    for label, M, K, N in cases:
        layer = PentaLinear(K, N, bias=False).to(DEVICE)
        x_fp32 = torch.randn(M, K, device=DEVICE)
        x_bf16 = x_fp32.to(torch.bfloat16)

        # Reference: STE forward (PyTorch)
        with torch.no_grad():
            y_ref = layer(x_fp32)

        # Kernel: 3-bit packed
        packed, K_orig, scale = quantize_and_pack(layer.weight.detach())
        y_ker = penta_linear(x_bf16, packed, scale, K_orig)

        # Compare
        diff = (y_ref.float() - y_ker.float()).abs()
        max_d = diff.max().item()
        rel_d = (diff / (y_ref.float().abs().clamp(min=1e-6))).mean().item()
        ok = max_d < 0.2   # BF16 quantization tolerance
        print(f"  {'✅' if ok else '❌'}  {label} ({M},{K}→{N})  "
              f"max={max_d:.4f}  rel={rel_d:.4f}")
        all_ok = all_ok and ok
    return all_ok


# ── 3. Throughput Benchmark ───────────────────────────────────────────────────
def benchmark_throughput():
    print("\n" + "═" * 60)
    print("  TEST 3 — Throughput Benchmark")
    print("═" * 60)
    print(f"  {'Shape':<30} {'PyTorch':>10} {'Kernel':>10} {'Speedup':>9}")
    print(f"  {'-'*62}")

    cases = [
        ("B=1,  K=768,  N=768  ", 1,   768,  768),
        ("B=8,  K=768,  N=3072 ", 8,   768, 3072),
        ("B=32, K=768,  N=3072 ", 32,  768, 3072),
        ("B=64, K=768,  N=3072 ", 64,  768, 3072),
        ("B=64, K=3072, N=768  ", 64, 3072,  768),
    ]

    results = []
    for label, M, K, N in cases:
        layer    = PentaLinear(K, N, bias=False).to(DEVICE)
        x_fp32   = torch.randn(M, K, device=DEVICE)
        x_bf16   = x_fp32.to(torch.bfloat16)
        packed, K_orig, scale = quantize_and_pack(layer.weight.detach())

        # Warmup
        for _ in range(WARMUP):
            with torch.no_grad():
                _ = layer(x_fp32)
            _ = penta_linear(x_bf16, packed, scale, K_orig)
        torch.cuda.synchronize()

        # PyTorch STE
        t0 = time.perf_counter()
        for _ in range(ITERS):
            with torch.no_grad():
                _ = layer(x_fp32)
        torch.cuda.synchronize()
        t_pt = (time.perf_counter() - t0) / ITERS * 1000   # ms

        # Kernel
        t0 = time.perf_counter()
        for _ in range(ITERS):
            _ = penta_linear(x_bf16, packed, scale, K_orig)
        torch.cuda.synchronize()
        t_ker = (time.perf_counter() - t0) / ITERS * 1000  # ms

        speedup = t_pt / t_ker
        print(f"  {label:<30} {t_pt:>8.3f}ms {t_ker:>8.3f}ms {speedup:>8.2f}×")
        results.append({"label": label, "pytorch_ms": t_pt, "kernel_ms": t_ker, "speedup": speedup})

    return results


# ── 4. VRAM Footprint ─────────────────────────────────────────────────────────
def print_memory():
    print("\n" + "═" * 60)
    print("  TEST 4 — VRAM Footprint Comparison")
    print("═" * 60)
    print(f"  {'Layer (N×K)':<22} {'FP16':>8} {'BitNet':>8} {'PentaNet':>10} {'vs FP16':>9} {'vs BitNet':>10}")
    print(f"  {'-'*70}")

    layers = [
        ("Q/K/V (768×768)", 768, 768),
        ("FFN-Up (768×3072)", 768, 3072),
        ("FFN-Down (3072×768)", 3072, 768),
        ("Total 12L GPT-2", 12 * (3 * 768 * 768 + 768 * 768 + 768 * 3072 + 3072 * 768), 1),
    ]

    for label, N, K in layers:
        if K == 1:  # total hack
            total_params = N
            fp16_mb   = total_params * 2 / 1e6
            packed_mb = total_params * 3 / 8 / 1e6
            tern_mb   = total_params * 2 / 8 / 1e6
            print(f"  {label:<22} {fp16_mb:>7.1f}M {tern_mb:>7.1f}M {packed_mb:>9.1f}M "
                  f"{fp16_mb/packed_mb:>8.1f}× {tern_mb/packed_mb:>9.1f}×")
        else:
            s = memory_stats(N, K)
            print(f"  {label:<22} {s['fp16_MB']:>7.2f}M {s['ternary_MB']:>7.2f}M {s['packed_MB']:>9.2f}M "
                  f"{s['penta_vs_fp16_ratio']:>8.1f}× {s['penta_vs_ternary_ratio']:>9.1f}×")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print(f"\n{'#'*60}")
    print(f"  🔬  PentaKernel Benchmark Suite")
    print(f"  GPU : {torch.cuda.get_device_name(0) if DEVICE=='cuda' else 'CPU'}")
    print(f"{'#'*60}")

    if DEVICE == "cpu":
        print("\n⚠️  CUDA not available — cannot run Triton kernel on CPU.")
        return

    ok1 = test_roundtrip()
    ok2 = test_parity()
    results = benchmark_throughput()
    print_memory()

    print(f"\n{'═'*60}")
    print(f"  SUMMARY")
    print(f"{'═'*60}")
    print(f"  Roundtrip:  {'✅ PASS' if ok1 else '❌ FAIL'}")
    print(f"  Parity:     {'✅ PASS' if ok2 else '❌ FAIL'}")
    mean_speedup = sum(r['speedup'] for r in results) / len(results)
    print(f"  Speedup:    {mean_speedup:.2f}× average over all shapes")

    with open("benchmark_kernel_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved → benchmark_kernel_results.json\n")


if __name__ == "__main__":
    main()
