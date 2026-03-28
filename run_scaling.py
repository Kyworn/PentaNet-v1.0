"""
run_scaling.py — PentaNet 300-500M scaling experiment.

Runs pentanet vs bitnet at xlarge scale (~345M params, GPT-2 medium)
across 3 seeds. This is the experiment that validates whether the
−6.4% PPL gap observed at 124M params holds at 345M params.

Usage:
    python run_scaling.py
    # or with a specific GPU:
    CUDA_VISIBLE_DEVICES=0 python run_scaling.py
"""
import os
import subprocess
import time
import json


def count_params():
    """Quick sanity check: print xlarge param count before launching."""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    try:
        from train_pentagpt import GPTConfig, PentaGPT
        cfg = GPTConfig(size="xlarge")
        model = PentaGPT(cfg)
        n = sum(p.numel() for p in model.parameters())
        print(f"  xlarge param count: {n/1e6:.1f}M")
        del model
    except Exception as e:
        print(f"  (param count check failed: {e})")


def run():
    print("=" * 65)
    print("  PentaNet Scaling Experiment — xlarge (~345M params)")
    print("  Validates whether −6.4% PPL gap holds at GPT-2 medium scale")
    print("=" * 65)

    count_params()

    python_bin = './.venv-gpu/bin/python'

    if not os.path.exists('data/wikitext-103/train.bin'):
        print("\n📥 Preparing WikiText-103 data...")
        subprocess.run([python_bin, 'prepare_data.py'], check=True)

    seeds = [42, 1337, 2026]
    modes = ['pentanet', 'bitnet']

    results = []
    start_time = time.time()

    for seed in seeds:
        for mode in modes:
            print(f"\n{'=' * 60}")
            print(f"  Launching: size=xlarge  mode={mode.upper()}  seed={seed}")
            print(f"{'=' * 60}\n")

            cmd = [
                python_bin, 'train_pentagpt.py',
                '--size',          'xlarge',
                '--mode',          mode,
                '--seed',          str(seed),
                '--max_iters',     '10000',
                '--eval_interval', '500',
                '--batch_size',    '4',   # reduced from 8 — xlarge is 2.8× larger
            ]

            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            try:
                result = subprocess.run(cmd, check=True, capture_output=False, env=env)
                results.append({"size": "xlarge", "mode": mode, "seed": seed, "status": "ok"})
            except subprocess.CalledProcessError as e:
                print(f"\n❌ Run failed (code {e.returncode}). Aborting.")
                results.append({"size": "xlarge", "mode": mode, "seed": seed, "status": "failed"})
                break
        else:
            continue
        break

    total_h = (time.time() - start_time) / 3600
    print(f"\n✅ Scaling runs done in {total_h:.2f}h")
    print("\nSummary:")
    for r in results:
        print(f"  {r['mode']:10s}  seed={r['seed']}  {r['status']}")

    with open("scaling_run_log.json", "w") as f:
        json.dump({"total_hours": total_h, "runs": results}, f, indent=2)
    print("\nLog → scaling_run_log.json")
    print("\nNext: collect PPL from each run's stdout/log and compare to large results.")


if __name__ == "__main__":
    run()
