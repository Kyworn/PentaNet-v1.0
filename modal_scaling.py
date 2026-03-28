"""
modal_scaling.py — PentaNet scaling experiment on Modal A100s.

Runs pentanet + bitnet in parallel (seed=42, 50k iters, xlarge ~354M params).

Deploy:
    modal run modal_scaling.py
"""

import modal

app = modal.App("pentanet-scaling")

# ── Image with code baked in ─────────────────────────────────────────────────

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1",
        "numpy",
        "tiktoken",
        "transformers",
    )
    .add_local_file("/home/zorko/AI_Search/train_pentagpt.py", "/code/train_pentagpt.py")
    .add_local_file("/home/zorko/AI_Search/penta_kernel.py",   "/code/penta_kernel.py")
)

# ── Volumes ──────────────────────────────────────────────────────────────────

results_vol = modal.Volume.from_name("pentanet-results", create_if_missing=True)
data_vol    = modal.Volume.from_name("pentanet-data",    create_if_missing=True)


# ── Upload data (run once) ───────────────────────────────────────────────────

@app.function(
    image=modal.Image.debian_slim(),
    volumes={"/data": data_vol},
    timeout=600,
)
def upload_data(train_bytes: bytes, val_bytes: bytes):
    import os
    dst = "/data/wikitext-103"
    os.makedirs(dst, exist_ok=True)
    with open(f"{dst}/train.bin", "wb") as f:
        f.write(train_bytes)
    with open(f"{dst}/validation.bin", "wb") as f:
        f.write(val_bytes)
    data_vol.commit()
    print(f"Uploaded train.bin ({len(train_bytes)//1024**2}MB) and validation.bin ({len(val_bytes)//1024**2}MB)")


# ── Training run ─────────────────────────────────────────────────────────────

@app.function(
    image=image,
    gpu="A100",
    timeout=60 * 60 * 6,
    volumes={
        "/results": results_vol,
        "/data":    data_vol,
    },
)
def train_run(mode: str, seed: int, max_iters: int = 50000, lr: float = 3e-4):
    import subprocess, sys, os
    os.makedirs("/results", exist_ok=True)
    os.chdir("/code")

    result = subprocess.run(
        [
            sys.executable, "/code/train_pentagpt.py",
            "--size",          "xlarge",
            "--mode",          mode,
            "--seed",          str(seed),
            "--max_iters",     str(max_iters),
            "--eval_interval", "500",
            "--batch_size",    "8",
            "--lr",            str(lr),
            "--data_dir",      "/data/wikitext-103",
            "--no_wandb",
        ],
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    results_vol.commit()
    return {"mode": mode, "seed": seed, "returncode": result.returncode}


# ── Orchestrator ─────────────────────────────────────────────────────────────

@app.local_entrypoint()
def main():
    import json

    # Upload data first
    print("Uploading WikiText-103 to Modal Volume...")
    with open("/home/zorko/AI_Search/data/wikitext-103/train.bin", "rb") as f:
        train_bytes = f.read()
    with open("/home/zorko/AI_Search/data/wikitext-103/validation.bin", "rb") as f:
        val_bytes = f.read()
    upload_data.remote(train_bytes, val_bytes)
    print("Data uploaded.")

    # LR ablation: pentanet with lower LR
    print("\nLaunching PentaNet LR ablation on A100 (seed=42, 5k iters)...")
    jobs = [("pentanet", 42), ("pentanet", 1337)]
    results = list(train_run.starmap(jobs, kwargs={"max_iters": 5000, "lr": 1e-4}))

    print("\n── Results ──")
    for r in results:
        print(r)

    with open("modal_scaling_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved → modal_scaling_results.json")
