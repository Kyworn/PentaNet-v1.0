import os
import subprocess
import time

def run():
    print("🚀 Starting benchmark suite: PentaNet vs BitNet on WikiText-103")
    
    python_bin = './.venv-gpu/bin/python'
    
    # 1. Prepare data
    if not os.path.exists('data/wikitext-103/train.bin'):
        print("📥 Preparing WikiText-103 data...")
        subprocess.run([python_bin, 'prepare_data.py'], check=True)
        
    seeds = [42, 1337, 2026]
    modes = ['pentanet', 'bitnet']
    
    start_time = time.time()
    
    for seed in seeds:
        for mode in modes:
            print(f"\n{'='*60}")
            print(f"🌟 Launching RUN: Mode={mode.upper()}, Seed={seed}")
            print(f"{'='*60}\n")
            
            cmd = [
                python_bin, 'train_pentagpt.py',
                '--size', 'large',
                '--mode', mode,
                '--seed', str(seed),
                '--max_iters', '10000',  # Full 10k iterations as requested
                '--eval_interval', '500', 
                '--batch_size', '8'
            ]
            
            # Using Popen so it streams output nicely, or subprocess.run
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"❌ Run failed with code {e.returncode}. Aborting suite.")
                return
                
    total_time = (time.time() - start_time) / 3600
    print(f"\n✅ All benchmark runs completed successfully in {total_time:.2f} hours.")

if __name__ == "__main__":
    run()
