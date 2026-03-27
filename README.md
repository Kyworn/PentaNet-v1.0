# PentaNet — Native Pentanary Quantization for LLMs

**Author:** Zorko · Independent Researcher · [zorko.xyz](https://zorko.xyz)

> PentaNet extends extreme quantization beyond BitNet's ternary `{-1, 0, +1}` to pentanary `{-2, -1, 0, +1, +2}`, achieving a **6.4% perplexity improvement** on WikiText-103 while preserving zero-multiplier inference (additions + bit-shifts only).

## Key Results

| Model | Mean PPL | Std | Seeds |
|:---|:---:|:---:|:---:|
| **PentaNet** {-2..+2} | **180.32** | ±2.09 | 42, 1337, 2026 |
| BitNet {-1..+1} | 192.63 | ±3.52 | 42, 1337, 2026 |

- **124M parameter** GPT-2-style transformer
- **WikiText-103** (~100M tokens)
- Trained on a single **RTX 5080** (16 GB)
- No collapse: ±2 buckets maintain ~11% occupancy through all 10k iterations

## Project Structure

```
├── README.md
├── PentaNet_NeurIPS_Draft.md       # Full technical report (markdown)
├── train_pentagpt.py               # Core training script (PentaNet + BitNet)
├── pentanet_layer.py               # PentaLinear layer implementation
├── prepare_data.py                 # WikiText-103 data preparation
├── run_benchmark.py                # 3-seed benchmark orchestrator
├── paper/
│   ├── PentaNet_Technical_Report.pdf
│   └── figures/
├── scripts/                        # Visualization & utilities
│   ├── compile_pdf.py
│   ├── export_figures.py
│   ├── generate_dashboard.py
│   └── plot_results.py
└── models/                         # JSON logs + model checkpoints
    ├── pentanet_large_s{42,1337,2026}_results.json
    └── bitnet_large_s{42,1337,2026}_results.json
```

## Quick Start

```bash
# 1. Setup
python -m venv .venv-gpu && source .venv-gpu/bin/activate
pip install torch transformers datasets

# 2. Prepare data
python prepare_data.py

# 3. Run full benchmark (3 seeds × 2 architectures, ~2h15 on RTX 5080)
python run_benchmark.py

# 4. Visualize results
python scripts/generate_dashboard.py   # Interactive HTML dashboard
python scripts/export_figures.py       # Publication-quality PNG/PDF
python scripts/compile_pdf.py          # Compile full paper PDF
```

## Model Weights (HuggingFace)

Pre-trained checkpoints are available on HuggingFace:
> 🤗 [kyworn/pentanet-124m](https://huggingface.co/kyworn/pentanet-124m) 

## Citation

```bibtex
@techreport{zorko2026pentanet,
  title     = {PentaNet: Native Pentanary Quantization for Large Language Models},
  author    = {Zorko},
  year      = {2026},
  url       = {https://zorko.xyz}
}
```

## License

MIT
