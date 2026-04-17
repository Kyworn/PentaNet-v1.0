---
language: en
license: mit
tags:
  - quantization
  - pentanary
  - bitnet
  - language-model
  - native-quantization
  - extreme-quantization
datasets:
  - wikitext
metrics:
  - perplexity
model-index:
  - name: PentaNet-124M
    results:
      - task:
          type: text-generation
          name: Language Modeling
        dataset:
          type: wikitext
          name: WikiText-103
        metrics:
          - type: perplexity
            value: 180.32
            name: Validation Perplexity (mean, 3 seeds)
---

# PentaNet — Native Pentanary Quantization for LLMs

**Author:** Zorko · [zorko.xyz](https://zorko.xyz)

> PentaNet extends extreme quantization beyond BitNet's ternary `{-1, 0, +1}` to pentanary `{-2, -1, 0, +1, +2}`, achieving a **6.4% perplexity improvement** at 124M params on WikiText-103 while preserving zero-multiplier arithmetic. Scaling experiments show this advantage does not transfer to larger models (345M+) — the pentanary space requires more sophisticated scaling than absmean + STE.

## Key Results

### 124M params (12 layers × 768 embed)

| Model | Mean PPL | Std | Seeds |
|:---|:---:|:---:|:---:|
| **PentaNet** {-2..+2} | **180.32** | ±2.09 | 42, 1337, 2026 |
| BitNet {-1..+1} | 192.63 | ±3.52 | 42, 1337, 2026 |

- **124M parameter** GPT-2-style transformer
- **WikiText-103** (~100M tokens)
- Trained on a single **RTX 5080** (16 GB)
- No collapse: ±2 buckets maintain ~11% occupancy through all 10k iterations

### 345M params (24 layers × 1024 embed)

| Model | PPL | Note |
|:---|:---:|:---|
| **BitNet** {-1..+1} | **273.0** | 67% outer state usage |
| PentaNet {-2..+2} | 320.1 | 22% outer state usage |
| PentaNet sf=0.8 | 618 | 34% outer state usage (short_wide 12×1536) |

See [INVESTIGATION.md](INVESTIGATION.md) for the full scaling analysis and scale_factor ablation.

### Text Generation Example (124M params, 20min training)
*(Prompt: "The history of the internet began with")*

```text
⏳ Generating with BitNet (Ternary {-1, 0, 1}) ...
🤖 BITNET S42: The history of the internet began with the <unk> to be a way , <unk> , which was the first recent of the <unk> , and the city and the <unk> . The French army was the first to be the first @-@ scale

⏳ Generating with PentaNet (Pentanary {-2, -1, 0, 1, 2}) ...
🤖 PENTANET S42: The history of the internet began with the original level of the other . The term of the original world was to the public court of the United States in July 2013 in February 15 , 2015 , as well as the team of $ 2 @,@ 000 . In the same year , the
```
*Notice how BitNet struggles with vocabulary collapse (`<unk>`) and repetitive stuttering, while PentaNet generates fluent, grammatically correct Wikipedia-style coherent sentences (despite being factually hallucinatory due to the small size).*
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
│   └── pentanet_analysis.py
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

## V2 Resurrection Paths

The scaling investigation shows absmean + STE is the bottleneck. Three concrete directions for PentaNet v2:

1. **Learnable Scale per Layer** — Replace the fixed absmean with a per-layer parameter optimized jointly with weights, letting each layer find its own optimal quantization grid entropy.
2. **Distillation from FP32** — Stop training from scratch. Use a pre-trained FP32 model as teacher so pentanary weights inherit meaningful structure instead of discovering it from random init.
3. **Soft Quantization** — Replace the hard Round+Clip STE with continuous relaxations (Gumbel-Softmax or temperature-scaled) so weights slide smoothly toward ±2 instead of being thrown there abruptly.

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
