# PentaNet Scaling Investigation

## Problem
At 345M params (24 layers, xlarge), PentaNet underperforms BitNet.

---

## Experiment 1: Baseline xlarge (10k iters)

**Config:** 24 layers x 1024 embed, seed=42, batch_size=4

| Model | Final PPL | Outer State Usage |
|-------|----------|------------------|
| **BitNet** {-1,0,+1} | **273.0** | 67.4% (±1) |
| PentaNet {-2..+2} | 320.1 | 22.2% (±2) |

**Finding:** PentaNet barely uses its outer states. The absmean scale is too large — weights never reach ±2.

Per-layer analysis: all layers lose ±2 uniformly (no specific layer collapsing). Scale factors stable (~0.02).

---

## Experiment 2: Short & Wide (2k iters, sf=1.0)

**Config:** 12 layers x 1536 embed (417M params), seed=42, batch_size=4, scale_factor=1.0

| Model | PPL @ 2k | ±2 Occupancy |
|-------|---------|-------------|
| PentaNet short_wide | 1249 | ~23% |

**Finding:** Same problem as xlarge — not a depth issue, it's the scale.

---

## Experiment 3: Scale Factor Ablation (2k iters)

**Config:** short_wide, seed=42, batch_size=4

| scale_factor | PPL @ 2k | ±2 Occupancy | Distribution |
|-------------|---------|-------------|-------------|
| 1.0 (default) | 1249 | 23% | Gaussian, timid |
| **0.8** | **1166** | **34%** | **Balanced** |
| 0.6 | 1200 | 47% | U-shape, saturated |

**Finding:** sf=0.8 is the sweet spot. ±2 at 34% = flat distribution allowing fine-tuning.

---

## Experiment 4: Scale Factor 0.8 — Full 10k (FINAL)

**Config:** short_wide, seed=42, batch_size=4, sf=0.8, 10000 iters

| Iter | PPL | ±2% |
|------|-----|-----|
| 0 | 76891 | 33.8% |
| 1000 | 1665 | 33.4% |
| 2000 | 1234 | 32.3% |
| 3000 | 995 | 32.5% |
| 6000 | 751 | 31.3% |
| 9000 | 625 | 30.0% |
| **10000** | **618** | **31.1%** |

**Final:** PPL 618, ±2 stable at ~31%.

### Summary Comparison

| Config | Model | Params | PPL (10k) |
|--------|-------|--------|----------|
| xlarge 24x1024 | BitNet | 354M | **273** |
| xlarge 24x1024 | PentaNet sf=1.0 | 354M | 320 |
| short_wide 12x1536 | PentaNet sf=0.8 | 418M | 618 |

**Conclusion:** BitNet xlarge still wins. The scale_factor improves PentaNet's internal distribution but is not enough to close the gap.

---

## Final Diagnosis

### Why BitNet Wins
BitNet has 3 states — no risk of bad distribution. Absmean scales naturally well for {-1, 0, +1}.

### Root Cause
The pentanary space {-2,-1,0,+1,+2} with absmean scaling is too difficult to optimize:
- sf=1.0: outer states under-used (22%) → loss of resolution
- sf=0.6: outer states over-used (47%) → gradient saturation
- sf=0.8: good balance (34%) but PPL still worse than BitNet

### Verdict
**PentaNet is not viable as a BitNet alternative at scale in its current form.** The 6.4% advantage at 124M does not transfer. Pentanary is a richer but harder optimization space — absmean + standard STE cannot exploit it.

### Future Directions
- Learnable per-layer scale_factor
- Different STE (soft quantization, Gumbel-softmax)
- Per-neuron scaling instead of per-layer
- Initialization from pre-trained FP32 model

---

## Code Changes Applied

- `--scale_factor` CLI arg in `train_pentagpt.py`
- `QuantLinear` accepts `scale_factor` parameter
- `count_weight_distribution()` uses `m.scale_factor`
- New size `short_wide` (12 layers x 1536 embed)
- Per-layer bucket logging in `results_log`
