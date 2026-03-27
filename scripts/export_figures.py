"""Generate publication-quality PPL convergence figure (white background, PDF-ready)."""
import json, glob, math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Publication styling
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['DejaVu Sans']
rcParams['font.size'] = 12
rcParams['axes.linewidth'] = 1.2

files = sorted(glob.glob('models/*_s[0-9]*_results.json'))
runs = []
for f in files:
    with open(f) as fh:
        d = json.load(fh)
        if len(d.get('log', [])) >= 10:
            runs.append(d)

# ── Figure 1: PPL Convergence ──
fig, ax = plt.subplots(figsize=(8, 5))

color_map = {
    ('pentanet', 42): '#1f77b4', ('pentanet', 1337): '#2ca02c', ('pentanet', 2026): '#9467bd',
    ('bitnet', 42): '#d62728',   ('bitnet', 1337): '#ff7f0e',   ('bitnet', 2026): '#8c564b',
}
style_map = {'pentanet': '-', 'bitnet': '--'}

for run in runs:
    logs = run['log'][1:]  # skip iter 0
    iters = [l['iter'] for l in logs]
    ppls  = [l['ppl'] for l in logs]
    mode  = run['mode']
    seed  = run['seed']
    color = color_map.get((mode, seed), '#333333')
    ax.plot(iters, ppls,
            linestyle=style_map[mode],
            color=color,
            linewidth=2.0,
            label=f'{mode.upper()} (seed {seed})')

ax.set_yscale('log')
ax.set_xlabel('Training Iterations', fontsize=13)
ax.set_ylabel('Validation Perplexity', fontsize=13)
ax.set_title('PentaNet vs BitNet — Perplexity Convergence on WikiText-103', fontsize=14, fontweight='bold')
ax.legend(fontsize=9, ncol=2, loc='upper right', framealpha=0.9)
ax.grid(True, which='both', ls='--', alpha=0.3)
ax.tick_params(labelsize=11)

fig.tight_layout()
fig.savefig('figure1_ppl_convergence.png', dpi=300, facecolor='white')
fig.savefig('figure1_ppl_convergence.pdf', facecolor='white')
print("✅ figure1_ppl_convergence.png (300 dpi, white bg)")
print("✅ figure1_ppl_convergence.pdf")

# ── Figure 2: Weight Distribution Evolution ──
fig2, ax2 = plt.subplots(figsize=(8, 5))

penta42 = [r for r in runs if r['mode'] == 'pentanet' and r['seed'] == 42]
if penta42:
    logs = penta42[0]['log']
    iters = [l['iter'] for l in logs]
    buckets = ['-2', '-1', '0', '1', '2']
    colors = ['#d62728', '#1f77b4', '#2ca02c', '#9467bd', '#ff7f0e']
    
    for b, c in zip(buckets, colors):
        pcts = []
        for l in logs:
            w = l['weights']
            total = sum(w.values())
            pcts.append(w[b] / total * 100)
        ax2.plot(iters, pcts, color=c, linewidth=2.5, label=f'Bucket [{b}]')

    ax2.set_xlabel('Training Iterations', fontsize=13)
    ax2.set_ylabel('Weight Distribution (%)', fontsize=13)
    ax2.set_title('PentaNet Weight Bucket Stability (Seed 42)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10, ncol=5, loc='upper center', bbox_to_anchor=(0.5, -0.12), framealpha=0.9)
    ax2.grid(True, ls='--', alpha=0.3)
    ax2.tick_params(labelsize=11)
    ax2.set_ylim(0, 40)

    fig2.tight_layout()
    fig2.savefig('figure2_weight_distribution.png', dpi=300, facecolor='white')
    fig2.savefig('figure2_weight_distribution.pdf', facecolor='white')
    print("✅ figure2_weight_distribution.png (300 dpi, white bg)")
    print("✅ figure2_weight_distribution.pdf")
