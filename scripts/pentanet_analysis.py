#!/usr/bin/env python3
"""
PentaNet Phase 1 — Proof of Concept (Offline)
===============================================
Compare la perte d'information entre :
  - Binaire    {-1, +1}          (1.0 bit)
  - Ternaire   {-1, 0, +1}      (1.58 bit)  — BitNet b1.58
  - Pentanaire {-2, -1, 0, +1, +2} (2.32 bit) — PentaNet
  - INT4       [-7, +7]          (4.0 bit)

Sur des poids simulés réalistes (distribution gaussienne d'un transformer).
"""

import torch
import numpy as np
import json
import os

torch.manual_seed(42)
np.random.seed(42)

# ============================================================
# 1. GÉNÉRATION DE POIDS RÉALISTES
# ============================================================
print("=" * 60)
print("🧬 PentaNet Phase 1 — Analyse de Quantification")
print("=" * 60)

# Simuler les poids d'un modèle 124M params (GPT-2 scale)
# Distribution : Gaussienne N(0, σ) avec σ variant selon le type de couche
total_params = 124_000_000

print(f"\n📥 Génération de poids simulés (distribution transformer réaliste)...")
print(f"   Total paramètres : {total_params:,} ({total_params/1e6:.1f}M)")

# Les poids d'un transformer réel : 
# - Attention : σ ≈ 0.02
# - FFN/MLP  : σ ≈ 0.02-0.04
# - LayerNorm: σ ≈ 0.1 (autour de 1.0)
# - Embeddings: σ ≈ 0.02

attention_params = int(total_params * 0.33)  # ~33% attention
ffn_params = int(total_params * 0.50)        # ~50% FFN
layernorm_params = int(total_params * 0.01)  # ~1% LayerNorm  
embedding_params = total_params - attention_params - ffn_params - layernorm_params

layer_configs = {
    "attention": (attention_params, 0.02),
    "mlp/ffn":   (ffn_params, 0.03),
    "layernorm": (layernorm_params, 0.1),
    "embedding": (embedding_params, 0.02),
}

all_weights_parts = []
layer_weights = {}

for name, (n, sigma) in layer_configs.items():
    w = torch.randn(n) * sigma
    all_weights_parts.append(w)
    layer_weights[name] = w
    print(f"   {name:>12}: {n:>12,} params, σ={sigma}")

all_weights = torch.cat(all_weights_parts)
print(f"   {'TOTAL':>12}: {len(all_weights):>12,} params")

# ============================================================
# 2. STATISTIQUES DE LA DISTRIBUTION DES POIDS
# ============================================================
print("\n📊 Distribution des poids originaux (FP32) :")
mean = all_weights.mean().item()
std = all_weights.std().item()
abs_mean = all_weights.abs().mean().item()
min_w = all_weights.min().item()
max_w = all_weights.max().item()
print(f"   Moyenne     : {mean:.6f}")
print(f"   Écart-type  : {std:.6f}")
print(f"   |Moyenne|   : {abs_mean:.6f}")
print(f"   Min / Max   : [{min_w:.4f}, {max_w:.4f}]")

# ============================================================
# 3. FONCTIONS DE QUANTIFICATION
# ============================================================

def quantize_ternary(weights, block_size=64):
    """Ternaire {-1, 0, +1} style BitNet b1.58 — absmean scaling."""
    n = len(weights)
    quantized = torch.zeros_like(weights)
    scales = []
    
    for i in range(0, n, block_size):
        block = weights[i:i+block_size]
        scale = block.abs().mean()
        if scale == 0:
            scales.append(0.0)
            continue
        normalized = block / scale
        q = torch.clamp(torch.round(normalized), -1, 1)
        quantized[i:i+block_size] = q * scale
        scales.append(scale.item())
    
    return quantized, scales


def quantize_pentanary(weights, block_size=64):
    """Pentanaire {-2, -1, 0, +1, +2} — PentaNet, absmean scaling ×2."""
    n = len(weights)
    quantized = torch.zeros_like(weights)
    scales = []
    
    for i in range(0, n, block_size):
        block = weights[i:i+block_size]
        # Scale : absmean / 1 pour que la majorité tombe dans [-2, +2]
        scale = block.abs().mean()
        if scale == 0:
            scales.append(0.0)
            continue
        normalized = block / scale
        q = torch.clamp(torch.round(normalized), -2, 2)
        quantized[i:i+block_size] = q * scale
        scales.append(scale.item())
    
    return quantized, scales


def quantize_pentanary_maxscale(weights, block_size=64):
    """Pentanaire {-2, -1, 0, +1, +2} — scaling par max/2."""
    n = len(weights)
    quantized = torch.zeros_like(weights)
    scales = []
    
    for i in range(0, n, block_size):
        block = weights[i:i+block_size]
        scale = block.abs().max() / 2.0
        if scale == 0:
            scales.append(0.0)
            continue
        normalized = block / scale
        q = torch.clamp(torch.round(normalized), -2, 2)
        quantized[i:i+block_size] = q * scale
        scales.append(scale.item())
    
    return quantized, scales


def quantize_binary(weights, block_size=64):
    """Binaire {-1, +1} — baseline 1-bit."""
    n = len(weights)
    quantized = torch.zeros_like(weights)
    scales = []
    
    for i in range(0, n, block_size):
        block = weights[i:i+block_size]
        scale = block.abs().mean()
        if scale == 0:
            scales.append(0.0)
            continue
        q = torch.sign(block)
        q[q == 0] = 1
        quantized[i:i+block_size] = q * scale
        scales.append(scale.item())
    
    return quantized, scales


def quantize_int4(weights, block_size=64):
    """INT4 symétrique [-7, +7] — baseline 4-bit."""
    n = len(weights)
    quantized = torch.zeros_like(weights)
    scales = []
    
    for i in range(0, n, block_size):
        block = weights[i:i+block_size]
        scale = block.abs().max() / 7.0
        if scale == 0:
            scales.append(0.0)
            continue
        normalized = block / scale
        q = torch.clamp(torch.round(normalized), -7, 7)
        quantized[i:i+block_size] = q * scale
        scales.append(scale.item())
    
    return quantized, scales

# ============================================================
# 4. EXÉCUTION DES QUANTIFICATIONS
# ============================================================
print("\n⚙️  Quantification en cours...")

methods = {
    "Binaire {-1,+1}       (1.0 bit)": quantize_binary,
    "Ternaire {-1,0,+1}    (1.58 bit) — BitNet": quantize_ternary,
    "PentaNet-absmean       (2.32 bit)": quantize_pentanary,
    "PentaNet-maxscale      (2.32 bit)": quantize_pentanary_maxscale,
    "INT4 [-7,+7]           (4.0 bit)": quantize_int4,
}

results = {}

for name, func in methods.items():
    q_weights, scales = func(all_weights.clone())
    
    mse = ((all_weights - q_weights) ** 2).mean().item()
    rmse = mse ** 0.5
    cos_sim = torch.nn.functional.cosine_similarity(
        all_weights.unsqueeze(0), q_weights.unsqueeze(0)
    ).item()
    signal_power = (all_weights ** 2).mean().item()
    noise_power = mse
    snr_db = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')
    
    # Relative error
    rel_err = (((all_weights - q_weights).abs()) / (all_weights.abs() + 1e-10)).mean().item()
    
    results[name] = {
        "mse": mse,
        "rmse": rmse,
        "cosine_similarity": cos_sim,
        "snr_db": snr_db,
        "relative_error": rel_err,
    }
    
    print(f"\n   📐 {name}")
    print(f"      MSE            : {mse:.10f}")
    print(f"      RMSE           : {rmse:.10f}")
    print(f"      Cosine Sim     : {cos_sim:.8f}")
    print(f"      SNR            : {snr_db:.2f} dB")
    print(f"      Err. relative  : {rel_err*100:.2f}%")

# ============================================================
# 5. COMPARAISON DIRECTE
# ============================================================
print("\n" + "=" * 60)
print("📊 TABLEAU COMPARATIF")
print("=" * 60)

print(f"\n   {'Méthode':<35} {'MSE':>12} {'Cosine':>10} {'SNR(dB)':>8} {'Err%':>8}")
print(f"   {'─'*35} {'─'*12} {'─'*10} {'─'*8} {'─'*8}")

for name, r in results.items():
    short = name.split("(")[0].strip()
    print(f"   {short:<35} {r['mse']:>12.10f} {r['cosine_similarity']:>10.8f} {r['snr_db']:>8.2f} {r['relative_error']*100:>7.2f}%")

# ============================================================
# 6. PENTANET VS BITNET
# ============================================================
print("\n" + "=" * 60)
print("🔬 PENTANET vs BITNET — Analyse détaillée")
print("=" * 60)

# Utiliser le meilleur scaling pour PentaNet
penta_absmean = results["PentaNet-absmean       (2.32 bit)"]
penta_maxscale = results["PentaNet-maxscale      (2.32 bit)"]
ternary = results["Ternaire {-1,0,+1}    (1.58 bit) — BitNet"]
int4 = results["INT4 [-7,+7]           (4.0 bit)"]

# Choisir le meilleur PentaNet
if penta_absmean["mse"] < penta_maxscale["mse"]:
    penta = penta_absmean
    penta_best = "absmean"
else:
    penta = penta_maxscale
    penta_best = "maxscale"

print(f"\n   Meilleur scaling PentaNet : {penta_best}")

mse_improvement = (1 - penta["mse"] / ternary["mse"]) * 100
snr_improvement = penta["snr_db"] - ternary["snr_db"]
cos_improvement = penta["cosine_similarity"] - ternary["cosine_similarity"]

print(f"\n   PentaNet vs BitNet Ternaire :")
print(f"   ├── MSE réduite de           : {mse_improvement:+.1f}%")
print(f"   ├── SNR supérieur de          : {snr_improvement:+.2f} dB")
print(f"   ├── Cosine Sim gain           : {cos_improvement:+.10f}")
print(f"   └── Erreur relative PentaNet  : {penta['relative_error']*100:.2f}% (vs {ternary['relative_error']*100:.2f}% BitNet)")

print(f"\n   Coût mémoire (pour 124M params) :")
print(f"   ├── BitNet   : ~{total_params * 1.58 / 8 / 1e6:.1f} Mo  (1.58 bit/param)")
print(f"   ├── PentaNet : ~{total_params * 2.66 / 8 / 1e6:.1f} Mo  (2.66 bit/param) → +{((2.66/1.58)-1)*100:.0f}% mémoire")
print(f"   └── INT4     : ~{total_params * 4 / 8 / 1e6:.1f} Mo  (4.0 bit/param)")

print(f"\n   Ratio qualité/mémoire :")
mem_penta = 2.66
mem_tern = 1.58
mem_int4 = 4.0
quality_penta = penta["snr_db"]
quality_tern = ternary["snr_db"]
quality_int4 = int4["snr_db"]
print(f"   ├── BitNet   : {quality_tern/mem_tern:.2f} dB/bit")
print(f"   ├── PentaNet : {quality_penta/mem_penta:.2f} dB/bit")
print(f"   └── INT4     : {quality_int4/mem_int4:.2f} dB/bit")

# ============================================================
# 7. ANALYSE PAR TYPE DE COUCHE
# ============================================================
print("\n" + "=" * 60)
print("🔬 ANALYSE PAR TYPE DE COUCHE")
print("=" * 60)

for cat, w in layer_weights.items():
    q_tern, _ = quantize_ternary(w.clone())
    q_penta, _ = quantize_pentanary(w.clone())
    
    mse_t = ((w - q_tern) ** 2).mean().item()
    mse_p = ((w - q_penta) ** 2).mean().item()
    improvement = (1 - mse_p / mse_t) * 100 if mse_t > 0 else 0
    
    cos_t = torch.nn.functional.cosine_similarity(w.unsqueeze(0), q_tern.unsqueeze(0)).item()
    cos_p = torch.nn.functional.cosine_similarity(w.unsqueeze(0), q_penta.unsqueeze(0)).item()
    
    print(f"\n   {cat.upper()} ({len(w):,} params, σ={w.std():.4f})")
    print(f"   ├── BitNet    MSE: {mse_t:.10f}  cos: {cos_t:.8f}")
    print(f"   ├── PentaNet  MSE: {mse_p:.10f}  cos: {cos_p:.8f}")
    print(f"   └── Gain PentaNet : {improvement:+.1f}% MSE")

# ============================================================
# 8. DISTRIBUTION DES VALEURS PENTANAIRES
# ============================================================
print("\n" + "=" * 60)
print("📊 DISTRIBUTION DES VALEURS PENTANAIRES")
print("=" * 60)

block_size = 64
value_counts = {-2: 0, -1: 0, 0: 0, 1: 0, 2: 0}
for i in range(0, len(all_weights), block_size):
    block = all_weights[i:i+block_size]
    scale = block.abs().mean()
    if scale == 0:
        value_counts[0] += len(block)
        continue
    normalized = block / scale
    q = torch.clamp(torch.round(normalized), -2, 2).int()
    for v in [-2, -1, 0, 1, 2]:
        value_counts[v] += (q == v).sum().item()

total_q = sum(value_counts.values())
print(f"\n   Valeur | Nombre         | Pourcentage | Barre")
print(f"   -------|----------------|-------------|------")
for v in [-2, -1, 0, 1, 2]:
    count = value_counts[v]
    pct = count / total_q * 100
    bar = "█" * int(pct)
    label = {-2: "-2", -1: "-1", 0: " 0", 1: "+1", 2: "+2"}[v]
    print(f"     {label}   | {count:>14,} | {pct:>9.1f}%  | {bar}")

# Calcul de l'entropie effective
probs = np.array([value_counts[v] / total_q for v in [-2, -1, 0, 1, 2]])
probs = probs[probs > 0]
entropy = -np.sum(probs * np.log2(probs))
print(f"\n   Entropie effective : {entropy:.3f} bits (max théorique: {np.log2(5):.3f} bits)")
print(f"   Efficacité        : {entropy / np.log2(5) * 100:.1f}%")

# ============================================================
# 9. VERDICT
# ============================================================
print("\n" + "=" * 60)
print("🏆 VERDICT")
print("=" * 60)

if mse_improvement > 40:
    verdict = "TRÈS PROMETTEUR"
    emoji = "🔥"
    detail = f"PentaNet réduit la MSE de {mse_improvement:.1f}% vs BitNet. Le gain justifie le surcoût mémoire de 68%."
elif mse_improvement > 20:
    verdict = "PROMETTEUR"
    emoji = "✅"
    detail = f"PentaNet réduit la MSE de {mse_improvement:.1f}% vs BitNet. À valider avec un entraînement natif."
elif mse_improvement > 5:
    verdict = "MARGINAL"
    emoji = "⚠️"
    detail = f"PentaNet réduit la MSE de seulement {mse_improvement:.1f}% vs BitNet. Le surcoût mémoire de 68% est difficilement justifiable."
else:
    verdict = "NON CONCLUANT"
    emoji = "❌"
    detail = f"Gain de {mse_improvement:.1f}%. Les 2 états supplémentaires n'apportent pas assez vs BitNet."

print(f"\n   {emoji} {verdict}")
print(f"   {detail}")

print(f"\n   ⚠️  RAPPELS IMPORTANTS :")
print(f"   1. Ceci est une analyse PTQ sur des poids SIMULÉS (pas un vrai modèle)")
print(f"   2. BitNet b1.58 perd en PTQ mais récupère en entraînement natif")
print(f"   3. Le vrai test = entraîner un modèle nativement pentanaire")
print(f"   4. Le ratio qualité/bit est plus important que la MSE brute")

# Sauvegarder
output = {
    "model": "Simulated Transformer 124M (Gaussian weights)",
    "total_params": total_params,
    "results": {k: v for k, v in results.items()},
    "pentanet_vs_bitnet": {
        "best_scaling": penta_best,
        "mse_reduction_pct": mse_improvement,
        "snr_gain_db": snr_improvement,
        "memory_overhead_pct": ((2.66/1.58)-1)*100,
    },
    "pentanary_distribution": {str(k): v for k, v in value_counts.items()},
    "entropy_bits": entropy,
    "verdict": verdict,
}

output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pentanet_results.json")
with open(output_path, "w") as f:
    json.dump(output, f, indent=2)

print(f"\n   💾 Résultats sauvegardés dans pentanet_results.json")
print("=" * 60)
