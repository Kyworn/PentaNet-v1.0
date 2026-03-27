import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. IMPLÉMENTATION DES COUCHES (STE - Straight-Through Estimator)
# ============================================================

class PentaLinear(nn.Module):
    """
    Couche Linéaire pour PentaNet contrainte à {-2, -1, 0, +1, +2}.
    Utilise le Straight-Through Estimator (STE) pour la backpropagation.
    """
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Les poids maîtres (latents) restent en haute précision (FP32) pour accumuler les petits gradients
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * (1.0 / in_features**0.5))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        # 1. Calcul du scale (absmean, validé en Phase 1)
        scale = self.weight.abs().mean().clamp(min=1e-8)
        
        # 2. Quantification Pentanaire
        w_scaled = self.weight / scale
        w_quant = torch.clamp(torch.round(w_scaled), -2, 2)
        
        # 3. Straight-Through Estimator (STE)
        # En forward: w_quant * scale
        # En backward: comme si la dérivée de round() et clamp() était 1
        w_ste = (w_quant * scale - self.weight).detach() + self.weight

        # Optionnel: on peut aussi quantifier les activations (souvent en INT8 pour les 1-bit LLMs)
        # Mais pour cette preuve de concept, on se concentre sur les poids
        return F.linear(x, w_ste, self.bias)


class BitLinear(nn.Module):
    """
    Baseline : Couche Linéaire Ternaire {-1, 0, +1} (BitNet b1.58).
    """
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * (1.0 / in_features**0.5))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        scale = self.weight.abs().mean().clamp(min=1e-8)
        w_scaled = self.weight / scale
        w_quant = torch.clamp(torch.round(w_scaled), -1, 1)
        w_ste = (w_quant * scale - self.weight).detach() + self.weight
        return F.linear(x, w_ste, self.bias)

# ============================================================
# 2. TOY TRAINING LOOP (Preuve d'apprentissage)
# ============================================================
def train_toy_model():
    print("🚀 Début de l'entraînement Toy (Preuve d'apprentissage natif)")
    torch.manual_seed(42)

    # Paramètres du dataset Toy
    batch_size = 64
    in_dim = 128
    hidden_dim = 256
    out_dim = 10
    epochs = 400
    lr = 0.01

    # Création d'un dataset "Teacher-Student"
    # Le teacher est un simple réseau FP32 non linéaire
    teacher = nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, out_dim)
    )
    
    # Données fixes pour l'entraînement (overfitting test)
    X = torch.randn(batch_size, in_dim)
    with torch.no_grad():
        Y = teacher(X) # Cible

    # Création des étudiants : FP32 (baseline haute), PentaNet, BitNet
    student_fp32 = nn.Sequential(
        nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, out_dim)
    )
    student_penta = nn.Sequential(
        PentaLinear(in_dim, hidden_dim), nn.ReLU(), PentaLinear(hidden_dim, out_dim)
    )
    student_ternary = nn.Sequential(
        BitLinear(in_dim, hidden_dim), nn.ReLU(), BitLinear(hidden_dim, out_dim)
    )

    models = {
        "FP32 (Idéal)": student_fp32,
        "PentaNet {-2..+2}": student_penta,
        "BitNet {-1,0,1}": student_ternary
    }

    optimizers = {name: torch.optim.AdamW(model.parameters(), lr=lr) 
                  for name, model in models.items()}
    
    criterion = nn.MSELoss()
    history = {name: [] for name in models}

    print("\nEntraînement en cours (400 epochs)...")
    for epoch in range(epochs):
        for name, model in models.items():
            opt = optimizers[name]
            opt.zero_grad()
            
            output = model(X)
            loss = criterion(output, Y)
            loss.backward()
            opt.step()
            
            history[name].append(loss.item())

        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1:3d} | " + " | ".join(f"{n}: {history[n][-1]:.4f}" for n in models))

    print("\n✅ Entraînement terminé !")
    
    # Vérification des poids de PentaNet (est-ce que ça a bien convergé vers des valeurs discrètes ?)
    penta_layer = student_penta[0]
    scale = penta_layer.weight.abs().mean().item()
    unique_vals = torch.unique(torch.clamp(torch.round(penta_layer.weight / scale), -2, 2))
    print(f"\n🔍 Vérification des poids PentaNet (Couche 1) :")
    print(f"Valeurs quantifiées utilisées en Forward : {unique_vals.int().tolist()}")
    
    return history

if __name__ == "__main__":
    train_toy_model()
