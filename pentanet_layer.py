import torch
import torch.nn as nn
import torch.nn.functional as F

# Optional: PentaKernel fast inference (requires CUDA + Triton)
try:
    from penta_kernel import quantize_and_pack, penta_linear as _penta_linear_fast
    _KERNEL_AVAILABLE = True
except ImportError:
    _KERNEL_AVAILABLE = False


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

    def to_fast_inference(self) -> 'PentaLinearFast':
        """
        Convert this layer to a PentaLinearFast module for optimized
        CUDA inference using the 3-bit Triton kernel.

        Call this after training is complete, before inference:
            model.eval()
            for m in model.modules():
                if isinstance(m, PentaLinear):
                    # Replace in parent module manually, or use:
                    pass
            # Or use: convert_to_fast_inference(model)
        """
        assert _KERNEL_AVAILABLE, "penta_kernel not found — is Triton installed?"
        assert self.weight.is_cuda, "Weights must be on CUDA"
        return PentaLinearFast.from_pentalinear(self)


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
# 1c. FAST INFERENCE LAYER (3-bit Triton Kernel)
# ============================================================

class PentaLinearFast(nn.Module):
    """
    Drop-in replacement for PentaLinear using the 3-bit Triton kernel.
    Weights are stored in packed int32 (3 bits effective per weight).

    Memory vs FP16:   ~5.33× smaller
    Memory vs BitNet:  1.5× more bits → 1.47× more information (log2(5)/log2(3))

    Usage:
        model.eval()
        model = convert_to_fast_inference(model.cuda())
        y = model(x_bfloat16)
    """

    def __init__(self, packed_w: torch.Tensor, scale: float, K_orig: int,
                 bias: torch.Tensor | None = None):
        super().__init__()
        assert _KERNEL_AVAILABLE, "penta_kernel (Triton) is required"
        self.register_buffer('packed_w', packed_w)   # (N, K_packs) int32
        self.scale  = scale
        self.K_orig = K_orig
        if bias is not None:
            self.register_buffer('bias', bias)
        else:
            self.bias = None

    @classmethod
    def from_pentalinear(cls, layer: 'PentaLinear') -> 'PentaLinearFast':
        """Quantize + 3-bit pack the master weights from a trained PentaLinear."""
        packed, K_orig, scale = quantize_and_pack(layer.weight.detach())
        bias = layer.bias.detach().clone() if layer.bias is not None else None
        return cls(packed, scale, K_orig, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype != torch.bfloat16:
            x = x.to(torch.bfloat16)
        return _penta_linear_fast(x, self.packed_w, self.scale, self.K_orig, self.bias)

    def extra_repr(self) -> str:
        N, K_packs = self.packed_w.shape
        mb = N * self.K_orig * 3 / 8 / 1e6
        return (f'in={self.K_orig}, out={N}, scale={self.scale:.4f}, '
                f'packed_size={mb:.3f} MB (3-bit)')


def convert_to_fast_inference(model: nn.Module) -> nn.Module:
    """
    Recursively convert all PentaLinear layers in a model to PentaLinearFast.
    Call after training completes, before deployment.

    Example:
        model.eval()
        model = convert_to_fast_inference(model.cuda())
        y = model(x.to(torch.bfloat16))
    """
    assert _KERNEL_AVAILABLE, "penta_kernel (Triton) is required"
    for name, module in list(model.named_children()):
        if isinstance(module, PentaLinear):
            setattr(model, name, PentaLinearFast.from_pentalinear(module))
        else:
            convert_to_fast_inference(module)
    return model


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
