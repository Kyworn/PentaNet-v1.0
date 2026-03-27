import os
import math
import argparse
import time
import json
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from transformers import AutoTokenizer

# =========================================================================
# 1. COUCHE LINEAIRE QUANTIFIÉE (PentaNet vs BitNet)
# =========================================================================

class QuantLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=False, mode='pentanet'):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.mode = mode  # 'pentanet' or 'bitnet'
        
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * (1.0 / math.sqrt(in_features)))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        scale = self.weight.abs().mean().clamp(min=1e-8)
        w_scaled = self.weight / scale
        
        if self.mode == 'pentanet':
            w_quant = torch.clamp(torch.round(w_scaled), -2, 2)
        else:
            w_quant = torch.clamp(torch.round(w_scaled), -1, 1)
            
        # Straight-Through Estimator (STE)
        w_ste = (w_quant * scale - self.weight).detach() + self.weight
        return F.linear(x, w_ste, self.bias)

# =========================================================================
# 2. ARCHITECTURE GPT
# =========================================================================

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = QuantLinear(config.n_embd, 3 * config.n_embd, bias=config.bias, mode=config.mode)
        self.c_proj = QuantLinear(config.n_embd, config.n_embd, bias=config.bias, mode=config.mode)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = QuantLinear(config.n_embd, 4 * config.n_embd, bias=config.bias, mode=config.mode)
        self.gelu = nn.GELU()
        self.c_proj = QuantLinear(4 * config.n_embd, config.n_embd, bias=config.bias, mode=config.mode)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class PentaGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = QuantLinear(config.n_embd, config.vocab_size, bias=False, mode=config.mode)
        self.transformer.wte.weight = self.lm_head.weight  # weight tying

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None
            
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


class GPTConfig:
    def __init__(self, size="small", mode='pentanet', vocab_size=50257, block_size=256):
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.bias = False
        self.dropout = 0.1
        self.mode = mode

        if size == "small":
            self.n_layer = 6
            self.n_head = 6
            self.n_embd = 384
        elif size == "medium":
            self.n_layer = 8
            self.n_head = 8
            self.n_embd = 512
        elif size == "large":
            self.n_layer = 12
            self.n_head = 12
            self.n_embd = 768


class DataLoader:
    def __init__(self, data_path, batch_size, block_size, seed=42):
        self.data = np.fromfile(data_path, dtype=np.uint32)
        self.batch_size = batch_size
        self.block_size = block_size
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def get_batch(self, device):
        # random sampling a batch
        ix = self.rng.randint(0, len(self.data) - self.block_size, (self.batch_size,))
        x = torch.stack([torch.from_numpy(self.data[i:i + self.block_size].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(self.data[i + 1:i + 1 + self.block_size].astype(np.int64)) for i in ix])
        return x.to(device), y.to(device)


# =========================================================================
# 4. ENTRAÎNEMENT — DUEL PentaNet vs BitNet
# =========================================================================

def count_weight_distribution(model):
    """Count how many weights fall into each quantized bucket."""
    counts = {-2: 0, -1: 0, 0: 0, 1: 0, 2: 0}
    total = 0
    for m in model.modules():
        if isinstance(m, QuantLinear):
            scale = m.weight.abs().mean().clamp(min=1e-8)
            w_scaled = m.weight / scale
            if m.mode == 'pentanet':
                w_q = torch.clamp(torch.round(w_scaled), -2, 2)
            else:
                w_q = torch.clamp(torch.round(w_scaled), -1, 1)
            for val in counts:
                counts[val] += (w_q == val).sum().item()
            total += w_q.numel()
    return counts, total


def train():
    parser = argparse.ArgumentParser(description="PentaGPT Training — Duel PentaNet vs BitNet")
    parser.add_argument('--size', type=str, default='large', choices=['small', 'medium', 'large'])
    parser.add_argument('--mode', type=str, default='pentanet', choices=['pentanet', 'bitnet'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_iters', type=int, default=10000)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--eval_interval', type=int, default=500)
    parser.add_argument('--eval_iters', type=int, default=50)
    parser.add_argument('--data_dir', type=str, default='data/wikitext-103')
    parser.add_argument('--no_wandb', action='store_true', help='Disable WandB logging')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # --- Load Data binary ---
    train_bin = os.path.join(args.data_dir, 'train.bin')
    val_bin = os.path.join(args.data_dir, 'validation.bin')
    
    if not os.path.exists(train_bin):
        print(f"ERROR: {train_bin} not found. Run prepare_data.py first.")
        return

    # Using GPT2Tokenizer vocab size
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    vocab_size = len(tokenizer)

    config = GPTConfig(size=args.size, mode=args.mode, vocab_size=vocab_size, block_size=512)
    model = PentaGPT(config).to(device)

    train_loader = DataLoader(train_bin, args.batch_size, config.block_size, seed=args.seed)
    val_loader = DataLoader(val_bin, args.batch_size, config.block_size, seed=args.seed)

    n_params = sum(p.numel() for p in model.parameters())
    mode_label = "PENTANET {-2..+2}" if args.mode == 'pentanet' else "BITNET {-1..+1}"
    print(f"🚀 Training {mode_label} ({args.size}, seed={args.seed}) : {n_params / 1e6:.2f}M params on {device}")

    # --- WandB ---
    use_wandb = not args.no_wandb
    if use_wandb:
        try:
            import wandb
            # Only track if not disabled
            wandb.init(project="PentaNet-Scaling", name=f"{args.mode}-{args.size}-s{args.seed}",
                       config={"mode": args.mode, "size": args.size, "params_M": n_params / 1e6,
                               "lr": args.lr, "max_iters": args.max_iters, "batch_size": args.batch_size, "seed": args.seed})
        except Exception:
            print("WandB not found/configured, running without.")
            use_wandb = False

    @torch.no_grad()
    def estimate_loss(eval_iters=args.eval_iters):
        model.eval()
        losses = {}
        # We reuse the same loaders, which advances their RNG state. It's fine for approx eval
        for split, loader in [('train', train_loader), ('val', val_loader)]:
            batch_losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                xb, yb = loader.get_batch(device)
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    logits, loss = model(xb, yb)
                batch_losses[k] = loss.item()
                del logits # Free memory
            losses[split] = batch_losses.mean().item()
        model.train()
        return losses

    # --- Optimizer + LR Schedule ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)

    def get_lr(it):
        warmup_iters = min(500, args.max_iters // 10)
        min_lr = args.lr / 10
        if it < warmup_iters:
            return args.lr * (it + 1) / warmup_iters
        if it >= args.max_iters:
            return min_lr
        decay_ratio = (it - warmup_iters) / (args.max_iters - warmup_iters)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (args.lr - min_lr)

    # --- Training Loop ---
    os.makedirs('models', exist_ok=True)
    best_val_loss = float('inf')
    results_log = []
    t0 = time.time()

    for it in range(args.max_iters):
        # Update LR
        lr = get_lr(it)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        xb, yb = train_loader.get_batch(device)
        
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            logits, loss = model(xb, yb)
            
        # Free logits immediately to save memory (shape is BxTxVocab_size = massive)
        del logits

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # --- Eval ---
        if it % args.eval_interval == 0 or it == args.max_iters - 1:
            torch.cuda.empty_cache()  # Clear cache before eval
            losses = estimate_loss()
            train_loss = losses['train']
            val_loss = losses['val']
            ppl = math.exp(val_loss) if val_loss < 20 else float('inf')
            elapsed = time.time() - t0

            # Weight distribution
            w_counts, w_total = count_weight_distribution(model)

            # Generate sample
            model.eval()
            context = torch.zeros((1, 1), dtype=torch.long, device=device) + tokenizer.bos_token_id if tokenizer.bos_token_id else torch.zeros((1, 1), dtype=torch.long, device=device)
            gen_ids = model.generate(context, max_new_tokens=40, temperature=0.8, top_k=40)[0].tolist()
            gen_text = tokenizer.decode(gen_ids).replace('\n', ' ')
            model.train()

            print(f"\n[{args.mode.upper()} S{args.seed} | Iter {it:5d}/{args.max_iters}] "
                  f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  PPL={ppl:.1f}  "
                  f"lr={lr:.2e}  time={elapsed:.0f}s")
            # Only print relevant buckets (ignore -2, 2 for bitnet)
            if args.mode == "bitnet":
                print(f"  Weights: {{-1: {w_counts[-1]}, 0: {w_counts[0]}, 1: {w_counts[1]}}}")
            else:
                print(f"  Weights: {w_counts}")
            print(f"  Gen: \"{gen_text}\"")

            entry = {"iter": it, "train_loss": train_loss, "val_loss": val_loss,
                     "ppl": ppl, "lr": lr, "elapsed": elapsed, "weights": w_counts}
            results_log.append(entry)

            if use_wandb:
                import wandb
                log_data = {"iter": it, "train_loss": train_loss, "val_loss": val_loss,
                            "ppl": ppl, "lr": lr}
                for k, v in w_counts.items():
                    log_data[f"w_{k}"] = v / max(w_total, 1)
                wandb.log(log_data)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                ckpt_path = f"models/{args.mode}_{args.size}_s{args.seed}_best.pt"
                torch.save(model.state_dict(), ckpt_path)

    # --- Final Summary ---
    total_time = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"🏁 FINISHED: {args.mode.upper()} ({args.size}, seed={args.seed}) — {n_params / 1e6:.2f}M params")
    print(f"   Best val_loss: {best_val_loss:.4f}  |  PPL: {math.exp(best_val_loss) if best_val_loss < 20 else 'inf'}")
    print(f"   Total time: {total_time:.0f}s")
    print(f"{'=' * 60}")

    # Save results
    results_path = f"models/{args.mode}_{args.size}_s{args.seed}_results.json"
    with open(results_path, 'w') as f:
        json.dump({"mode": args.mode, "size": args.size, "seed": args.seed, "params_M": n_params / 1e6,
                   "best_val_loss": best_val_loss, "total_time": total_time,
                   "log": results_log}, f, indent=2)
    print(f"📊 Results saved to {results_path}")

    if use_wandb:
        import wandb
        wandb.finish()


if __name__ == '__main__':
    train()
