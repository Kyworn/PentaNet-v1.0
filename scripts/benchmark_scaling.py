"""
PentaNet Scaling Benchmark
===========================
Small:  20M params, 100k iters  (~50 min)
Medium: 51M params, 30k iters   (~35 min)
Large:  85M params, 15k iters   (~25 min)
Total: ~2h on RTX 5080
"""
import os, math, time, json
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm

# ── Tokenizer (tiktoken si dispo, sinon char-level) ──────────────────────────
try:
    import tiktoken
    _enc = tiktoken.get_encoding("gpt2")
    def encode(text): return _enc.encode(text)
    def decode(ids):  return _enc.decode(ids)
    VOCAB_SIZE_BASE = 50304
    print("✓ tiktoken GPT-2 chargé")
except ImportError:
    print("⚠ tiktoken absent → tokenizer char-level")
    _char2i, _i2char = {}, {}
    def encode(text):
        global _char2i, _i2char
        if not _char2i:
            chars = sorted(set(text))
            _char2i = {c: i for i, c in enumerate(chars)}
            _i2char = {i: c for i, c in enumerate(chars)}
        return [_char2i.get(c, 0) for c in text]
    def decode(ids):
        return ''.join(_i2char.get(i, '') for i in ids)
    VOCAB_SIZE_BASE = None  # fixé après encode

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ── PentaLinear ───────────────────────────────────────────────────────────────
class PentaLinear(nn.Module):
    def __init__(self, in_f, out_f):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_f, in_f) * (1.0 / math.sqrt(in_f)))

    def forward(self, x):
        s = self.weight.abs().mean().clamp(min=1e-8)
        wq = torch.clamp(torch.round(self.weight / s), -2, 2)
        w  = (wq * s - self.weight).detach() + self.weight
        return F.linear(x, w)

    def weight_dist(self):
        with torch.no_grad():
            s  = self.weight.abs().mean().clamp(min=1e-8)
            wq = torch.clamp(torch.round(self.weight / s), -2, 2)
            n  = wq.numel()
            return {int(v): float((wq == v).sum()) / n for v in [-2, -1, 0, 1, 2]}

# ── Architecture GPT ──────────────────────────────────────────────────────────
class CausalSelfAttention(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.c_attn = PentaLinear(C.n_embd, 3 * C.n_embd)
        self.c_proj = PentaLinear(C.n_embd, C.n_embd)
        self.n_head, self.n_embd = C.n_head, C.n_embd
        self.register_buffer("mask", torch.tril(torch.ones(C.block_size, C.block_size))
                             .view(1, 1, C.block_size, C.block_size))

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        def sh(t): return t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q, k, v = sh(q), sh(k), sh(v)
        att = (q @ k.transpose(-2,-1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = (att @ v).transpose(1,2).contiguous().view(B, T, C)
        return self.c_proj(y)

class MLP(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.fc   = PentaLinear(C.n_embd, 4 * C.n_embd)
        self.proj = PentaLinear(4 * C.n_embd, C.n_embd)
    def forward(self, x):
        return self.proj(F.gelu(self.fc(x)))

class Block(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.ln1  = nn.LayerNorm(C.n_embd, bias=False)
        self.attn = CausalSelfAttention(C)
        self.ln2  = nn.LayerNorm(C.n_embd, bias=False)
        self.mlp  = MLP(C)
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class PentaGPT(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.C = C
        self.wte   = nn.Embedding(C.vocab_size, C.n_embd)
        self.wpe   = nn.Embedding(C.block_size, C.n_embd)
        self.drop  = nn.Dropout(0.1)
        self.blocks = nn.ModuleList([Block(C) for _ in range(C.n_layer)])
        self.ln_f  = nn.LayerNorm(C.n_embd, bias=False)
        self.head  = PentaLinear(C.n_embd, C.vocab_size)
        self.wte.weight = self.head.weight  # weight tying

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos  = torch.arange(T, device=idx.device)
        x    = self.drop(self.wte(idx) + self.wpe(pos))
        for b in self.blocks: x = b(x)
        x    = self.ln_f(x)
        if targets is not None:
            logits = self.head(x)
            loss   = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return logits, loss
        return self.head(x[:, [-1], :]), None

    @torch.no_grad()
    def generate(self, idx, n, temp=0.8, top_k=40):
        for _ in range(n):
            ic  = idx[:, -self.C.block_size:]
            log = self(ic)[0][:, -1, :] / temp
            v, _ = torch.topk(log, min(top_k, log.size(-1)))
            log[log < v[:, [-1]]] = -float('Inf')
            idx = torch.cat([idx, torch.multinomial(F.softmax(log,-1), 1)], dim=1)
        return idx

    def global_weight_dist(self):
        counts = {v: 0.0 for v in [-2,-1,0,1,2]}
        total  = 0
        for m in self.modules():
            if isinstance(m, PentaLinear):
                n = m.weight.numel()
                for v, f in m.weight_dist().items():
                    counts[v] += f * n
                total += n
        return {v: counts[v]/total for v in counts}

# ── Config ────────────────────────────────────────────────────────────────────
class C:
    block_size = 256

class CSmall(C):
    n_layer=6;  n_head=6;  n_embd=384;  batch_size=32; lr=3e-4; max_iters=100_000

class CMedium(C):
    n_layer=10; n_head=8;  n_embd=512;  batch_size=20; lr=3e-4; max_iters=30_000

class CLarge(C):
    n_layer=12; n_head=12; n_embd=768;  batch_size=12; lr=2e-4; max_iters=15_000

RUNS = [("small", CSmall), ("medium", CMedium), ("large", CLarge)]

# ── Data ──────────────────────────────────────────────────────────────────────
def load_corpus():
    texts = []
    for f in ["shakespeare.txt", "wikitext2_train.txt"]:
        if os.path.exists(f):
            with open(f, encoding="utf-8") as fh:
                texts.append(fh.read())
            print(f"  📖 {f} ({os.path.getsize(f)/1e6:.1f} MB)")
    if not texts:
        texts = [("To be or not to be, that is the question. "
                  "Whether 'tis nobler in the mind to suffer. ") * 30_000]
        print("  ⚠ corpus de secours")
    return "\n".join(texts)

def get_batch(data, block_size, batch_size):
    ix = np.random.randint(0, len(data) - block_size, (batch_size,))
    x  = torch.stack([torch.from_numpy(data[i:i+block_size].astype(np.int64)) for i in ix])
    y  = torch.stack([torch.from_numpy(data[i+1:i+1+block_size].astype(np.int64)) for i in ix])
    return x.to(DEVICE), y.to(DEVICE)

# ── Single run ────────────────────────────────────────────────────────────────
def run(name, Cfg, data, vocab_size):
    print(f"\n{'='*70}")
    print(f"  🧬  PentaGPT-{name.upper()}  |  {DEVICE.upper()}")
    print(f"{'='*70}")

    class Config(Cfg):
        pass
    Config.vocab_size = vocab_size

    model = PentaGPT(Config).to(DEVICE)
    n_p   = sum(p.numel() for p in model.parameters())
    print(f"  Params : {n_p/1e6:.1f}M  |  lr={Config.lr}  |  batch={Config.batch_size}  |  iters={Config.max_iters:,}")

    opt  = torch.optim.AdamW(model.parameters(), lr=Config.lr, weight_decay=0.1, betas=(0.9,0.95))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=Config.max_iters, eta_min=Config.lr/10)

    rec = {
        "name": name, "params_M": round(n_p/1e6, 1),
        "loss_curve": [], "samples": [],
        "dist_start": model.global_weight_dist(),
        "dist_end": None,
        "final_loss": None, "best_loss": None,
        "train_time_s": None, "iters_per_sec": None,
    }

    eval_every = Config.max_iters // 20   # 20 checkpoints
    smooth = None
    t0 = time.time()

    pbar = tqdm(range(Config.max_iters), desc=f"[{name}]")
    for it in pbar:
        xb, yb = get_batch(data, Config.block_size, Config.batch_size)
        _, loss = model(xb, yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        lv = loss.item()
        smooth = lv if smooth is None else 0.98*smooth + 0.02*lv

        if it % eval_every == 0 or it == Config.max_iters - 1:
            rec["loss_curve"].append({"iter": it, "loss": round(lv,4), "smooth": round(smooth,4)})
            model.eval()
            ctx = torch.zeros((1,1), dtype=torch.long, device=DEVICE)
            gen = model.generate(ctx, 60)[0].tolist()
            sample = decode(gen).replace('\n',' ').strip()
            rec["samples"].append({"iter": it, "text": sample})
            model.train()
            pbar.set_description(f"[{name}] loss={lv:.3f} (~{smooth:.3f})")
            print(f"\n  [it {it:6d}/{Config.max_iters}] loss={lv:.4f} | lr={sched.get_last_lr()[0]:.2e}")
            print(f"  ► {sample[:120]}")

    elapsed = time.time() - t0
    rec["dist_end"]     = model.global_weight_dist()
    rec["train_time_s"] = round(elapsed, 1)
    rec["iters_per_sec"]= round(Config.max_iters / elapsed, 2)
    rec["final_loss"]   = rec["loss_curve"][-1]["loss"]
    rec["best_loss"]    = min(e["loss"] for e in rec["loss_curve"])
    print(f"\n  ✅  {name} terminé  {elapsed/60:.1f}min  |  best={rec['best_loss']:.4f}  |  {rec['iters_per_sec']:.1f} it/s")
    return rec

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"\n{'#'*70}")
    print(f"  🔥  PentaNet Scaling Benchmark")
    print(f"  Device : {DEVICE}")
    if DEVICE == 'cuda':
        print(f"  GPU    : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM   : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    print(f"{'#'*70}\n")

    full_text  = load_corpus()
    raw        = encode(full_text)
    vocab_size = VOCAB_SIZE_BASE or ((max(raw) + 64) // 64 * 64)
    data       = np.array(raw, dtype=np.int32)
    print(f"\n  📊  {len(full_text)/1e6:.2f}M chars  |  {len(data)/1e6:.2f}M tokens  |  vocab={vocab_size}\n")

    all_results = []
    t0_total   = time.time()

    for name, Cfg in RUNS:
        r = run(name, Cfg, data, vocab_size)
        all_results.append(r)
        with open("benchmark_results.json","w") as f:
            json.dump(all_results, f, indent=2)
        print(f"  💾  benchmark_results.json mis à jour")

    # ── Rapport ───────────────────────────────────────────────────────────────
    total_min = (time.time() - t0_total) / 60
    print(f"\n\n{'#'*70}")
    print(f"  📊  RAPPORT FINAL — PentaNet Scaling  ({total_min:.0f} min)")
    print(f"{'#'*70}")
    hdr = f"  {'Size':<8} {'Params':>7} {'it/s':>7} {'LossInit':>10} {'LossFinal':>10} {'BestLoss':>10} {'Time':>7}"
    print(hdr)
    print(f"  {'-'*65}")
    for r in all_results:
        i0 = r["loss_curve"][0]["loss"]
        print(f"  {r['name']:<8} {r['params_M']:>6.1f}M {r['iters_per_sec']:>7.1f}"
              f" {i0:>10.4f} {r['final_loss']:>10.4f} {r['best_loss']:>10.4f} {r['train_time_s']/60:>6.1f}m")

    print(f"\n  Poids {'{-2,-1,0,+1,+2}'} distribution (fin entraînement):")
    print(f"  {'Size':<8} {'−2':>8} {'−1':>8} {'0':>8} {'+1':>8} {'+2':>8}")
    print(f"  {'-'*50}")
    for r in all_results:
        d = r["dist_end"]
        print(f"  {r['name']:<8} {d[-2]:>8.3f} {d[-1]:>8.3f} {d[0]:>8.3f} {d[1]:>8.3f} {d[2]:>8.3f}")
    print(f"\n  Données : benchmark_results.json\n{'#'*70}\n")

if __name__ == "__main__":
    main()
