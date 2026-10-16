"""Exp 02: weak vs strong MLP activation ablation in GPT-2 small (mass-matched)."""
import sys, json, time, torch, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, "../common"); from data import get_tokens, device
from transformer_lens import HookedTransformer
torch.manual_seed(0); dev = device()
m = HookedTransformer.from_pretrained("gpt2", device=dev); m.eval()
toks = get_tokens(m)[:128].to(dev); B = 16
def masks(a, q=0.75):
    """a: [tok, n] activations. returns weak mask (zero |a|<per-neuron p75) and mass-matched strong mask."""
    A = a.abs(); thr = torch.quantile(A.float().cpu(), q, dim=0).to(A.device)
    weak = A < thr; wmass = (A * weak).sum(0)
    s, idx = A.sort(0, descending=True); cs = s.cumsum(0)
    k = (cs < wmass).sum(0) + 1  # smallest k with cumsum >= wmass
    rank = torch.empty_like(idx); rank.scatter_(0, idx, torch.arange(A.shape[0], device=A.device)[:, None].expand_as(idx))
    strong = rank < k
    rnd = torch.rand_like(A) < (k.float() / A.shape[0])  # random entries, same count as strong
    return weak, strong, rnd
def run(layers, mode):
    tot = 0.
    for i in range(0, len(toks), B):
        t = toks[i:i+B]
        def hook(act, hook):
            sh = act.shape; a = act.reshape(-1, sh[-1])
            if mode == "none": return act
            w, s, r = masks(a); mk = {"weak": w, "strong": s, "random": r}[mode]
            return (a * (~mk)).reshape(sh)
        with torch.no_grad():
            loss = m.run_with_hooks(t, return_type="loss", fwd_hooks=[(f"blocks.{l}.mlp.hook_post", hook) for l in layers])
        tot += loss.item() * len(t)
    return tot / len(toks)
t0 = time.time(); base = run([], "none"); res = {"base": base, "per_layer": {}, "all": {}}
for mode in ["weak", "strong", "random"]:
    res["per_layer"][mode] = [run([l], mode) - base for l in range(12)]
    res["all"][mode] = run(list(range(12)), mode) - base
    print(mode, res["all"][mode], [round(x, 3) for x in res["per_layer"][mode]], flush=True)
res["runtime_s"] = time.time() - t0; res["n_tokens"] = int(toks.numel())
json.dump(res, open("results.json", "w"), indent=1)
fig, ax = plt.subplots(figsize=(7, 4))
for mode, c in [("weak", "#4caf50"), ("strong", "#e53935"), ("random", "#888")]:
    ax.plot(range(12), res["per_layer"][mode], "o-", c=c, label=f"{mode} (all layers: +{res['all'][mode]:.2f})")
ax.set_yscale("symlog", linthresh=0.01); ax.set_xlabel("GPT-2 small layer (MLP ablated alone)"); ax.set_ylabel("Δ CE loss (nats)")
ax.set_title(f"Weak 75% vs mass-matched strongest activations · base loss {base:.2f}"); ax.legend(); plt.tight_layout(); plt.savefig("plot.png", dpi=130)
