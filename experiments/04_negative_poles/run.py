"""Exp 04: are negative poles coherent? GPT-2 small MLP pre-activations.
For random neurons (layers 2-10), take top-K and bottom-K pre-activation contexts and measure set coherence
(mean pairwise cosine of a fixed reference rep: layer-6 resid_post, mean-centered) vs random-K position sets.
Also dumps text examples for a qualitative read."""
import sys, json, time, torch, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, "../common"); from data import get_tokens, device
from transformer_lens import HookedTransformer
torch.manual_seed(0); dev = device(); t0 = time.time()
m = HookedTransformer.from_pretrained("gpt2", device=dev)
toks = get_tokens(m).to(dev); L = toks.shape[1]
LAYERS = [2, 4, 6, 8, 10]; NPER = 30; K = 20
names = [f"blocks.{l}.mlp.hook_pre" for l in LAYERS] + ["blocks.6.hook_resid_post"]
acc = {n: [] for n in names}; neurons = {l: torch.randperm(3072)[:NPER] for l in LAYERS}
with torch.no_grad():
    for i in range(0, len(toks), 32):
        _, c = m.run_with_cache(toks[i:i+32], names_filter=lambda n: n in names)
        for l in LAYERS: acc[f"blocks.{l}.mlp.hook_pre"].append(c[f"blocks.{l}.mlp.hook_pre"][:, :, neurons[l]].reshape(-1, NPER).float().cpu())
        acc["blocks.6.hook_resid_post"].append(c["blocks.6.hook_resid_post"].reshape(-1, 768).float().cpu())
R = torch.cat(acc["blocks.6.hook_resid_post"]); R = torch.nn.functional.normalize(R - R.mean(0), dim=-1)
valid = (torch.arange(len(R)) % L) != 0
def coh(idx): S = R[idx] @ R[idx].T; n = len(idx); return float((S.sum() - S.diag().sum()) / (n * (n - 1)))
flat = toks.cpu().reshape(-1)
def ctx(i): s = max(i - 8, (i // L) * L); return m.to_string(flat[s:i]) + " ⟦" + m.to_string(flat[i:i+1]) + "⟧"
res, examples = {"top": [], "bottom": [], "random": []}, []
for l in LAYERS:
    A = torch.cat(acc[f"blocks.{l}.mlp.hook_pre"]); A[~valid] = float("nan")
    for j in range(NPER):
        a = A[:, j].nan_to_num(0.)
        a_top = a.clone(); a_top[~valid] = -1e9; a_bot = a.clone(); a_bot[~valid] = 1e9
        top = a_top.topk(K).indices; bot = (-a_bot).topk(K).indices
        rnd = torch.nonzero(valid).squeeze(1)[torch.randperm(int(valid.sum()))[:K]]
        res["top"].append(coh(top)); res["bottom"].append(coh(bot)); res["random"].append(coh(rnd))
        if j < 2: examples.append({"neuron": f"L{l}.{int(neurons[l][j])}", "top": [ctx(int(i)) for i in top[:5]], "bottom": [ctx(int(i)) for i in bot[:5]]})
T, Bo, Ra = map(np.array, (res["top"], res["bottom"], res["random"]))
summ = {"top_mean": T.mean(), "bottom_mean": Bo.mean(), "random_mean": Ra.mean(),
        "bottom_gt_random_frac": float((Bo > Ra).mean()), "top_gt_random_frac": float((T > Ra).mean()),
        "bottom_gt_top_frac": float((Bo > T).mean()), "n_neurons": len(T), "runtime_s": time.time() - t0,
        "per_layer_bottom": {l: float(Bo[i*NPER:(i+1)*NPER].mean()) for i, l in enumerate(LAYERS)},
        "per_layer_top": {l: float(T[i*NPER:(i+1)*NPER].mean()) for i, l in enumerate(LAYERS)}}
summ = {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in summ.items()}
json.dump({"summary": summ, "examples": examples}, open("results.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps(summ, indent=1))
fig, ax = plt.subplots(figsize=(7, 4)); bins = np.linspace(min(Ra.min(), 0), max(T.max(), Bo.max()), 30)
for v, c, n in [(Ra, "#888", "random positions"), (T, "#4caf50", "top-20 (positive pole)"), (Bo, "#e53935", "bottom-20 (negative pole)")]:
    ax.hist(v, bins, alpha=.55, color=c, label=f"{n}: mean {v.mean():.2f}")
ax.set_xlabel("set coherence (mean pairwise cosine, L6 resid)"); ax.set_ylabel("# neurons")
ax.set_title("GPT-2 small: are negative poles coherent?"); ax.legend(fontsize=8); plt.tight_layout(); plt.savefig("plot.png", dpi=130)
