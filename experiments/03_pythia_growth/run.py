"""Exp 03: do neurons grow by local extension? Pythia-70m checkpoints.
For random MLP neurons, compare top-k contexts at checkpoint t that were NOT in its top-k at t-1
against its own t-1 top-k (same neuron) vs another neuron's t-1 top-k (shuffled baseline).
Similarity uses a fixed reference representation: final-checkpoint layer-3 residual at that position."""
import sys, json, time, torch, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, "../common"); from data import get_tokens, device
from transformer_lens import HookedTransformer
torch.manual_seed(0); dev = device()
STEPS = [1000, 4000, 16000, 64000, 143000]; LAYERS = [1, 3, 5]; NPER = 40; K = 30
t0 = time.time()
ref = HookedTransformer.from_pretrained("pythia-70m", device=dev)
toks = get_tokens(ref, n_seq=384, name="pythia").to(dev)
def run_cache(model, names):
    out = {n: [] for n in names}
    with torch.no_grad():
        for i in range(0, len(toks), 32):
            _, c = model.run_with_cache(toks[i:i+32], names_filter=lambda n: n in names)
            for n in names: out[n].append(c[n].reshape(-1, c[n].shape[-1]).float().cpu())
    return {n: torch.cat(v) for n, v in out.items()}
R = run_cache(ref, ["blocks.3.hook_resid_post"])["blocks.3.hook_resid_post"]
R = R - R.mean(0); R = torch.nn.functional.normalize(R, dim=-1)
R[torch.arange(len(R)) % toks.shape[1] == 0] = 0  # ignore BOS positions
neurons = {l: torch.randperm(2048)[:NPER] for l in LAYERS}
tops = {}
for s in STEPS:
    m = ref if s == 143000 else HookedTransformer.from_pretrained("pythia-70m", checkpoint_value=s, device=dev)
    A = run_cache(m, [f"blocks.{l}.mlp.hook_post" for l in LAYERS])
    for l in LAYERS:
        a = A[f"blocks.{l}.mlp.hook_post"][:, neurons[l]]
        a[torch.arange(len(a)) % toks.shape[1] == 0] = -1e9
        tops[(s, l)] = a.topk(K, dim=0).indices.T  # [NPER, K]
    if m is not ref: del m
    print("step", s, round(time.time() - t0), flush=True)
def sim(new_idx, old_idx):  # mean over new contexts of max cosine to old set
    return float((R[new_idx] @ R[old_idx].T).max(1).values.mean())
res = {"same": [], "shuffled": [], "frac_new": [], "pairs": []}
for a, b in zip(STEPS[:-1], STEPS[1:]):
    same, shuf, fn = [], [], []
    for l in LAYERS:
        for j in range(NPER):
            old, cur = tops[(a, l)][j], tops[(b, l)][j]
            new = cur[~torch.isin(cur, old)]
            fn.append(len(new) / K)
            if len(new) == 0: continue
            o = (j + 1 + torch.randint(NPER - 1, ())) % NPER
            same.append(sim(new, old)); shuf.append(sim(new, tops[(a, l)][o]))
    res["pairs"].append(f"{a}->{b}"); res["same"].append(same); res["shuffled"].append(shuf); res["frac_new"].append(float(np.mean(fn)))
    print(a, b, "same", np.mean(same), "shuf", np.mean(shuf), "frac_new", np.mean(fn), "win", np.mean(np.array(same) > np.array(shuf)), flush=True)
summ = [{"pair": p, "same": float(np.mean(s)), "shuffled": float(np.mean(h)), "win_rate": float(np.mean(np.array(s) > np.array(h))), "n": len(s), "frac_new": f}
        for p, s, h, f in zip(res["pairs"], res["same"], res["shuffled"], res["frac_new"])]
json.dump({"summary": summ, "runtime_s": time.time() - t0, "layers": LAYERS, "neurons_per_layer": NPER, "K": K}, open("results.json", "w"), indent=1)
fig, ax = plt.subplots(figsize=(7, 4)); x = np.arange(len(summ))
ax.bar(x - .2, [d["same"] for d in summ], .4, label="same neuron (t-1 top-k)", color="#4caf50")
ax.bar(x + .2, [d["shuffled"] for d in summ], .4, label="other neuron (shuffled)", color="#888")
for i, d in enumerate(summ): ax.text(i, max(d["same"], d["shuffled"]) + .01, f"win {d['win_rate']:.0%}\nnew {d['frac_new']:.0%}", ha="center", fontsize=8)
ax.set_xticks(x, [d["pair"] for d in summ], fontsize=8); ax.set_ylabel("mean max-cosine of NEW top contexts")
ax.set_title("Pythia-70m: are newly acquired contexts near old ones?"); ax.legend(fontsize=8); plt.tight_layout(); plt.savefig("plot.png", dpi=130)
