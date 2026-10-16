"""Exp 01: Do correlated inputs + hidden neurons push a Toy-Models-of-Superposition
model toward shared ("PCA-like"/generalizing) representations instead of arbitrary packing?
Replicates the essay's toy claim (approx. 0.37 / 0.80 / 0.79 share for conditions a/b/c)."""
import json, time, torch, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
torch.set_num_threads(4)
N, G, M, H = 20, 4, 5, 40   # inputs, groups, bottleneck, hidden neurons
GS = N // G
def batch(bs, p):
    g = (torch.rand(bs, G) < p).float().repeat_interleave(GS, 1)
    return g * torch.rand(bs, N)
class Toy(torch.nn.Module):
    def __init__(s, hidden):
        super().__init__(); s.W = torch.nn.Parameter(torch.randn(M, N) * 0.3)
        s.b = torch.nn.Parameter(torch.zeros(N)); s.hidden = hidden
        if hidden:
            s.h = torch.nn.Linear(M, H); s.o = torch.nn.Linear(H, N)
    def forward(s, x):
        z = x @ s.W.T
        if s.hidden: return torch.relu(s.o(torch.relu(s.h(z))))
        return torch.relu(z @ s.W + s.b)
def group_share(W):  # share of W energy in groups' shared directions (top-G data PCs = group means)
    P = torch.zeros(G, N)
    for g in range(G): P[g, g*GS:(g+1)*GS] = 1 / GS**0.5
    return float(((W @ P.T)**2).sum() / (W**2).sum())
def neuron_purity(m):  # for hidden models: fraction of each neuron's input mass from its top group
    E = (m.h.weight @ m.W).detach().abs().reshape(H, G, GS).sum(-1)
    return float((E.max(1).values / E.sum(1)).mean())
res = []
t0 = time.time()
for name, hid, p in [("a_toy_sparse", False, .1), ("b_hidden40_sparse", True, .1), ("c_toy_dense", False, .9)]:
    for seed in range(3):
        torch.manual_seed(seed); m = Toy(hid); opt = torch.optim.Adam(m.parameters(), 3e-3)
        for step in range(6000):
            x = batch(1024, p); loss = ((m(x) - x)**2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        r = dict(cond=name, seed=seed, density=p, loss=float(loss), share=group_share(m.W.detach()))
        if hid: r["purity"] = neuron_purity(m)
        res.append(r); print(r, flush=True)
        if seed == 0:
            Wn = torch.nn.functional.normalize(m.W.detach(), dim=0); res_cos = (Wn.T @ Wn).numpy()
            np.save(f"cos_{name}.npy", res_cos)
# random baseline for share: random W
rs = [group_share(torch.randn(M, N)) for _ in range(200)]
summary = {c: {k: (float(np.mean([r[k] for r in res if r["cond"] == c])), float(np.std([r[k] for r in res if r["cond"] == c])))
               for k in ["share", "loss"] + (["purity"] if c.startswith("b") else [])} for c in dict.fromkeys(r["cond"] for r in res)}
summary["random_W_share"] = (float(np.mean(rs)), float(np.std(rs)))
summary["random_purity_floor"] = 1 / G
summary["essay_reported_share"] = {"a": 0.37, "b": 0.80, "c": 0.79}
summary["runtime_s"] = time.time() - t0
json.dump(dict(runs=res, summary=summary), open("results.json", "w"), indent=1)
print(json.dumps(summary, indent=1))
fig, ax = plt.subplots(1, 4, figsize=(15, 3.8))
conds = ["a_toy_sparse", "b_hidden40_sparse", "c_toy_dense"]
mu = [summary[c]["share"][0] for c in conds]; sd = [summary[c]["share"][1] for c in conds]
ax[0].bar(["a toy\np=.1", "b +40 hidden\np=.1", "c toy\np=.9"], mu, yerr=sd, color=["#888", "#4caf50", "#2196f3"])
ax[0].scatter(range(3), [.37, .80, .79], marker="x", color="red", zorder=3, label="essay")
ax[0].axhline(summary["random_W_share"][0], ls="--", c="k", lw=1, label="random W")
ax[0].set_ylim(0, 1); ax[0].set_title("Bottleneck share in group directions"); ax[0].legend(fontsize=7)
for i, c in enumerate(conds):
    ax[i+1].imshow(np.load(f"cos_{c}.npy"), cmap="RdBu", vmin=-1, vmax=1); ax[i+1].set_title(f"cos(inputs) {c}", fontsize=9)
    ax[i+1].set_xticks([]); ax[i+1].set_yticks([])
plt.tight_layout(); plt.savefig("plot.png", dpi=130)
