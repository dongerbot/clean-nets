"""Exp 05: toy scale sweep. Exp 01 hidden-neuron model with H in {5..160}, more groups (8 groups x 5 = 40 inputs).
Does shrinking capacity make neurons coarser (cover more groups but still coherent) or arbitrary?
Coarseness = effective #groups per neuron (inverse participation of group mass). Arbitrariness test:
for neurons covering >=2 groups, are the groups they combine similar? Here groups are made hierarchically
related (4 supergroups of 2 groups each, sharing a common component), so 'coarse generalization' should merge
sibling groups, while 'arbitrary packing' merges siblings at chance (1/7)."""
import json, time, torch, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
torch.set_num_threads(4)
G, GS, M, SG = 8, 5, 10, 4; N = G * GS
def batch(bs, p=.1):
    sg = (torch.rand(bs, SG) < p).float()                       # supergroup on
    sub = (torch.rand(bs, G) < .5).float() * sg.repeat_interleave(2, 1)  # each child on w.p. .5 given parent
    sub = torch.maximum(sub, sg.repeat_interleave(2, 1) * (torch.rand(bs, G) < .3).float())
    return sub.repeat_interleave(GS, 1) * torch.rand(bs, N)
class Toy(torch.nn.Module):
    def __init__(s, H):
        super().__init__(); s.W = torch.nn.Parameter(torch.randn(M, N) * .3); s.h = torch.nn.Linear(M, H); s.o = torch.nn.Linear(H, N)
    def forward(s, x): return torch.relu(s.o(torch.relu(s.h(x @ s.W.T))))
rows = []; t0 = time.time()
for H in [5, 10, 20, 40, 80, 160]:
    for seed in range(3):
        torch.manual_seed(seed); m = Toy(H); opt = torch.optim.Adam(m.parameters(), 3e-3)
        for _ in range(4000):
            x = batch(1024); loss = ((m(x) - x)**2).mean(); opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            E = (m.h.weight @ m.W).clamp(min=0).reshape(H, G, GS).sum(-1)  # positive input mass per group
            alive = E.sum(1) > 1e-3 * E.sum(1).max(); E = E[alive]
            P = E / E.sum(1, keepdim=True); eff = 1 / (P**2).sum(1)
            top2 = P.topk(2, 1).indices; sib = (top2[:, 0] // 2 == top2[:, 1] // 2).float()
            multi = eff >= 1.5
        rows.append(dict(H=H, seed=seed, loss=float(loss), eff_groups=float(eff.mean()), alive=int(alive.sum()),
                         sibling_frac=float(sib[multi].mean()) if multi.any() else float("nan"), n_multi=int(multi.sum())))
        print(rows[-1], flush=True)
Hs = sorted(set(r["H"] for r in rows)); agg = lambda k: [np.nanmean([r[k] for r in rows if r["H"] == h]) for h in Hs]
summ = dict(H=Hs, eff_groups=agg("eff_groups"), sibling_frac=agg("sibling_frac"), loss=agg("loss"), chance_sibling=1/7, runtime_s=time.time()-t0)
json.dump(dict(runs=rows, summary=summ), open("results.json", "w"), indent=1); print(json.dumps(summ, indent=1))
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
ax[0].plot(Hs, summ["eff_groups"], "o-"); ax[0].set_xscale("log"); ax[0].set_xlabel("hidden neurons H"); ax[0].set_ylabel("effective groups per neuron"); ax[0].set_title("Coarseness vs capacity")
ax[1].plot(Hs, summ["sibling_frac"], "o-", c="#4caf50", label="observed"); ax[1].axhline(1/7, ls="--", c="k", label="arbitrary (1/7)")
ax[1].set_xscale("log"); ax[1].set_ylim(0, 1); ax[1].set_xlabel("hidden neurons H"); ax[1].set_ylabel("top-2 groups are siblings"); ax[1].set_title("Multi-group neurons: related or arbitrary?"); ax[1].legend()
plt.tight_layout(); plt.savefig("plot.png", dpi=130)
