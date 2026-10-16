# clean-nets

Testing claims from Nick Cammarata, *Neural Networks Are Already Clean* (https://already-clean.goodfire.com/).
Core claim under test: apparent polysemanticity is mostly *recursive generalization* (neurons grow into related territory), not *superposition* (arbitrary packing of unrelated things).

Setup: `python3 -m venv .venv && .venv/bin/pip install torch numpy matplotlib` (Apple Silicon; exp 01 runs on CPU in ~15 s).

## Experiment log

### 01 — Toy model: correlated inputs + hidden neurons (2026-10-15)
- **Hypothesis (essay, Toy Models section):** in the Toy Models of Superposition setup (20 inputs, 5-d bottleneck), sparse independent-ish reconstruction packs (low share of bottleneck in group directions, essay ≈0.37); adding 40 hidden ReLU neurons after the bottleneck (essay ≈0.80) or making inputs dense (essay ≈0.79) pushes toward shared, group-level ("PCA-like") representations.
- **Method:** 20 inputs in 4 groups of 5 that fire together (group on w.p. p, values U[0,1]); `x -> W (5x20) -> [ReLU(Linear 5->40) -> Linear 40->20] or W^T+b -> ReLU`; MSE, Adam 3e-3, 6000 steps x 1024, 3 seeds. Metric: fraction of ||W||² lying in the 4 group-mean directions (random W ≈ 0.20). Hidden-neuron purity: share of each neuron's effective input (|W_h W|) from its top group (floor 0.25). Script: `experiments/01_toy_correlated/run.py`.
- **Result (mean ± sd, 3 seeds):** (a) toy p=.1: **0.374 ± 0.009** · (b) +40 hidden p=.1: **0.713 ± 0.012** · (c) toy p=.9: **0.793 ± 0.001** · random W 0.20 ± 0.05. Hidden-neuron group purity in (b): **0.58 ± 0.02** (floor 0.25).
- **Interpretation:** (a) and (c) replicate the essay's numbers closely; (b) reproduces the direction (much more group-shared than the original toy) but lower than the essay's 0.80 — my data-generation/architecture details likely differ (essay's exact setup isn't specified). Hidden neurons are group-leaning but far from pure. This supports "the toy packs because nothing downstream reuses shared structure", but it's a toy; it says nothing yet about real LMs. Plot: `experiments/01_toy_correlated/plot.png`.

### 02 — Weak vs strong MLP activation ablation, GPT-2 small (2026-10-15)
- **Hypothesis (essay, "Do Weak Activations Matter?"):** zeroing the weakest 75% of each neuron's activations costs little, while zeroing the strongest activations with the *same total mass* costs a lot.
- **Method:** GPT-2 small (TransformerLens 2.x, MPS), 128 × 128 Pile-10k tokens (16,384). For each MLP neuron (`hook_post`, per batch of 2,048 tokens): *weak* = entries with |a| < neuron's p75; *strong* = the fewest top-|a| entries whose summed |a| equals the weak mass; *random* = random entries, same count as strong. Ablate one layer at a time and all 12 at once; ΔCE vs base 3.543 nats. `experiments/02_weak_strong/`.
- **Result:** All layers: weak **+1.72**, strong **+4.66**, random(same count as strong) **+0.72** nats. Per layer, weak +0.01–0.07 (L0 0.069), strong +0.07–0.26 except **L0 +1.37** and L11 +0.26.
- **Interpretation (mixed):** Asymmetry is real (strong ≈ 2–20× weak per layer, 2.7× overall), supporting "strong activations carry the load". But weak activations are **not free in GPT-2**: removing 75% of entries in every layer costs 1.7 nats, far more than the essay's 72%→71% vision result suggests, and more than removing far fewer random entries. Caveats: GELU negatives get zeroed too; one dataset slice; per-layer costs are small, so the cost mostly compounds across layers.

### 03 — Do neurons grow by local extension? Pythia-70m checkpoints (2026-10-15)
- **Hypothesis (recursive generalization):** contexts a neuron newly acquires between checkpoints are near its earlier top contexts, not arbitrary jumps.
- **Method:** Pythia-70m at steps 1k, 4k, 16k, 64k, 143k; 40 random MLP neurons each in layers 1, 3, 5 (120 total); top-30 contexts over 384 × 128 Pile tokens. For each neuron and checkpoint pair, the *new* top contexts (not in the previous top-30) are compared with the previous top-30 by mean max-cosine in a fixed reference space (final-model L3 residual), against a shuffled baseline (the previous top-30 of another neuron in the same layer). `experiments/03_pythia_growth/`.
- **Result:**

  | pair | same | shuffled | same > shuffled | new contexts |
  |---|---|---|---|---|
  | 1k→4k | 0.366 | 0.274 | 90% | 93% |
  | 4k→16k | 0.418 | 0.267 | 98% | 84% |
  | 16k→64k | 0.427 | 0.259 | 98% | 80% |
  | 64k→143k | 0.464 | 0.258 | 99% | 64% |

- **Interpretation (supportive, with caveats):** new contexts sit much closer to the neuron's own earlier ones than to another neuron's, increasingly so over training. This is consistent with local growth. Caveats: the reference space is the final model (shared surface tokens could drive this), only 5 checkpoints, and "near old contexts" doesn't by itself rule out other explanations; it shows that drift is local.

### 04 — Are negative poles coherent? GPT-2 small (2026-10-15)
- **Hypothesis:** a neuron's most-negative pre-activation contexts form a coherent set (a contrast), not noise.
- **Method:** 150 random neurons (30 each in L2/4/6/8/10), `hook_pre`; top-20 and bottom-20 contexts over 16k tokens; set coherence = mean pairwise cosine of the L6 residual (mean-centered) vs 20 random positions. `experiments/04_negative_poles/`.
- **Result:** coherence: top **0.255**, bottom **0.242**, random **0.115**. Bottom > random for **99.3%** of neurons; bottom > top for 48.7%. Qualitative examples (results.json): L2.2732 + government/term/serve vs − `https://github`/£ price; L4.2459 − "ob" word pieces.
- **Interpretation (supportive, weak):** negative poles are about as coherent as positive poles and far above random. Caveats: extreme sets often share the same token, which inflates coherence trivially; the baseline centering includes BOS outliers; there is no LLM or human judge yet.

### 05 — Toy scale sweep (2026-10-15)
- **Hypothesis (essay):** less capacity → coarser but still *related* neurons, not arbitrary mixtures.
- **Method:** 8 groups × 5 inputs in 4 sibling pairs (hierarchical co-firing), 10-d bottleneck, hidden H ∈ {5…160}, 3 seeds, 4,000 steps. Coarseness = effective # groups per neuron (inverse participation of positive input mass). For multi-group neurons: fraction whose top-2 groups are siblings (chance 1/7 ≈ 0.14). `experiments/05_toy_sweep/`.
- **Result:** effective groups per neuron 3.3 / 5.4 / 5.1 / 4.6 / 4.3 / 4.0 for H = 5 / 10 / 20 / 40 / 80 / 160; sibling fraction 0.00 / 0.20 / 0.18 / 0.26 / 0.18 / 0.19.
- **Interpretation (negative / inconclusive):** neurons get somewhat narrower with more capacity, but in every setting they mix about 4–5 of 8 groups and pair siblings only slightly above chance. This toy did **not** show "coarse but related" neurons. Likely design issues: neurons read a learned 10-d bottleneck with no privileged basis, the positive-mass metric is crude, and the loss is non-monotonic in H (under-trained). A better test would drop the bottleneck or measure function (ablation) rather than weights.
