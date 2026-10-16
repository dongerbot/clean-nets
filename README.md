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
