import os, torch
CACHE = os.path.join(os.path.dirname(__file__), "pile_tokens_{}.pt")
def get_tokens(model, n_seq=256, seq_len=128, name="gpt2"):
    p = CACHE.format(name)
    if os.path.exists(p): return torch.load(p)
    from datasets import load_dataset
    ds = load_dataset("NeelNanda/pile-10k", split="train")
    out = []
    for t in ds["text"]:
        ids = model.to_tokens(t[:4000], prepend_bos=True)[0]
        if len(ids) >= seq_len: out.append(ids[:seq_len])
        if len(out) >= n_seq: break
    toks = torch.stack(out); torch.save(toks, p); return toks
def device():
    return "mps" if torch.backends.mps.is_available() else "cpu"
