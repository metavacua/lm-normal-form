import numpy as np
from huggingface_hub import hf_hub_download
from safetensors import safe_open
p = hf_hub_download("SpectraSuite/TriLM_99M_Unpacked", "model.safetensors", revision="61cd2c766000fc544595d6570ad1990017d0cb43")
with safe_open(p, framework="numpy") as f:
    names = [k for k in f.keys() if "layers.15.self_attn.v_proj" in k]
    print(names)
    W = f.get_tensor(names[0]).astype(np.float64)
print(W.shape, "distinct values:", np.unique(W).size)
n = np.linalg.norm(W, axis=1)
print("zero rows:", int((n == 0).sum()), "min/max row norm:", n.min(), n.max())
Wn = W / np.where(n[:, None] > 0, n[:, None], 1)
G = Wn @ Wn.T
np.fill_diagonal(G, 0)
iu = np.argwhere(np.abs(G) > 1 - 1e-9)
iu = [(i, j, G[i, j]) for i, j in iu if i < j]
print("pairs of rows with |cos| = 1:", len(iu), iu[:12])
# the same for every layer
f2 = safe_open(p, framework="numpy")
for l in range(16):
    W = f2.get_tensor(f"model.layers.{l}.self_attn.v_proj.weight").astype(np.float64)
    n = np.linalg.norm(W, axis=1)
    Wn = W / np.where(n[:, None] > 0, n[:, None], 1)
    G = np.abs(Wn @ Wn.T); np.fill_diagonal(G, 0)
    print(l, "zero rows", int((n == 0).sum()), "pairs |cos|>1-1e-9:", int((G > 1 - 1e-9).sum() // 2))
