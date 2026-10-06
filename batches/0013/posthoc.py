# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
# Diagnostics written after the first run of batch 0013, once P8 had failed for TriLM 390M. They are not cells of the batch and carry no
# prediction; they are what was done to find out why. Both read the weights one tensor at a time, so that they run on a small machine.
#   layer0 OUT.txt    TriLM 390M cut to its first layer and the first 4,096 rows of its embedding and head (the columns, which are the hidden
#                     coordinates, are all there): the tensors in which the canonical form of the first run, and the amended one, differ
#                     under each part of the group and under random signs of zeros; the dead planes of the layer; what the signs of zeros
#                     do to the digests of the raw tensors
#   census OUT.json   dead.census of the five models at their pinned revisions
#   toys OUT.txt      the six dead cases of special.selftest_degenerate on four toy Llamas: is the canonical form of the first run, and the amended one,
#                     canonical (the canonical forms of four random images, the sign of zeros included, equal to the model's, bit for bit)?
# The canonical form of the first run is read from git, commit 6886e90 (batches/0013/canon.py as it was when the run was started).
import json, os, subprocess, sys, types
import numpy as np
import ml_dtypes
BF16 = ml_dtypes.bfloat16  # importing ml_dtypes registers bfloat16 with numpy, which reading a bfloat16 file needs
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import forms as F
import symmetry as S
import canon as C
import dead

FIRST_RUN_COMMIT = "6886e90"
MODELS = [("smol-instruct", "HuggingFaceTB/SmolLM2-135M-Instruct", "12fd25f77366fa6b3b4b768ec3050bf629380bac"),
          ("smol-base", "HuggingFaceTB/SmolLM2-135M", "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"),
          ("floatlm-99m", "SpectraSuite/FloatLM_99M", "0516fbe2979fcad7c26805885273562c88c3de90"),
          ("trilm-99m", "SpectraSuite/TriLM_99M_Unpacked", "61cd2c766000fc544595d6570ad1990017d0cb43"),
          ("trilm-390m", "SpectraSuite/TriLM_390M_Unpacked", "79cd1fe5a9650ea5342baa2034ecb1b2aaf792e9")]


def first_run_canon():
    src = subprocess.run(["git", "show", f"{FIRST_RUN_COMMIT}:batches/0013/canon.py"], capture_output=True, text=True, check=True, cwd=HERE).stdout
    mod = types.ModuleType("canon_first_run")
    exec(compile(src, "canon_first_run.py", "exec"), mod.__dict__)
    return mod


def open_model(repo, rev):
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    cfg = json.load(open(hf_hub_download(repo, "config.json", revision=rev)))
    return cfg, safe_open(hf_hub_download(repo, "model.safetensors", revision=rev), framework="np")


def layer0(out):
    old = first_run_canon()
    repo, rev = MODELS[4][1], MODELS[4][2]
    cfg, f = open_model(repo, rev)
    cfg["num_hidden_layers"] = 1
    d = F.Dims(types.SimpleNamespace(**cfg))
    P = {}
    with f:
        P[F.EMBED] = f.get_tensor(F.EMBED).astype(np.float32)[:4096].copy()
        P["lm_head.weight"] = f.get_tensor("lm_head.weight").astype(np.float32)[:4096].copy()
        P["model.norm.weight"] = f.get_tensor("model.norm.weight").astype(np.float32)
        for k in f.keys():
            if k.startswith("model.layers.0."):
                P[k] = f.get_tensor(k).astype(np.float32)
    lines = [f"{repo} at {rev}, cut to layer 0 and the first 4,096 rows of the embedding and the head: {len(P)} tensors, hidden size {d.H}, {d.nh} heads of {d.hd}"]
    nh, hd, H, h2 = d.nh, d.hd, d.H, d.hd // 2
    qb, kb = P[F.key(0, F.Q)].reshape(nh, hd, H), P[F.key(0, F.K)].reshape(nh, hd, H)
    zk = np.array([[not kb[g, p].any() and not kb[g, p + h2].any() for p in range(h2)] for g in range(nh)])
    zq = np.array([[not qb[g, p].any() and not qb[g, p + h2].any() for p in range(h2)] for g in range(nh)])
    lines.append(f"rotary planes of layer 0: {nh * h2}; zero in the key {int(zk.sum())}, zero in the query {int(zq.sum())}, zero in both {int((zk & zq).sum())}, "
                 f"zero in one only {int((zk ^ zq).sum())}")
    for name, mod in (("canonical form of the first run", old), ("amended canonical form", C)):
        base = mod.canonical(P, d)
        per = {}
        for part in ("qk", "vo", "units-signed", "units", "heads", "residual", "zsigns"):
            per[part] = [sorted(k.split("layers.0.")[-1] for k in base if mod.canonical(S.act(P, d, np.random.default_rng(113 + t), False, (part,)), d)[k].tobytes() != base[k].tobytes())
                         for t in range(3)]
        whole = [sorted(k.split("layers.0.")[-1] for k in base if mod.canonical(S.act(P, d, np.random.default_rng(300 + t), False, ("residual", "units", "heads", "vo", "qk", "zsigns")), d)[k].tobytes() != base[k].tobytes())
                 for t in range(3)]
        lines.append(f"{name}: canonical tensors that differ from the model's canonical form, for 3 random elements of each part")
        for part, v in per.items():
            lines.append(f"    {part:13s} {[len(x) for x in v]}" + (f"  e.g. {v[0]}" if v[0] else ""))
        lines.append(f"    {'whole group and the sign of zeros':13s} {[len(x) for x in whole]}")
    img = S.act(P, d, np.random.default_rng(1), False, ("zsigns",))
    lines.append(f"raw tensors whose bytes change when only the signs of zeros are changed: {sum(1 for k in P if img[k].tobytes() != P[k].tobytes())} of {len(P)}; "
                 f"every one of the {len(P)} tensors with its values unchanged: {all(np.array_equal(img[k], P[k]) for k in P)}")
    text = "\n".join(lines)
    print(text)
    open(out, "w").write(text + "\n")


def census(out):
    res = {}
    for tag, repo, rev in MODELS:
        cfg, f = open_model(repo, rev)
        d = F.Dims(types.SimpleNamespace(**cfg))
        with f:
            res[tag] = dead.census(lambda k: f.get_tensor(k).astype(np.float32), d, "lm_head.weight" in f.keys())
        print(tag, json.dumps(res[tag]["totals"]), json.dumps(res[tag]["embedding"]))
    json.dump(res, open(out, "w"), indent=1, sort_keys=True)


def toys(out):
    sys.argv = sys.argv[:1]
    import special
    old = first_run_canon()
    same = lambda A, B: set(A) == set(B) and all(A[k].tobytes() == B[k].tobytes() for k in A)
    lines = ["for each toy and kind of dead structure: the first run's canonical form / the amended one (canonical, NOT canonical, or the exception it raises)"]
    for tied in (True, False):
        for nkv in (3, 6):
            model, d, P = special.toy(tied, nkv)
            row = []
            for what in ("qk-both", "vo-both", "units-both", "units-mixed", "qk-mixed", "vo-mixed"):
                Pd = special.kill(P, d, what)
                res = []
                for mod in (old, C):
                    try:
                        base = mod.canonical(Pd, d)
                        ok = all(same(base, mod.canonical(S.act(Pd, d, np.random.default_rng(t), tied, special.EVERY), d)) for t in range(4))
                        res.append("canonical" if ok else "NOT canonical")
                    except Exception as e:
                        res.append(type(e).__name__)
                row.append(f"{what}: {res[0]} / {res[1]}")
            lines.append(f"[{'tied' if tied else 'untied'} head, {d.rep} query head{'s' if d.rep > 1 else ''} per key/value head] " + " | ".join(row))
    text = "\n".join(lines)
    print(text)
    open(out, "w").write(text + "\n")


if __name__ == "__main__":
    {"layer0": layer0, "census": census, "toys": toys}[sys.argv[1]](sys.argv[2])
