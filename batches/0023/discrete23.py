# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
# Batch 0023, the discrete part of the group, counted. The Jacobian (cov23.py) sees the continuous part of the group of a model; the permutations and signs are discrete and it does not see
# them. Here every element of three finite families of candidate transformations is applied to a random model of batch 0015's model.py (float64, generic weights, the norm weights random and
# not frozen) and the logits of the transformed model are compared with the original's on several random sequences; the number of candidates that leave the function unchanged is counted. The families
# are written without gauge.py (they are the naive candidates, not the derived transformations), so that the count says which of them are symmetries:
#   hidden   a signed permutation Q of the hidden coordinates applied to the embedding, the head, every matrix that reads the stream (x Q) and every matrix that writes it (Q^T x), with the norm
#            weights either left as they are or permuted along with Q (the unsigned permutation): 4! * 2^4 = 384 candidates each
#   units    for the three units of the gated block: a permutation of the rows of the gate matrix, another of the rows of the up matrix, a third of the columns of the down matrix, and a sign on
#            each unit (up row and down column): 6 * 6 * 6 * 8 = 1728 candidates
#   heads    a permutation of the four query heads (their rows of wq and columns of wo) and one of the two key/value groups (their rows of wk and wv): 24 * 2 = 48 candidates
# Usage: discrete23.py OUT.json
import itertools, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "0015"))
from model import Arch, forward, init_params  # noqa: E402
from ops import FloatOps  # noqa: E402

ARCH = Arch(d=4, n_heads=4, n_kv=2, hd=2, d_ff=3, n_layers=1, vocab=5, tied=False)
SEEDS = [int(x) for x in os.environ.get("LMNF_SEEDS", "31,32,33").split(",")]          # the registered draws; the smoke run sets LMNF_SEEDS to a draw that is not registered
SEQS = 4
T = 4
INVARIANT, VIOLATED = 1e-9, 1e-4          # a candidate is invariant if the largest difference of the logits is at most the first, violated if at least the second; anything between is "unclear" and refutes


def draw(seed):
    ops = FloatOps(eps=1e-6)
    rng = np.random.default_rng(seed)
    P = init_params(ARCH, ops, rng)
    seqs = [[int(t) for t in rng.integers(0, ARCH.vocab, T)] for _ in range(SEQS)]
    return ops, P, seqs


def logits(P, ops, seqs):
    return np.concatenate([forward(ARCH, P, s, ops) for s in seqs], axis=0)


def hidden_candidate(P, perm, signs, permute_norms):
    a = ARCH
    Q = np.zeros((a.d, a.d))
    ab = np.zeros((a.d, a.d))
    for j in range(a.d):
        Q[j, perm[j]] = signs[j]
        ab[j, perm[j]] = 1.0
    R = dict(P)
    R["embed"], R["lm_head"] = P["embed"] @ Q, P["lm_head"] @ Q
    for i in range(a.n_layers):
        for n in ("wq", "wk", "wv", "wg", "wu"):
            R[f"l{i}.{n}"] = P[f"l{i}.{n}"] @ Q
        for n in ("wo", "wd"):
            R[f"l{i}.{n}"] = Q.T @ P[f"l{i}.{n}"]
    if permute_norms:
        for k in ("norm_f",) + tuple(f"l{i}.{n}" for i in range(a.n_layers) for n in ("attn_norm", "mlp_norm")):
            R[k] = P[k] @ ab
    return R


def units_candidate(P, pg, pu, pd, s):
    R = dict(P)
    for i in range(ARCH.n_layers):
        R[f"l{i}.wg"] = P[f"l{i}.wg"][list(pg)]
        R[f"l{i}.wu"] = P[f"l{i}.wu"][list(pu)] * np.array(s)[:, None]
        R[f"l{i}.wd"] = P[f"l{i}.wd"][:, list(pd)] * np.array(s)[None, :]
    return R


def heads_candidate(P, sh, sg):
    a = ARCH
    R = dict(P)
    for i in range(a.n_layers):
        wq = P[f"l{i}.wq"].reshape(a.n_heads, a.hd, a.d)[list(sh)]
        wo = P[f"l{i}.wo"].reshape(a.d, a.n_heads, a.hd)[:, list(sh)]
        wk = P[f"l{i}.wk"].reshape(a.n_kv, a.hd, a.d)[list(sg)]
        wv = P[f"l{i}.wv"].reshape(a.n_kv, a.hd, a.d)[list(sg)]
        R[f"l{i}.wq"], R[f"l{i}.wo"] = wq.reshape(a.n_heads * a.hd, a.d), wo.reshape(a.d, a.n_heads * a.hd)
        R[f"l{i}.wk"], R[f"l{i}.wv"] = wk.reshape(a.n_kv * a.hd, a.d), wv.reshape(a.n_kv * a.hd, a.d)
    return R


def classify(diffs):
    d = np.array(diffs)
    return {"candidates": int(d.size), "invariant": int((d <= INVARIANT).sum()), "violated": int((d >= VIOLATED).sum()), "unclear": int(((d > INVARIANT) & (d < VIOLATED)).sum()),
            "largest_difference_of_the_invariant": float(d[d <= INVARIANT].max()) if (d <= INVARIANT).any() else None, "smallest_difference_of_the_violated": float(d[d >= VIOLATED].min()) if (d >= VIOLATED).any() else None}


def main(out):
    res = {"tolerance_invariant": INVARIANT, "tolerance_violated": VIOLATED, "arch": vars(ARCH), "seeds": SEEDS, "families": {}}
    perms3 = list(itertools.permutations(range(3)))
    for name in ("hidden-norms-fixed", "hidden-norms-permuted", "units", "heads"):
        res["families"][name] = []
    for seed in SEEDS:
        ops, P, seqs = draw(seed)
        base = logits(P, ops, seqs)
        for name, permute in (("hidden-norms-fixed", False), ("hidden-norms-permuted", True)):
            diffs = []
            for perm in itertools.permutations(range(ARCH.d)):
                for signs in itertools.product((1.0, -1.0), repeat=ARCH.d):
                    diffs.append(float(np.max(np.abs(logits(hidden_candidate(P, perm, signs, permute), ops, seqs) - base))))
            r = classify(diffs)
            r["seed"] = seed
            res["families"][name].append(r)
        diffs, coherent_ok = [], True
        for pg, pu, pd in itertools.product(perms3, repeat=3):
            for s in itertools.product((1.0, -1.0), repeat=3):
                d = float(np.max(np.abs(logits(units_candidate(P, pg, pu, pd, s), ops, seqs) - base)))
                diffs.append(d)
                if (pg == pu == pd) != (d <= INVARIANT) and (d <= INVARIANT or d >= VIOLATED):
                    coherent_ok = False
        r = classify(diffs)
        r.update({"seed": seed, "the_invariant_ones_are_exactly_those_with_the_same_permutation_of_the_three_matrices": coherent_ok})
        res["families"]["units"].append(r)
        diffs, group_ok = [], True
        for sh in itertools.permutations(range(ARCH.n_heads)):
            for sg in itertools.permutations(range(ARCH.n_kv)):
                d = float(np.max(np.abs(logits(heads_candidate(P, sh, sg), ops, seqs) - base)))
                diffs.append(d)
                consistent = all(sg[i // ARCH.rep] == sh[i] // ARCH.rep for i in range(ARCH.n_heads))
                if consistent != (d <= INVARIANT) and (d <= INVARIANT or d >= VIOLATED):
                    group_ok = False
        r = classify(diffs)
        r.update({"seed": seed, "the_invariant_ones_are_exactly_those_that_keep_each_head_with_its_group": group_ok})
        res["families"]["heads"].append(r)
        print(f"seed {seed}: hidden fixed norms {res['families']['hidden-norms-fixed'][-1]['invariant']}/384, permuted norms {res['families']['hidden-norms-permuted'][-1]['invariant']}/384, "
              f"units {res['families']['units'][-1]['invariant']}/1728, heads {res['families']['heads'][-1]['invariant']}/48", flush=True)
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
