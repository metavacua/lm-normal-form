# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
# Batch 0024: the eight null variants (batches/0014/nulls.py) do what they are said to do, on a synthetic set of float32 tensors with the names of a Llama checkpoint: only the norm weights change; each
# changes by at most two units in the last place of float32; nearly all of them change; the sign patterns of nullr1 .. nullr6 are mixed and differ from one another; the eight variants are pairwise
# different; a variant is the same when made twice. A failure stops the job.
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "0014"))
import nulls  # noqa: E402


def ulp(a):
    return np.spacing(np.abs(a).astype(np.float32))


def main():
    rng = np.random.default_rng(5)
    P = {"model.embed_tokens.weight": rng.standard_normal((50, 16)).astype(np.float32), "lm_head.weight": rng.standard_normal((50, 16)).astype(np.float32),
         "model.norm.weight": (1 + 0.3 * rng.standard_normal(16)).astype(np.float32)}
    for i in range(3):
        P[f"model.layers.{i}.input_layernorm.weight"] = (1 + 0.3 * rng.standard_normal(16)).astype(np.float32)
        P[f"model.layers.{i}.post_attention_layernorm.weight"] = (1 + 0.3 * rng.standard_normal(16)).astype(np.float32)
        P[f"model.layers.{i}.self_attn.q_proj.weight"] = rng.standard_normal((16, 16)).astype(np.float32)
    made = {v: nulls.make_null(P, v) for v in nulls.NULLS}
    assert len(nulls.NULLS) == 8
    for v, R in made.items():
        assert sorted(R) == sorted(P), v
        for k, a in P.items():
            if nulls.is_norm_weight(k):
                step = np.abs(R[k].astype(np.float64) - a.astype(np.float64)) / ulp(a)
                assert step.max() <= 2.0, (v, k, step.max())
                assert (R[k] != a).mean() >= 0.9, (v, k, (R[k] != a).mean())
                assert R[k].dtype == np.float32
            else:
                assert np.array_equal(R[k], a), (v, k)
        again = nulls.make_null(P, v)
        assert all(np.array_equal(again[k], R[k]) for k in P), v
    names = list(made)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            assert any(not np.array_equal(made[names[i]][k], made[names[j]][k]) for k in P if nulls.is_norm_weight(k)), (names[i], names[j])
    for v in names:
        if v.startswith("nullr"):
            d = np.concatenate([(made[v][k].astype(np.float64) - P[k]) for k in P if nulls.is_norm_weight(k)])
            assert (d > 0).any() and (d < 0).any(), v
    assert all(np.all(made["nullm"][k].astype(np.float64) <= P[k].astype(np.float64)) for k in P if nulls.is_norm_weight(k))
    assert all(np.all(made["nullall"][k].astype(np.float64) >= P[k].astype(np.float64)) for k in P if nulls.is_norm_weight(k))
    try:
        nulls.make_null(P, "scale")
    except ValueError:
        pass
    else:
        raise AssertionError("an exact variant must not be accepted as a null")
    print("test_nulls: the eight null variants change only the norm weights, by at most two units in the last place, distinctly and reproducibly")


if __name__ == "__main__":
    main()
