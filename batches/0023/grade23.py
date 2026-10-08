#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
# Grades batch 0023. The claims below, with their thresholds, were committed before any run of the registered experiments (docs/batches/0023.md has the table that `--table` prints from this file); the
# experiments write measurements and no verdict, and this script reads the measurements and applies the thresholds. A claim has one of three statuses: "as predicted", "REFUTED", "not run" (a file or a
# number that the claim needs is missing, or a validity condition of the claim failed: nothing is concluded). Controls (checks that the experiment can fail, and that the harness works) are listed apart
# and are never counted among the claims; the registered test of "is a language model" is listed apart as an inclusion test of the checkpoints. A number that is not a number (NaN) never satisfies a threshold.
# The exit status is 0 only if every claim is "as predicted", every control holds and every checkpoint passes the inclusion test.
#   grade23.py RESULTS_DIR OUT_DIR       grade
#   grade23.py --table                   print the claims as a Markdown table (for the registration)
#   grade23.py --selftest                every claim holds on a fixture built to satisfy it, becomes REFUTED when any one of its own inputs is made to violate it, and is "not run" on no data
import copy, json, math, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


class NotRun(Exception):
    pass


def resolve(tree, path):
    """The values at `path` (tokens separated by '/'; '*' stands for every element of a dict or a list)."""
    cur = [tree]
    for tok in path.split("/"):
        nxt = []
        for node in cur:
            if tok == "*":
                if isinstance(node, dict):
                    nxt.extend(node.values())
                elif isinstance(node, list):
                    nxt.extend(node)
                else:
                    raise NotRun(f"{path}: '*' on something that is not a container")
            elif isinstance(node, dict) and tok in node:
                nxt.append(node[tok])
            elif isinstance(node, list) and tok.isdigit() and int(tok) < len(node):
                nxt.append(node[int(tok)])
            else:
                raise NotRun(f"{path}: nothing at '{tok}'")
        cur = nxt
    if not cur:
        raise NotRun(f"{path}: no values")
    for v in cur:
        if v is None:
            raise NotRun(f"{path}: a value is null")
    return cur


def put(tree, path, value):
    """Write `value` at `path` into a fixture ('*' becomes the key 'k0')."""
    node = tree
    toks = path.split("/")
    for i, tok in enumerate(toks):
        tok = "k0" if tok == "*" else tok
        if i == len(toks) - 1:
            if isinstance(value, dict) and isinstance(node.get(tok), dict):
                node[tok].update(value)
            else:
                node[tok] = value
        else:
            node = node.setdefault(tok, {})


def fmt(v):
    return f"{v:.3g}" if num(v) else repr(v)


def num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and not math.isnan(v)


class Pred:
    paths = ()


class Le(Pred):
    def __init__(self, path, thr):
        self.path, self.thr, self.paths = path, thr, (path,)

    def check(self, t):
        vals = resolve(t, self.path)
        worst = max(vals, key=lambda v: (not num(v), v if num(v) else 0))
        return all(num(v) and v <= self.thr for v in vals), f"{self.path}: largest {fmt(worst)}, at most {self.thr:g} was predicted"

    def fixture(self, t, ok):
        put(t, self.path, self.thr / 10 if ok else self.thr * 10 + 1)

    def text(self):
        return f"{self.path} ≤ {self.thr:g}"


class Ge(Pred):
    def __init__(self, path, thr):
        self.path, self.thr, self.paths = path, thr, (path,)

    def check(self, t):
        vals = resolve(t, self.path)
        worst = min(vals, key=lambda v: (not num(v), v if num(v) else 0))
        return all(num(v) and v >= self.thr for v in vals), f"{self.path}: smallest {fmt(worst)}, at least {self.thr:g} was predicted"

    def fixture(self, t, ok):
        put(t, self.path, self.thr * 10 + 1 if ok else self.thr / 10)

    def text(self):
        return f"{self.path} ≥ {self.thr:g}"


class Eq(Pred):
    def __init__(self, path, val):
        self.path, self.val, self.paths = path, val, (path,)

    def check(self, t):
        vals = resolve(t, self.path)
        return all(v == self.val and type(v) is type(self.val) for v in vals), f"{self.path}: {vals[:4]}, {self.val!r} was predicted"

    def fixture(self, t, ok):
        bad = (not self.val) if isinstance(self.val, bool) else (self.val + 1 if num(self.val) else str(self.val) + "x")
        put(t, self.path, self.val if ok else bad)

    def text(self):
        return f"{self.path} = {self.val!r}"


class Gt(Pred):
    def __init__(self, a, b):
        self.a, self.b, self.paths = a, b, (a, b)

    def check(self, t):
        va, vb = resolve(t, self.a), resolve(t, self.b)
        return all(num(x) and num(y) and x > y for x, y in zip(va, vb)), f"{self.a} = {va[:3]} against {self.b} = {vb[:3]}: the first was predicted to be larger"

    def fixture(self, t, ok):
        put(t, self.a, 10.0 if ok else 1.0)
        put(t, self.b, 1.0 if ok else 10.0)

    def text(self):
        return f"{self.a} > {self.b}"


class Gap(Pred):
    """The rank is sharp: in every row of a list, the singular value at the cut is at least `ratio` times the one after it."""
    def __init__(self, rows_path, ratio=1e6):
        self.path, self.ratio, self.paths = rows_path, ratio, (rows_path,)

    def check(self, t):
        rows = resolve(t, self.path)
        try:
            gaps = [r["sv_at_cut"] / r["sv_after_cut"] if r["sv_after_cut"] > 0 else 1e300 for r in rows]
        except (KeyError, TypeError) as e:
            raise NotRun(f"{self.path}: rows without sv_at_cut / sv_after_cut ({e})")
        return all(num(x) and x >= self.ratio for x in gaps), f"{self.path}: smallest gap {fmt(min(gaps))}, at least {self.ratio:g} was required"

    def fixture(self, t, ok):
        put(t, self.path, {"sv_at_cut": 1.0, "sv_after_cut": 1e-12 if ok else 0.5})

    def text(self):
        return f"{self.path}: sv_at_cut / sv_after_cut ≥ {self.ratio:g}"


class Claim:
    def __init__(self, cid, group, severity, subject, text, preds, valid=(), files=()):
        self.id, self.group, self.severity, self.subject, self.text = cid, group, severity, subject, text
        self.preds, self.valid = list(preds), list(valid)

    def grade(self, t):
        try:
            for v in self.valid:
                ok, d = v.check(t)
                if not ok:
                    return "not run", "validity condition failed: " + d
            details, refuted = [], False
            for p in self.preds:
                ok, d = p.check(t)
                refuted = refuted or not ok
                details.append(("ok: " if ok else "FAILS: ") + d)
            return ("REFUTED" if refuted else "as predicted"), "; ".join(details)
        except NotRun as e:
            return "not run", str(e)


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# the registry
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
FINITE = "the Llama decoder equations at hidden size 4, 2 layers, 2 heads in one key/value group of head dimension 2, 5 symbols, seeded random parameters (the finite instance; not a language model)"
LLAMA = [("smol-instruct", "HuggingFaceTB/SmolLM2-135M-Instruct @12fd25f"), ("smol-base", "HuggingFaceTB/SmolLM2-135M @93efa2f"), ("floatlm-99m", "SpectraSuite/FloatLM_99M @0516fbe"),
         ("trilm-99m", "SpectraSuite/TriLM_99M_Unpacked @61cd2c7"), ("delphi-100k", "delphi-suite/v0-llama2-100k @c1372fb")]
ORTH = ["unit permutation", "signed permutation of the hidden coordinates", "head swap", "rotation of the stream"]
NONORTH = ["unit scaling by +-2^k", "invertible matrix on the value dimensions", "complex scalar on the rotary plane"]


def claims():
    C = []
    add = lambda *a, **k: C.append(Claim(*a, **k))
    # --- the dimension of the group of the Llama equations, and what changes it ---------------------------------------------------------------------------------------------------------
    add("R1", "dimension", "high", FINITE, "the rank deficiency of the Jacobian of the function is 26 (2 layers x (4 value/output + 2 query/key + 4 unit) + 6 rotations), at each of five draws, with a singular-value gap of at least 1e6",
        [Eq("cov/rank/variants/base/*/deficiency", 26), Gap("cov/rank/variants/base/*")])
    add("R2a", "dimension", "high", FINITE, "with the frequencies of the rotary planes as parameters the deficiency is still 26 (no new symmetry)", [Eq("cov/rank/variants/theta/*/deficiency", 26), Gap("cov/rank/variants/theta/*")])
    add("R2b", "dimension", "high", FINITE, "with the position of token t the proper time sum_{i<t} exp(b + w.x_i) (b, w parameters) and the frequencies fixed the deficiency is still 26", [Eq("cov/rank/variants/delta/*/deficiency", 26), Gap("cov/rank/variants/delta/*")])
    add("R2c", "dimension", "high", FINITE, "with both the frequencies and the proper time as parameters the deficiency is 28: one dilation (b -> b + s, theta -> theta e^-s) per layer", [Eq("cov/rank/variants/theta+delta/*/deficiency", 28), Gap("cov/rank/variants/theta+delta/*")])
    add("R3", "dimension", "moderate", FINITE, "with its own projection of the queries and the keys at every position (no rotary embedding) the deficiency is 62: 2 x (4 + 4 + 4) + 6 = 30 gauge dimensions and 2 x 16 for the queries' projection at position 0, which the softmax over one position never reads",
        [Eq("cov/rank/variants/posdep/*/deficiency", 62), Gap("cov/rank/variants/posdep/*")])
    add("R4", "dimension", "moderate", FINITE, "at the point of the position-dependent model that is the image of a point of the rotary model (W_t = R_t W): the two functions are equal (to 1e-12) and the fibre of the position-dependent model has dimension 62",
        [Le("cov/rank/constrained/*/function_difference", 1e-12), Eq("cov/rank/constrained/*/deficiency_covariant", 62)])
    add("R5a", "dimension", "low", FINITE + " with other hidden sizes and vocabularies", "when the vocabulary is at most the hidden size ((d, V) = (4, 3), (6, 6)) the deficiency exceeds the formula (2 x 10 + d(d-1)/2) by at least 1: the first layer sees at most d independent vectors (a post hoc rule; the four configurations of R5a and R5b had been run once, at draw 99, before registration, so this tests the rule on fresh draws and not on fresh configurations)",
        [Ge("cov/vocab/V_at_most_d/*/extra", 1)])
    add("R5b", "dimension", "low", FINITE + " with other hidden sizes and vocabularies", "when the vocabulary exceeds the hidden size ((d, V) = (5, 6), (4, 8)) the deficiency equals the formula (the same four configurations as R5a)", [Eq("cov/vocab/V_larger_than_d/*/extra", 0)])
    add("R6", "causal structure", "low", FINITE, "the stream at position t' after one layer, two layers, and the stack, depends on the input stream at position t if and only if t <= t': at most 1e-14 for t > t', at least 1e-8 for t <= t'",
        [Le("cov/causal/*/largest_dependence_on_a_later_position", 1e-14), Ge("cov/causal/*/smallest_dependence_on_an_earlier_or_equal_position", 1e-8)])
    # --- a training point -----------------------------------------------------------------------------------------------------------------------------------------------------------------
    add("F1", "dimension", "moderate", FINITE + ", trained by Adam (300 steps) toward a target it cannot realize", "at the point that training reaches the deficiency is still 26, with a gap of at least 1e6, at each of five seeds: training creates no redundancy that is not a symmetry",
        [Eq("fibre/runs/*/trained/deficiency", 26), Gap("fibre/runs/*/trained")], valid=[Gt("fibre/runs/*/loss_first", "fibre/runs/*/loss_last")])
    # --- the discrete part ---------------------------------------------------------------------------------------------------------------------------------------------------------------
    D = "the Llama decoder equations (hidden 4, 1 layer, 4 heads in 2 groups, 3 units, 5 symbols), random parameters, norm weights random"
    add("D1a", "discrete symmetries", "moderate", D, "of the 384 signed permutations of the hidden coordinates applied to the embedding, the head and every reader and writer with the norm weights left as they are, exactly 16 (the sign changes of the identity permutation) leave the logits unchanged and none is unclear, at each of three draws",
        [Eq("discrete/families/hidden-norms-fixed/*/invariant", 16), Eq("discrete/families/hidden-norms-fixed/*/unclear", 0)])
    add("D1b", "discrete symmetries", "moderate", D, "with the norm weights permuted along, all 384 leave the logits unchanged", [Eq("discrete/families/hidden-norms-permuted/*/invariant", 384), Eq("discrete/families/hidden-norms-permuted/*/unclear", 0)])
    add("D2", "discrete symmetries", "moderate", D, "of the 1728 triples of permutations of the gate rows, the up rows and the down columns of three units with a sign on each unit, exactly 48 leave the logits unchanged, they are those that apply the same permutation to the three matrices",
        [Eq("discrete/families/units/*/invariant", 48), Eq("discrete/families/units/*/unclear", 0), Eq("discrete/families/units/*/the_invariant_ones_are_exactly_those_with_the_same_permutation_of_the_three_matrices", True)])
    add("D3", "discrete symmetries", "moderate", D, "of the 48 pairs of a permutation of the 4 query heads and one of the 2 key/value groups, exactly 8 leave the logits unchanged, they are those that keep each head with its group",
        [Eq("discrete/families/heads/*/invariant", 8), Eq("discrete/families/heads/*/unclear", 0), Eq("discrete/families/heads/*/the_invariant_ones_are_exactly_those_that_keep_each_head_with_its_group", True)])
    # --- the optimizers -------------------------------------------------------------------------------------------------------------------------------------------------------------------
    O = "training rules applied to the finite instance (distillation of a random teacher on all 780 contexts)"
    fl = lambda rule, name, k: f"opt-field/rules/{rule}/{name}/{k}"
    fvalid = [Le("opt-field/functions_unchanged_largest_difference", 1e-12)]
    add("O1", "optimizers", "high", O, "the velocity that gradient descent (the Euclidean metric of the constants) gives the predictive distributions, at ten random points, is the same at theta and at g.theta (relative difference of the centered scores at most 1e-8) for the four equivalence transformations that are orthogonal on the constants, and differs (at least 1e-6) for the three that are not",
        [Le(fl("sgd", n, "largest"), 1e-8) for n in ORTH] + [Ge(fl("sgd", n, "smallest"), 1e-6) for n in NONORTH], valid=fvalid)
    add("O2", "optimizers", "high", O, "the velocity of the first step of Adam: the same (1e-8) for the signed permutation, the unit permutation and the head swap; different (1e-6) for the rotation and the three that are not orthogonal",
        [Le(fl("adam", n, "largest"), 1e-8) for n in ORTH[:3]] + [Ge(fl("adam", n, "smallest"), 1e-6) for n in [ORTH[3]] + NONORTH], valid=fvalid)
    add("O3", "optimizers", "high", O, "the velocity of natural gradient (the pseudo-inverse of the Fisher metric of the predictive distributions: F^+ grad L, singular values of its square root below 1e-10 of the largest taken as zero) is the same at theta and at g.theta (1e-8) for all seven transformations, orthogonal or not: the rule defines a dynamics on functions",
        [Le(fl("ngd-pinv", n, "largest"), 1e-8) for n in ORTH + NONORTH], valid=fvalid)
    add("O4", "optimizers", "high", O, "the same rule with the damping that adds a fixed multiple of the identity of the constants, (F + 1e-9 I)^-1 grad L, is the same (1e-8) for the four orthogonal transformations and different (1e-6) for the three that are not: the metric of the constants is the absolute object",
        [Le(fl("ngd-euclid", n, "largest"), 1e-8) for n in ORTH] + [Ge(fl("ngd-euclid", n, "smallest"), 1e-6) for n in NONORTH], valid=fvalid)
    add("O5", "optimizers", "high", O, "the damping by the Euclidean metric of the scores (F + lambda J^T J)^-1 grad L, a metric on functions, with lambda 1e-6 and 1e-2 times the mean nonzero eigenvalue of the Fisher blocks, is the same (1e-8) for all seven",
        [Le(fl(r, n, "largest"), 1e-8) for r in ("ngd-covdamp:1e-6", "ngd-covdamp:1e-2") for n in ORTH + NONORTH], valid=fvalid)
    add("O6", "optimizers", "moderate", O, "the covariant step is large in the constants: the median over the ten points of |v| for natural gradient over |v| for gradient descent is at least 100 (the norm is the Euclidean norm of the constants, which is not a covariant quantity, and the ratio depends on the cutoff of the pseudo-inverse, 1e-10 here)",
        [Ge("opt-field/median_step_norm_over_the_step_norm_of_sgd/ngd-pinv", 100.0)], valid=fvalid)
    g_ = lambda cfg, name: f"{cfg}/gauges/{name}/max_difference"
    sane = lambda cfg: [Gt(f"{cfg}/loss_first", f"{cfg}/loss_last")]
    add("O7", "optimizers", "high", O, "200 steps of gradient descent (lr 1) from theta and from g.theta: the centered scores stay within 1e-8 of each other for the four orthogonal transformations and part by at least 1e-5 for the three that are not",
        [Le(g_("opt-sgd", n), 1e-8) for n in ORTH] + [Ge(g_("opt-sgd", n), 1e-5) for n in NONORTH], valid=sane("opt-sgd"))
    add("O8", "optimizers", "high", O, "200 steps of Adam (lr 0.01): within 1e-8 for the signed permutation, the unit permutation and the head swap; apart by at least 1e-5 for the rotation and the three that are not orthogonal",
        [Le(g_("opt-adam", n), 1e-8) for n in ORTH[:3]] + [Ge(g_("opt-adam", n), 1e-5) for n in [ORTH[3]] + NONORTH], valid=sane("opt-adam"))
    # --- the conformance harness ------------------------------------------------------------------------------------------------------------------------------------------------------
    import mutate23
    H = "the Python side of the conformance test of batch 0022 (model.py, gauge.py, ops.py, gen22.py), mutated, against the stored output of the Lean definition"
    valid_m = [Eq(f"mutate/unmutated/{c}/status", "equal") for c in ("matrix-d3", "matrix-d2", "matrix-d2-s100", "matrix-d2-s200")]
    for mid, (path, edits, predicted, why) in mutate23.MUTANTS.items():
        if predicted != "detected":
            continue          # the mutants that the configurations cannot reach, or that are equivalent to the original, are recorded in mutate.json and are not claims
        add(f"MUT-{mid}", "conformance power", "high", H, f"mutant {mid} ({why}; {path.split('/')[-1]}) is detected by the comparison with Lean in at least one of the four stored configurations (a mismatch of the arrays, or an error of the generator)",
            [Eq(f"mutate/mutants/{mid}/detected", True)], valid=valid_m + [Eq(f"mutate/mutants/{mid}/predicted", predicted)])
    # --- the finite instance as a checkpoint in standard runtimes ---------------------------------------------------------------------------------------------------------------------
    B = "the finite instance written as a Hugging Face checkpoint and run on all 625 sequences of four symbols (all 780 contexts)"
    add("B1", "runtimes", "high", B, "Transformers in float64 (float32 islands replaced): the logits differ from the numpy forward pass by at most 1e-10 in relative norm and the argmax agrees at every position", [Le("bridge-torch/float64/relative_difference", 1e-10), Ge("bridge-torch/float64/argmax_agreement", 1.0)])
    add("B2", "runtimes", "moderate", B, "Transformers in float32 as shipped: relative difference at most 1e-5, argmax agreement at least 0.999", [Le("bridge-torch/float32/relative_difference", 1e-5), Ge("bridge-torch/float32/argmax_agreement", 0.999)])
    add("B3", "runtimes", "moderate", B, "candle (Rust, float32): relative difference at most 1e-5, argmax agreement at least 0.999", [Le("bridge-candle/candle_float32/relative_difference", 1e-5), Ge("bridge-candle/candle_float32/argmax_agreement", 0.999)])
    # --- language models: the Llama family -----------------------------------------------------------------------------------------------------------------------------------------------
    for key, name in LLAMA:
        f = f"systems-{key}"
        S = f"{name} (a Llama-architecture checkpoint)"
        v = [Eq(f"{f}/float64_islands_patched", True)]
        add(f"K1-{key}", "kinematics", "high", S, "shifting every position by c = 1, 16, 128 changes the logits by at most 1e-6 in relative norm (the rotary embedding makes the scores depend on t - s)", [Le(f"{f}/translation/*/largest", 1e-6)], valid=v)
        add(f"K2-{key}", "kinematics", "high", S, "rotating the planes of the queries and keys by 16 theta_p in the weights leaves the logits (1e-6) and equals, in the keys of the cache after the rotary embedding, the original at positions shifted by 16 (1e-6)",
            [Le(f"{f}/phase/logits/largest", 1e-6), Le(f"{f}/phase/keys_of_the_gauged_model_against_the_original_at_the_shifted_positions/largest", 1e-6)], valid=v)
        add(f"K3-{key}", "kinematics", "high", S, "multiplying all the angles of the rotary embedding by 0.5 or by 2 changes the logits by at least 1e-3: the frequencies are absolute", [Ge(f"{f}/dilation/*/smallest", 1e-3)], valid=v)
        add(f"K5-{key}", "kinematics", "high", S, "with the norm weights folded and the head untied, rotating the stream by a random orthogonal matrix leaves the logits (1e-6), the stream after the embedding and each layer equals the original times Q^T (1e-6), its Gram matrix is unchanged in every layer (1e-6), and the stream itself moves (at least 0.1 at the embedding)",
            [Le(f"{f}/rotation/logits/largest", 1e-6), Le(f"{f}/rotation/stream_equals_the_original_times_Q_transpose/largest", 1e-6), Le(f"{f}/rotation/gram_matrix_of_every_layer/largest", 1e-6), Ge(f"{f}/rotation/stream_moved_at_the_embedding/smallest", 0.1)], valid=v)
    # --- language models whose architecture differs --------------------------------------------------------------------------------------------------------------------------------------
    q, qn = "outside-qwen3-0.6b", "Qwen/Qwen3-0.6B @c1899de (norm of the queries and keys)"
    v = [Eq(f"{q}/float64_islands_patched", True)]
    add("Q1", "kinematics", "high", qn, "shifting every position by 1, 16, 128 changes the logits by at most 1e-6", [Le(f"{q}/translation/*/largest", 1e-6)], valid=v)
    add("Q2", "outside the proved class", "high", qn, "the phase element that reproduces a shift in the Llama (planes rotated by 16 theta_p) changes the logits by at least 1e-3: with the norm of the queries and keys the coincidence of the translation with a weight transformation fails", [Ge(f"{q}/phase/logits/smallest", 1e-3)], valid=v)
    add("Q3", "outside the proved class", "high", qn, "T1 (the scalar of a group and plane on the keys, its inverse on the queries) changes the logits by at least 1e-3", [Ge(f"{q}/qk_norm_transformations/T1_scalar_pair_per_group_and_plane/smallest", 1e-3)], valid=v)
    add("Q4", "outside the proved class", "high", qn, "T2 (a rotation of a plane applied to the keys and the queries, random angles) changes the logits by at least 1e-3", [Ge(f"{q}/qk_norm_transformations/T2_rotation_of_a_plane_on_keys_and_queries/smallest", 1e-3)], valid=v)
    add("Q5", "outside the proved class", "high", qn, "T3 (a positive scalar on the query projection of each head) changes the logits by at most 1e-4 (exact up to the epsilon of the norm)", [Le(f"{q}/qk_norm_transformations/T3_positive_scalar_on_one_query_head/largest", 1e-4)], valid=v)
    add("Q6", "outside the proved class", "high", qn, "T4 (a scalar per plane divided out of the gain of the query norm and multiplied into the gain of the key norm) changes the logits by at most 1e-6", [Le(f"{q}/qk_norm_transformations/T4_scalar_moved_from_the_query_gain_to_the_key_gain/largest", 1e-6)], valid=v)
    add("Q8", "kinematics", "high", qn, "multiplying the angles of the rotary embedding by 0.5 or 2 changes the logits by at least 1e-3", [Ge(f"{q}/dilation/*/smallest", 1e-3)], valid=v)
    add("Q9", "kinematics", "high", qn, "with the norm weights folded and the head untied, rotating the stream leaves the logits (1e-6) and follows the law of a vector (1e-6), the Gram matrix is unchanged (1e-6), the stream moves (0.1)",
        [Le(f"{q}/rotation/logits/largest", 1e-6), Le(f"{q}/rotation/stream_equals_the_original_times_Q_transpose/largest", 1e-6), Le(f"{q}/rotation/gram_matrix_of_every_layer/largest", 1e-6), Ge(f"{q}/rotation/stream_moved_at_the_embedding/smallest", 0.1)], valid=v)
    gp, gn = "outside-gpt2", "openai-community/gpt2 @607a30d (LayerNorm, learned positions)"
    add("G1", "outside the proved class", "high", gn, "adding the all-ones vector to the output of each block's attention and MLP output projections (weights and biases) changes the logits by at most 1e-6", [Le(f"{gp}/layernorm_transformations/L1_all_ones_added_to_the_outputs_of_the_two_output_projections/largest", 1e-6)])
    add("G2", "outside the proved class", "high", gn, "the dual shift of the readers of the two LayerNorms of each block changes the logits by at most 1e-6", [Le(f"{gp}/layernorm_transformations/L2_dual_shift_of_the_readers_of_the_two_layernorms/largest", 1e-6)])
    add("G3", "outside the proved class", "high", gn, "adding a constant c_v to every coordinate of the embedding rows of five tokens changes no other logit than those of the five tokens (at most 1e-6), and those change by c_v times the sum of the final LayerNorm's output (1e-6 relative)",
        [Le(f"{gp}/layernorm_transformations/L3_embedding_rows_of_five_tokens_shifted/unshifted_columns_largest", 1e-6), Le(f"{gp}/layernorm_transformations/L3_embedding_rows_of_five_tokens_shifted/shifted_columns_against_prediction_largest", 1e-6)])
    add("G4", "kinematics", "high", gn, "with learned absolute positions, shifting the positions by 1 or 16 changes the logits by at least 1e-3: translation is not a symmetry", [Ge(f"{gp}/absolute_positions/*/smallest", 1e-3)])
    pp, pn = "outside-pythia-160m", "EleutherAI/pythia-160m @50f5173 (LayerNorm, parallel residual, rotary embedding on a quarter of each head)"
    v = [Eq(f"{pp}/float64_islands_patched", True)]
    add("P1", "kinematics", "high", pn, "shifting every position by 1, 16, 128 changes the logits by at most 1e-6", [Le(f"{pp}/translation/*/largest", 1e-6)], valid=v)
    add("P2", "kinematics", "high", pn, "multiplying the angles of the rotary embedding by 0.5 or 2 changes the logits by at least 1e-3", [Ge(f"{pp}/dilation/*/smallest", 1e-3)], valid=v)
    add("P3", "outside the proved class", "high", pn, "adding the all-ones vector to the outputs of the attention's and the MLP's dense layers (weights and biases) and a constant to each embedding row changes the logits by at most 1e-6", [Le(f"{pp}/layernorm_transformations/writer_shift/largest", 1e-6)], valid=v)
    add("P4", "outside the proved class", "high", pn, "with the LayerNorms folded, an orthogonal Q that fixes the all-ones vector, applied to the stream, changes the logits (the constant offset of the final bias restored) by at most 1e-6", [Le(f"{pp}/layernorm_transformations/rotation_fixing_the_all_ones_vector/largest", 1e-6)], valid=v)
    add("P5", "outside the proved class", "high", pn, "a general orthogonal Q, which does not fix the all-ones vector, changes the logits by at least 1e-3", [Ge(f"{pp}/layernorm_transformations/general_orthogonal_rotation/smallest", 1e-3)], valid=v)
    # --- conversation and a request for a tool call ------------------------------------------------------------------------------------------------------------------------------------
    cq = "chat-qwen3-0.6b"
    Vq = ["H1_signed_permutation", "M1_units", "A1_value_output", "A6_head_swap", "H4_rotation_of_the_stream", "ALL", "T3_head_scalars", "T4_gain_move"]
    for w in ("tool_call", "conversation"):
        add(f"C-qwen3-{w}", "conversation", "high", qn, f"greedy decoding in float64 (at most 64 tokens, ending at the end-of-turn token) from the chat prompt ('{w}') gives the same tokens as the original for each of the eight transformed variants" + (", and the original's answer contains a <tool_call> block (validity)" if w == "tool_call" else ""),
            [Eq(f"{cq}/prompts/{w}/variants/{n}/identical_tokens", True) for n in Vq], valid=[Eq(f"{cq}/prompts/tool_call/original_contains_tool_call_tag", True)] if w == "tool_call" else [])
    cs = "chat-smol-instruct"
    add("C-smol-conversation", "conversation", "high", "HuggingFaceTB/SmolLM2-135M-Instruct @12fd25f", "greedy decoding in float64 (at most 64 tokens, ending at the end-of-turn token) from the chat prompt gives the same tokens as the original for each of the six transformed variants",
        [Eq(f"{cs}/prompts/conversation/variants/{n}/identical_tokens", True) for n in Vq[:6]])
    # --- fine-tuning ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    for key, name in (("smol-instruct", "HuggingFaceTB/SmolLM2-135M-Instruct @12fd25f"), ("delphi-100k", "delphi-suite/v0-llama2-100k @c1372fb")):
        f = f"finetune-{key}"
        sp, rot, us, vo = ("signed_permutation_of_the_hidden_coordinates", "rotation_of_the_stream", "scaling_of_the_feed_forward_units", "invertible_matrix_on_the_value_dimensions")
        valid = [Ge(f"{f}/optimizers/{o}/function_change_of_the_reference_run_relative", 1e-4) for o in ("sgd", "adamw")] + [Le(f"{f}/optimizers/{o}/variants/{n}/difference_before_training_relative", 1e-9) for o in ("sgd", "adamw") for n in (sp, rot, us, vo)]
        d = lambda o, n: f"{f}/optimizers/{o}/variants/{n}/difference_after_training_over_the_change_of_the_reference_run"
        add(f"H-{key}-sgd", "optimizers", "high", name, "ten steps of torch.optim.SGD (lr 0.05, float64, norm weights frozen) from gauge-equivalent weights: the runs stay within 1e-6 of each other (as a multiple of how far training moved the reference) for the signed permutation and the rotation, and part by at least 1e-2 for the unit scaling and the matrix on the value dimensions",
            [Le(d("sgd", sp), 1e-6), Le(d("sgd", rot), 1e-6), Ge(d("sgd", us), 1e-2), Ge(d("sgd", vo), 1e-2)], valid=valid)
        add(f"H-{key}-adamw", "optimizers", "high", name, "ten steps of torch.optim.AdamW (lr 2e-5, no weight decay): within 1e-6 for the signed permutation only; apart by at least 1e-2 for the rotation, the unit scaling and the matrix on the value dimensions",
            [Le(d("adamw", sp), 1e-6), Ge(d("adamw", rot), 1e-2), Ge(d("adamw", us), 1e-2), Ge(d("adamw", vo), 1e-2)], valid=valid)
    # claims whose outcome pattern had been observed in the pilots before registration (at other draws): replications
    for c in C:
        if c.id in REPLICATIONS:
            c.text += " (the outcome had been seen in the pilots before registration, at other draws: a replication)"
    return C


REPLICATIONS = ("R1", "R2a", "R2b", "R2c", "R3", "R4", "R6", "O1", "O2", "O3", "O4", "O5")


def controls():
    """Checks that the experiments can fail and that the harness works; never counted among the claims."""
    K = []
    for key, name in LLAMA:
        f = f"systems-{key}"
        K += [(f"control {key}: the rotation without the fold is not a symmetry (at least 1e-3)", [Ge(f"{f}/rotation_without_the_fold_control/smallest", 1e-3)]),
              (f"control {key}: the shift of 16 positions moves the cache of keys (mean over layers at least 0.1)", [Ge(f"{f}/phase/keys_moved_by_the_shift_control/smallest_over_sentences_of_the_mean_over_layers", 0.1)])]
    K += [("control chat Qwen3: the variant with the permutation applied to the layers but not to the embedding gives different tokens (tool call)", [Eq("chat-qwen3-0.6b/prompts/tool_call/variants/broken_control/identical_tokens", False)]),
          ("control chat Qwen3: the broken variant gives different tokens (conversation)", [Eq("chat-qwen3-0.6b/prompts/conversation/variants/broken_control/identical_tokens", False)]),
          ("control chat SmolLM2: the broken variant gives different tokens", [Eq("chat-smol-instruct/prompts/conversation/variants/broken_control/identical_tokens", False)])]
    return K


def inclusion():
    """The registered test of 'is a language model' (the mean next-token loss on membership.txt is at most 0.9 ln(vocabulary)), applied to each checkpoint: it decides which checkpoints the claims are about; it is not a claim."""
    names = [(f"systems-{key}", name) for key, name in LLAMA] + [("outside-qwen3-0.6b", "Qwen/Qwen3-0.6B @c1899de"), ("outside-gpt2", "openai-community/gpt2 @607a30d"), ("outside-pythia-160m", "EleutherAI/pythia-160m @50f5173")]
    return [(f"inclusion {name}: mean loss on membership.txt at most 0.9 ln(vocabulary)", [Le(f"{f}/membership/fraction_of_ln_vocabulary", 0.9)]) for f, name in names]


def about_a_checkpoint(c):
    """True if the claim is about a named checkpoint; False if it is about the finite instance or the conformance harness."""
    return not c.subject.startswith(("the Llama decoder equations", "training rules applied", "the Python side", "the finite instance written"))


def load_tree(d):
    t = {}
    for f in sorted(os.listdir(d)) if os.path.isdir(d) else []:
        if f.endswith(".json"):
            try:
                t[f[:-5]] = json.load(open(os.path.join(d, f)))
            except ValueError as e:
                t[f[:-5]] = {"__unreadable__": str(e)}
    return t


def grade(d, out):
    tree = load_tree(d)
    rows, counts = [], {}
    for c in claims():
        st, det = c.grade(tree)
        rows.append((c, st, det))
        counts.setdefault(c.group, {"as predicted": 0, "REFUTED": 0, "not run": 0})[st] += 1
    def run_checks(checks):
        res = []
        for text, preds in checks:
            try:
                ok = all(p.check(tree)[0] for p in preds)
                res.append((text, "holds" if ok else "FAILS"))
            except NotRun as e:
                res.append((text, f"not run ({e})"))
        return res
    ctl, inc = run_checks(controls()), run_checks(inclusion())
    os.makedirs(out, exist_ok=True)
    tot = {s: sum(1 for _, st, _ in rows if st == s) for s in ("as predicted", "REFUTED", "not run")}
    ck = [st for c, st, _ in rows if about_a_checkpoint(c)]
    other = [st for c, st, _ in rows if not about_a_checkpoint(c)]
    lines = [f"# Batch 0023, graded (claims: {len(rows)}; as predicted {tot['as predicted']}, REFUTED {tot['REFUTED']}, not run {tot['not run']}; controls: {sum(1 for _, s in ctl if s == 'holds')} of {len(ctl)} hold; inclusion test: {sum(1 for _, s in inc if s == 'holds')} of {len(inc)} checkpoints pass)", "",
             f"Claims about a named checkpoint: {len(ck)}, of which as predicted {ck.count('as predicted')}, REFUTED {ck.count('REFUTED')}, not run {ck.count('not run')}. Claims about the finite instance or the conformance harness: {len(other)}, of which as predicted {other.count('as predicted')}, REFUTED {other.count('REFUTED')}, not run {other.count('not run')}.", "",
             "| group | as predicted | REFUTED | not run |", "|---|---|---|---|"]
    lines += [f"| {g} | {v['as predicted']} | {v['REFUTED']} | {v['not run']} |" for g, v in sorted(counts.items())]
    lines += ["", "| id | group | severity | status | detail |", "|---|---|---|---|---|"]
    lines += [f"| {c.id} | {c.group} | {c.severity} | {st} | {det[:400].replace('|', '/')} |" for c, st, det in rows]
    lines += ["", "## Controls (not claims)", "", "| control | status |", "|---|---|"] + [f"| {t} | {s} |" for t, s in ctl]
    lines += ["", "## Inclusion test of the checkpoints (not claims)", "", "| checkpoint | status |", "|---|---|"] + [f"| {t} | {s} |" for t, s in inc]
    open(os.path.join(out, "graded23.md"), "w").write("\n".join(lines) + "\n")
    with open(os.path.join(out, "graded23.tsv"), "w") as f:
        f.write("id\tgroup\tseverity\tstatus\tsubject\tdetail\n")
        for c, st, det in rows:
            f.write(f"{c.id}\t{c.group}\t{c.severity}\t{st}\t{c.subject}\t{det}\n")
    print("\n".join(lines[:16 + len(counts)]))
    for c, st, det in rows:
        if st != "as predicted":
            print(f"{st}: {c.id} {det[:300]}")
    for t, s in ctl + inc:
        if s != "holds":
            print(f"{s}: {t}")
    return 0 if tot["REFUTED"] == 0 and tot["not run"] == 0 and all(s == "holds" for _, s in ctl + inc) else 1


def table():
    """The registered claims as a compact Markdown table; the long subjects of the finite instance and of the harness are given codes, listed first."""
    codes = {FINITE: "FI", "the Llama decoder equations (hidden 4, 1 layer, 4 heads in 2 groups, 3 units, 5 symbols), random parameters, norm weights random": "FI-d",
             "training rules applied to the finite instance (distillation of a random teacher on all 780 contexts)": "FI-o",
             "the Python side of the conformance test of batch 0022 (model.py, gauge.py, ops.py, gen22.py), mutated, against the stored output of the Lean definition": "harness",
             "the finite instance written as a Hugging Face checkpoint and run on all 625 sequences of four symbols (all 780 contexts)": "FI-ckpt"}
    print("Subjects: **FI** " + FINITE + "; **FI-d** the same at hidden 4, 1 layer, 4 heads in 2 groups, 3 units (norm weights random); **FI-o** " + "the finite instance under training rules: distillation of a random teacher on all 780 contexts"
          + "; **FI-ckpt** the finite instance written as a checkpoint; **harness** the Python side of the conformance test of batch 0022, mutated, against the stored output of the Lean definition. Every other subject is a named checkpoint.\n")
    print("| id | severity | subject | claim | what is compared with what |\n|---|---|---|---|---|")
    for c in claims():
        subj = codes.get(c.subject, c.subject if not c.subject.startswith("the Llama decoder equations at hidden size 4") else "FI (other sizes)")
        tests = "; ".join(p.text() for p in c.preds).replace("|", "/") + (" (validity: " + "; ".join(p.text() for p in c.valid).replace("|", "/") + ")" if c.valid else "")
        print(f"| {c.id} | {c.severity} | {subj} | {c.text} | {tests} |")


def selftest():
    cs = claims()
    ids = [c.id for c in cs]
    assert len(ids) == len(set(ids)), "claim ids are not unique"
    ok = {}
    for c in cs:
        for p in c.preds + c.valid:
            p.fixture(ok, True)
    # no claim has an empty set of predictions, and every claim holds on the fixture built to satisfy it
    for c in cs:
        assert c.preds, c.id
        st, det = c.grade(ok)
        assert st == "as predicted", (c.id, st, det)
    # each claim fails when any one of its own predictions is violated, and is "not run" when a validity condition fails or when there is no data at all
    for c in cs:
        for p in c.preds:
            t = copy.deepcopy(ok)
            p.fixture(t, False)
            st, det = c.grade(t)
            assert st == "REFUTED", (c.id, p.text(), st, det)
        for p in c.valid:
            t = copy.deepcopy(ok)
            p.fixture(t, False)
            st, det = c.grade(t)
            assert st == "not run", (c.id, p.text(), st, det)
        assert c.grade({})[0] == "not run", c.id
    # a NaN never satisfies a threshold; a missing number is not a pass
    t = {}
    put(t, "x", float("nan"))
    assert not Le("x", 1.0).check(t)[0] and not Ge("x", 0.0).check(t)[0]
    for text, preds in controls() + inclusion():
        t = {}
        for p in preds:
            p.fixture(t, True)
        assert all(p.check(t)[0] for p in preds), text
        t = {}
        for p in preds:
            p.fixture(t, False)
        assert not all(p.check(t)[0] for p in preds), text
    print(f"selftest: {len(cs)} claims, {len(controls())} controls and {len(inclusion())} inclusion checks; each holds on its fixture, is refuted when one of its own inputs is violated, is not run without data")
    return 0


if __name__ == "__main__":
    if sys.argv[1] == "--selftest":
        sys.exit(selftest())
    if sys.argv[1] == "--table":
        table()
        sys.exit(0)
    sys.exit(grade(sys.argv[1], sys.argv[2]))
