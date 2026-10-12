#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Batch 0027: the balanced-ternary reference (Python 3, standard library only).

The digit set {-1, 0, +1} is the set of symmetric residues of F_3, the set of truth values of Kleene's strong
logic (refuted, unknown, supported) and the weight alphabet of BitNet b1.58 and the other ternary-weight
language models. Four groups of exhaustive checks, none of which uses the canonical form it checks:

T-1  facts about the digit set: the residue system, the polynomial forms of min, max and negation, De Morgan,
     the reduced polynomials of F_3^n -> F_3 (rank of the evaluation matrix), the lexicographic order of fixed
     width numerals against the numeric order, truncation against rounding, and the byte coding of trits of
     ggml's TQ1_0 and TQ2_0 (ported from ggml-quants.c).
T-2  the absmean quantizer of BitNet b1.58, RoundClip(W / mean|W|, -1, 1), against the groups that act on
     weights: positive scalars, signs, permutations of rows and columns, and its own idempotence.
T-3  a block of n gated units over F_3: the unit (g, u, w) contributes s(g.x) (u.x) w to the output for the
     nonlinearity s (all 27 functions F_3 -> F_3). The orbits of the sign gauge (u, w) -> (-u, -w) with the
     permutations of the units are found by union-find over every ordered block; the canonical form is compared
     with them: for the sign gauge alone (the class "neither") the canonical form normalizes each unit by the
     leading non-zero trit and then sorts; for the classes "even" and "odd" the unit is normalized to the
     smallest member of its orbit under the generators, so those rows check the sorting and the count of the
     orbits (Burnside) and not the leading-trit rule. The function classes are counted from the tables, and for
     every s the generators of its class are checked to leave every unit's table unchanged; no other symmetry
     is searched for.
T-4  the same unit block for d = e = 1, whose function depends on the units only through three sums
     (P_-, P_0, P_+), so that the function classes are the image of the 2 x 3 matrix (P_-, P_0, P_+) ->
     (F(1), F(-1)) over F_3, restricted to the sums that n units can make (at most n non-zero entries; for
     n >= 2 that is the whole image): a prediction that does not enumerate blocks.

Usage: ternary.py TIER OUT.json     with TIER one of tiny, small.
"""

import itertools
import json
import random
import sys
import time
from fractions import Fraction
from math import comb, fsum

T = (-1, 0, 1)

TIERS = {
    "tiny": {"xc": [(1, 1, 2), (1, 1, 3), (1, 2, 2)]},
    "small": {"xc": [(1, 1, 2), (1, 1, 3), (1, 2, 2), (2, 1, 2), (1, 1, 4)]},
}
UNIONFIND_MAX_STATES = 600000


def m3(x):
    """The symmetric residue of an integer: the one trit congruent to it modulo 3."""
    return ((x + 1) % 3) - 1


def val(ds):
    """The value of a numeral whose digits are trits, most significant first."""
    v = 0
    for d in ds:
        v = 3 * v + d
    return v


# ---------------------------------------------------------------------------------------------- T-1


def rank_mod3(M):
    M = [r[:] for r in M]
    rk = 0
    rows, cols = len(M), len(M[0])
    for c in range(cols):
        piv = next((r for r in range(rk, rows) if M[r][c] % 3), None)
        if piv is None:
            continue
        M[rk], M[piv] = M[piv], M[rk]
        inv = 1 if M[rk][c] % 3 == 1 else 2
        M[rk] = [(v * inv) % 3 for v in M[rk]]
        for r in range(rows):
            if r != rk and M[r][c] % 3:
                f = M[r][c]
                M[r] = [(a - f * b) % 3 for a, b in zip(M[r], M[rk], strict=True)]
        rk += 1
    return rk


def tq1_encode5(trits):
    """ggml quantize_row_tq1_0_ref, one byte of five trits (the first is the most significant)."""
    q = 0
    for t in trits:
        q = q * 3 + (t + 1)  # -1, 0, 1 -> 0, 1, 2
    return (q * 256 + (243 - 1)) // 243  # ceiling division, as ggml writes it


def tq1_decode5(byte):
    """ggml dequantize_row_tq1_0, one byte: n-th trit = ((byte * 3^n mod 256) * 3 >> 8) - 1."""
    return tuple((((byte * 3**n) & 0xFF) * 3 >> 8) - 1 for n in range(5))


def tq1_encode4(trits):
    """The qh part of a TQ1_0 block: four trits, the first shifted to the most significant of five."""
    q = 0
    for t in trits:
        q = q * 3 + (t + 1)
    q *= 3
    return (q * 256 + (243 - 1)) // 243


def tq1_decode4(byte):
    return tuple((((byte * 3**n) & 0xFF) * 3 >> 8) - 1 for n in range(4))


def tq2_encode4(trits):
    """ggml quantize_row_tq2_0_ref, one byte of four trits: two bits each, the first in the low bits."""
    q = 0
    for n, t in enumerate(trits):
        q += ((t + 1) & 3) << (2 * n)
    return q


def tq2_decode4(byte):
    return tuple(((byte >> (2 * n)) & 3) - 1 for n in range(4))


def neg(x):
    """The negation of a trit: the change of sign."""
    return -x


def claims():
    out = {}
    out["residues_complete"] = all(m3(x) in T and (m3(x) - x) % 3 == 0 for x in range(-30, 31)) and {m3(x) for x in range(3)} == set(T)

    ok_int = ok_f3 = True
    for x, y in itertools.product(T, T):
        a = Fraction(x + y - x * x - y * y + x * y + x * x * y * y, 2)
        o = Fraction(x + y + x * x + y * y - x * y - x * x * y * y, 2)
        ok_int &= a == min(x, y) and o == max(x, y)
        ok_f3 &= m3(-(x + y - x * x - y * y + x * y + x * x * y * y)) == min(x, y)  # in F_3 one half is -1
        ok_f3 &= m3(-(x + y + x * x + y * y - x * y - x * x * y * y)) == max(x, y)
    out["k3_polynomials_over_Z"] = ok_int
    out["k3_polynomials_over_F3"] = ok_f3
    out["k3_de_morgan"] = all(-min(x, y) == max(-x, -y) and -max(x, y) == min(-x, -y) for x, y in itertools.product(T, T))
    out["k3_double_negation"] = all(neg(neg(x)) == x for x in T)
    out["k3_excluded_middle_fails_at"] = [x for x in T if max(x, -x) != 1]
    out["k3_non_contradiction_fails_at"] = [x for x in T if min(x, -x) != -1]

    ranks = {}
    for n in (1, 2, 3):
        pts = list(itertools.product(T, repeat=n))
        mons = list(itertools.product(range(3), repeat=n))
        M = [[0] * len(mons) for _ in pts]
        for i, p in enumerate(pts):
            for j, mon in enumerate(mons):
                v = 1
                for xi, ei in zip(p, mon, strict=True):
                    v = (v * (xi**ei)) % 3
                M[i][j] = v % 3
        ranks[str(n)] = {"points": len(pts), "rank": rank_mod3(M)}
    out["reduced_polynomial_rank"] = ranks

    order = {}
    for w in range(1, 8):
        strs = list(itertools.product(T, repeat=w))
        vals = [val(s) for s in strs]
        lead_ok = all((val(s) > 0) - (val(s) < 0) == next((d for d in s if d), 0) for s in strs)
        order[str(w)] = bool(
            strs == sorted(strs) and vals == sorted(vals) and len(set(vals)) == 3**w and vals[0] == -(3**w - 1) // 2 and vals[-1] == (3**w - 1) // 2 and lead_ok
        )
    out["numeral_order_lex_equals_numeric"] = order

    big_err = above = cases = 0
    for w in range(2, 7):
        for s in itertools.product(T, repeat=w):
            for k in range(1, w):
                err = Fraction(val(s), 3**k) - Fraction(val(s[: w - k]))
                cases += 1
                big_err += abs(err) > Fraction(1, 2)
                above += err < 0
    out["truncation_balanced"] = {
        "cases": cases,
        "error_above_half": big_err,
        "truncated_value_above_true_value": above,
    }
    floor_ok = True
    for w in range(2, 7):
        for s in itertools.product((0, 1, 2), repeat=w):
            for k in range(1, w):
                floor_ok &= Fraction(val(s[: w - k])) <= Fraction(val(s), 3**k) < Fraction(val(s[: w - k]) + 1)
    out["truncation_unbalanced_is_floor"] = floor_ok
    out["negation_digitwise"] = all(val(tuple(-d for d in s)) == -val(s) for s in itertools.product(T, repeat=6))
    out["f3_sum_wraps_numeral_sum_carries"] = [m3(1 + 1), val((1, -1))] == [-1, 2]

    t5 = list(itertools.product(T, repeat=5))
    codes = [tq1_encode5(t) for t in t5]
    out["tq1_0_five_trits"] = {
        "tuples": len(t5),
        "distinct_bytes": len(set(codes)),
        "max_byte": max(codes),
        "round_trip": all(tq1_decode5(c) == t for c, t in zip(codes, t5, strict=True)),
        "byte_order_is_lex_order": codes == sorted(codes) and len(set(codes)) == len(codes),
    }
    t4 = list(itertools.product(T, repeat=4))
    c4 = [tq1_encode4(t) for t in t4]
    c2 = [tq2_encode4(t) for t in t4]
    out["tq1_0_four_trits"] = {
        "tuples": len(t4),
        "distinct_bytes": len(set(c4)),
        "round_trip": all(tq1_decode4(c) == t for c, t in zip(c4, t4, strict=True)),
        "byte_order_is_lex_order": c4 == sorted(c4),
    }
    out["tq2_0_four_trits"] = {
        "tuples": len(t4),
        "distinct_bytes": len(set(c2)),
        "round_trip": all(tq2_decode4(c) == t for c, t in zip(c2, t4, strict=True)),
        "byte_order_is_lex_order": c2 == sorted(c2),
    }
    return out


# ---------------------------------------------------------------------------------------------- T-2


def absmean_quant(W, eps=0.0):
    """RoundClip(W / (gamma + eps), -1, 1) with gamma the mean absolute value (BitNet b1.58, equations 1 to 3).
    The mean is taken with fsum, so that it does not depend on the order of the entries; round is half to even,
    as torch.round is, which is odd-symmetric."""
    flat = [abs(x) for row in W for x in row]
    gamma = fsum(flat) / len(flat)
    q = [[max(-1, min(1, round(x / (gamma + eps)))) for x in row] for row in W]
    return q, gamma


def absmax_int8(x):
    """An int8 absmax quantizer for one token in the style of BitNet's: round(x * 127 / max|x|) clipped to [-128, 127]. The papers say the
    activations are scaled per token to [-Qb, Qb] by the maximum absolute value (BitNet v1 writes Qb = 2^(b-1) = 128); the constant 127 and the
    rounding are the common implementations' choice and this script's, not the papers'. The scale is the max-norm, so the result is a function
    of the positive-scale class of x."""
    m = max(abs(v) for v in x)
    return [max(-128, min(127, round(v * 127.0 / m))) for v in x] if m > 0 else [0] * len(x)


def quantizer_equivariance(seed=27, trials=400, shape=(4, 5)):
    rng = random.Random(seed)
    r, c = shape
    bad = {
        "sign_row": 0,
        "sign_col": 0,
        "row_col_perm": 0,
        "scale_pow2": 0,
        "scale_3": 0,
        "idempotent": 0,
        "scale_class_collapse": 0,
        "act_sign": 0,
        "act_perm": 0,
        "act_scale_pow2": 0,
        "act_scale_3": 0,
        "act_idempotent": 0,
    }
    for _ in range(trials):
        W = [[rng.gauss(0, 1) for _ in range(c)] for _ in range(r)]
        q, g = absmean_quant(W)
        # a sign flip of one row, and of one column: the quantizer commutes with it
        i, j = rng.randrange(r), rng.randrange(c)
        Ws = [[-x for x in row] if a == i else list(row) for a, row in enumerate(W)]
        bad["sign_row"] += absmean_quant(Ws)[0] != [[-x for x in row] if a == i else list(row) for a, row in enumerate(q)]
        Wc = [[-x if b == j else x for b, x in enumerate(row)] for row in W]
        bad["sign_col"] += absmean_quant(Wc)[0] != [[-x if b == j else x for b, x in enumerate(row)] for row in q]
        # a permutation of the rows and of the columns
        pr, pc = list(range(r)), list(range(c))
        rng.shuffle(pr)
        rng.shuffle(pc)
        qp, _ = absmean_quant([[W[a][b] for b in pc] for a in pr])
        bad["row_col_perm"] += qp != [[q[a][b] for b in pc] for a in pr]
        # positive scalars: powers of two are exact in binary floating point; 3 is not
        k = rng.randrange(-8, 9)
        q2, _ = absmean_quant([[x * 2.0**k for x in row] for row in W])
        bad["scale_pow2"] += q2 != q
        q3, _ = absmean_quant([[x * 3.0 for x in row] for row in W])
        bad["scale_3"] += q3 != q
        # idempotence on the dequantized weights: Quant(gamma * W~) = W~ (when W~ has a non-zero entry)
        if any(x for row in q for x in row):
            qd, _ = absmean_quant([[g * x for x in row] for row in q])
            bad["idempotent"] += qd != q
        # the class of all positive multiples is one ternary matrix
        qs_ = {tuple(map(tuple, absmean_quant([[x * s for x in row] for row in W])[0])) for s in (0.125, 0.5, 1.0, 2.0, 8.0)}
        bad["scale_class_collapse"] += len(qs_) != 1
        # the activation quantizer on one token: signs, permutations of the channels, positive scalars, idempotence on the dequantized token
        t = [rng.gauss(0, 1) for _ in range(c * 4)]
        a = absmax_int8(t)
        bad["act_sign"] += absmax_int8([-v for v in t]) != [-v for v in a]
        pc = list(range(len(t)))
        rng.shuffle(pc)
        bad["act_perm"] += absmax_int8([t[i] for i in pc]) != [a[i] for i in pc]
        bad["act_scale_pow2"] += absmax_int8([v * 2.0**k for v in t]) != a
        bad["act_scale_3"] += absmax_int8([v * 3.0 for v in t]) != a
        m = max(abs(v) for v in t)
        bad["act_idempotent"] += absmax_int8([v * m / 127.0 for v in a]) != a
    return {"trials": trials, "shape": list(shape), "violations": bad}


# ---------------------------------------------------------------------------------------------- T-3


def parity_class(s):
    """s = (s(-1), s(0), s(1)) as a function F_3 -> F_3."""
    even = s[0] == s[2]
    odd = s[1] == 0 and s[0] == m3(-s[2])
    return "zero" if (even and odd) else "even" if even else "odd" if odd else "neither"


def unit_orbits_pred(d, e, cls):
    """Burnside count of the orbits of one unit (g, u, w) in F_3^d x F_3^d x F_3^e under the gauge of the class."""
    n = 3 ** (2 * d + e)
    fa = 3**d  # fixed points of (g, u, w) -> (g, -u, -w): u = 0, w = 0
    if cls == "neither":
        return (n + fa) // 2
    if cls == "even":  # also (g, u, w) -> (-g, u, w), and their product
        return (n + fa + 3 ** (d + e) + 1) // 4
    if cls == "odd":  # also (g, u, w) -> (-g, u, -w), and their product (-g, -u, w)
        return (n + fa + 3**d + 3**e) // 4
    raise ValueError(cls)


def unit_perm(units, uid, f):
    return [uid[f(u)] for u in units]


def orbit_roots(U, n, gens):
    """Union-find over all U^n ordered blocks; the generators are the adjacent swaps and the per-unit permutations gens."""
    N = U**n
    parent = list(range(N))
    pw = [U ** (n - 1 - i) for i in range(n)]

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for st in range(N):
        digits = [(st // pw[i]) % U for i in range(n)]
        for i in range(n - 1):
            if digits[i] != digits[i + 1]:
                st2 = st + (digits[i + 1] - digits[i]) * pw[i] + (digits[i] - digits[i + 1]) * pw[i + 1]
                a, b = find(st), find(st2)
                if a != b:
                    parent[a] = b
        for gp in gens:
            for i in range(n):
                st2 = st + (gp[digits[i]] - digits[i]) * pw[i]
                if st2 != st:
                    a, b = find(st), find(st2)
                    if a != b:
                        parent[a] = b
    return [find(i) for i in range(N)], pw


def unit_space(d, e):
    """The units (g, u, w) in {-1, 0, 1}^(2d+e), their indices (lexicographic, which is the order of the numerals), the permutation of the units
    that is the sign gauge (g, u, w) -> (g, -u, -w), and the leading-trit normalizer: the sign that makes the leading non-zero trit of u (of w when
    u = 0) +1, applied to u and w."""
    units = list(itertools.product(T, repeat=2 * d + e))
    uid = {u: i for i, u in enumerate(units)}
    p_sgn = [uid[u[:d] + tuple(-x for x in u[d:])] for u in units]

    def lead_norm(u):
        g, uu, w = u[:d], u[d : 2 * d], u[2 * d :]
        c = next((a for a in uu if a), None)
        if c is None:
            c = next((a for a in w if a), 1)
        return g + tuple(c * a for a in uu) + tuple(c * a for a in w)

    return units, uid, p_sgn, lead_norm


def xc(d, e, n):
    t0 = time.time()
    units, uid, p_sgn, lead_norm = unit_space(d, e)
    U = len(units)
    xs = list(itertools.product(T, repeat=d))

    def dot(a, b):
        return m3(sum(p * q for p, q in zip(a, b, strict=True)))

    def f_even(u):
        """(g, u, w) -> (-g, u, w)"""
        return tuple(-x for x in u[:d]) + u[d:]

    def f_odd(u):
        """(g, u, w) -> (-g, u, -w)"""
        return tuple(-x for x in u[:d]) + u[d : 2 * d] + tuple(-x for x in u[2 * d :])

    p_even, p_odd = (unit_perm(units, uid, f) for f in (f_even, f_odd))
    GENS = {"neither": [p_sgn], "even": [p_sgn, p_even], "odd": [p_sgn, p_odd], "zero": [p_sgn, p_even, p_odd]}

    def closure_min(i, gens):
        seen, todo = {i}, [i]
        while todo:
            x = todo.pop()
            for gp in gens:
                y = gp[x]
                if y not in seen:
                    seen.add(y)
                    todo.append(y)
        return min(seen)

    norm_mono = [uid[lead_norm(u)] for u in units]
    norm_cls = {cls: [closure_min(i, g) for i in range(U)] for cls, g in GENS.items()}

    res = {"d": d, "e": e, "n": n, "unit_types": U, "ordered_blocks": U**n, "multisets": comb(U + n - 1, n)}

    # the leading-trit normal form of a unit is a representative of its sign orbit, constant on it, idempotent
    orb = norm_cls["neither"]
    res["lead_norm_is_a_normal_form_of_sign_orbits"] = bool(
        all(orb[norm_mono[i]] == orb[i] for i in range(U))  # representative: norm(u) is in the orbit of u
        and all(norm_mono[p_sgn[i]] == norm_mono[i] for i in range(U))  # invariant under the generator
        and all(norm_mono[norm_mono[i]] == norm_mono[i] for i in range(U))  # idempotent
        and len(set(norm_mono)) == len(set(orb))
    )  # one value per orbit

    # blocks: ordered states, canonical form, orbits by union-find
    N = U**n
    for cls in ("neither", "even", "odd"):
        pred_u = unit_orbits_pred(d, e, cls)
        orbits_u = len(set(norm_cls[cls]))
        row = {"unit_orbits": orbits_u, "unit_orbits_burnside": pred_u, "block_orbits_formula": comb(pred_u + n - 1, n)}
        nm = norm_mono if cls == "neither" else norm_cls[cls]
        if N <= UNIONFIND_MAX_STATES:
            roots, pw = orbit_roots(U, n, GENS[cls])
            nf_of_root, root_of_nf = {}, {}
            inv_ok = idem_ok = True
            for st in range(N):
                digits = [(st // pw[i]) % U for i in range(n)]
                key = tuple(sorted(nm[x] for x in digits))
                nf_of_root.setdefault(roots[st], key)
                root_of_nf.setdefault(key, roots[st])
                inv_ok &= nf_of_root[roots[st]] == key
                inv_ok &= root_of_nf[key] == roots[st]
                idem_ok &= tuple(sorted(nm[x] for x in key)) == key
            row.update(
                {
                    "block_orbits_unionfind": len(set(roots)),
                    "block_canonical_forms": len(root_of_nf),
                    "canonical_form_decisive": inv_ok,
                    "canonical_form_idempotent": idem_ok,
                }
            )
        res[cls] = row

    # functions of the blocks, for each of the 27 nonlinearities, over the multisets (the sum does not see the order)
    def table(u, s):
        g, uu, w = u[:d], u[d : 2 * d], u[2 * d :]
        t = []
        for x in xs:
            a = s[dot(g, x) + 1] * dot(uu, x)
            t.extend(m3(a * w[k]) for k in range(e))
        return tuple(t)

    nlins = list(itertools.product(T, repeat=3))
    per_s = []
    totals = {"gauge_violations": 0, "function_not_constant_on_orbit": 0}
    for s in nlins:
        cls = parity_class(s)
        tabs = [table(u, s) for u in units]
        gauge = cls if cls in ("even", "odd", "zero") else "neither"
        gens = GENS[gauge]
        # soundness: the tables are invariant under every generator of the gauge of the class
        viol = sum(1 for gp in gens for i in range(U) if tabs[gp[i]] != tabs[i])
        totals["gauge_violations"] += viol
        nm = norm_mono if gauge == "neither" else norm_cls[gauge]
        fn_of_key, fn_classes = {}, {}
        for ms in itertools.combinations_with_replacement(range(U), n):
            tab = tuple(sum(tabs[i][k] for i in ms) for k in range(len(tabs[0])))
            tab = tuple(m3(v) for v in tab)
            key = tuple(sorted(nm[i] for i in ms))
            if fn_of_key.setdefault(key, tab) != tab:
                totals["function_not_constant_on_orbit"] += 1
            fn_classes.setdefault(tab, set()).add(key)
        per_s.append(
            {
                "s": list(s),
                "parity": cls,
                "gauge_orbits": len(fn_of_key),
                "function_classes": len(fn_classes),
                "classes_with_several_orbits": sum(1 for v in fn_classes.values() if len(v) > 1),
                "max_orbits_in_a_class": max(len(v) for v in fn_classes.values()),
                "gauge_violations": viol,
            }
        )
    res["per_nonlinearity"] = per_s
    res["totals"] = totals
    res["parity_counts"] = {c: sum(1 for r in per_s if r["parity"] == c) for c in ("even", "odd", "zero", "neither")}
    live = [r for r in per_s if r["parity"] != "zero"]
    res["nonlinearities_with_fewer_function_classes_than_orbits"] = sum(1 for r in live if r["function_classes"] < r["gauge_orbits"])
    res["nonlinearities_with_equal_counts"] = sum(1 for r in live if r["function_classes"] == r["gauge_orbits"])

    # T-4: d = e = 1, the function through the three sums P_g = sum of u_i w_i over the units with gate g
    if d == 1 and e == 1:
        pred = []
        for s in nlins:
            s_m, s_0, s_p = s
            reach = {P for P in itertools.product(T, repeat=3) if sum(1 for v in P if v) <= n}  # (P_-, P_0, P_+)
            images = set()
            for Pm, P0, Pp in reach:
                f1 = m3(P0 * s_0 + Pp * s_p + Pm * s_m)  # F(1)
                fm = m3(-(P0 * s_0 + Pp * s_m + Pm * s_p))  # F(-1)
                images.add((f1, fm))
            pred.append(len(images))
        res["d1e1_function_classes_from_the_three_sums"] = pred
        res["d1e1_matches_enumeration"] = pred == [r["function_classes"] for r in per_s]
    res["seconds"] = round(time.time() - t0, 2)
    return res


def main(argv):
    tier, path = argv
    cfg = TIERS[tier]
    t0 = time.time()
    res = {"tier": tier, "claims": claims(), "quantizer": quantizer_equivariance(), "xc": []}
    print("claims done", flush=True)
    for c in cfg["xc"]:
        res["xc"].append(xc(*c))
        r = res["xc"][-1]
        print(
            "xc",
            c,
            "blocks",
            r["ordered_blocks"],
            "neither/even/odd orbits",
            r["neither"]["block_orbits_formula"],
            r["even"]["block_orbits_formula"],
            r["odd"]["block_orbits_formula"],
            r["seconds"],
            "s",
            flush=True,
        )
    res["seconds"] = round(time.time() - t0, 2)
    with open(path, "w") as f:
        json.dump(res, f, indent=1, sort_keys=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
