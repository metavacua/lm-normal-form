#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Batch 0027: the reference implementation (Python 3, standard library only) of the exhaustive checks.

Two finite models, every state enumerated, every number computed by a procedure that does not use the
canonical forms it checks (orbits by enumerating the whole group in X-A and by a flood fill over generators
in X-B, functions by tables). Sanity conditions that hold by construction (the signature procedure answers
exactly when its certificate holds; the rewriting search returns a non-empty set) are assertions, not results.

X-A  one matrix over F_p with r rows and c columns under independent permutations of rows and columns
     (a tensor under the relabelling of its own indices). Procedures compared: the naive "sort the rows,
     then the columns"; the signature procedure with its certificate; the signature procedure without it.
X-B  two matrices that share an index, A (r x K) and B (K x s), whose function is the product AB
     (a model: two tensors coupled through a shared index set, the "bond"). Equivalences: the bag of tensor
     keys (batch 0011's identity, a matrix and its transpose being one key), the orbit of the permutations
     of the bond, of the monomial group (permutations and scalars c, 1/c), of GL_K(F_p), and equality of the
     function AB. Canonical forms: for the monomial group (scale, then sort the slices (a_k, b_k)) and for
     GL_K (the reduced row echelon form of B, valid when B has full row rank). The rewriting system
     (scale a slice; swap two slices that are out of order) is checked for termination, unique normal forms
     and local confluence, and so is a system with a bad sort key (the column a_k alone).

Usage: ref.py TIER OUT.json     with TIER one of tiny, small, full.
"""

import itertools
import json
import sys
import time

TIERS = {
    "tiny": {
        "xa": [(2, 2, 2), (2, 3, 3), (3, 3, 3)],
        "xb": [(2, 2, 2, 2), (3, 2, 2, 2), (5, 1, 2, 1)],
    },
    "small": {
        "xa": [(2, 2, 2), (2, 3, 3), (3, 3, 3), (2, 3, 4), (2, 4, 4), (3, 3, 4)],
        "xb": [(2, 2, 2, 2), (3, 2, 2, 2), (5, 1, 2, 1), (2, 2, 3, 2), (5, 2, 2, 1), (3, 2, 3, 1)],
    },
    "full": {
        "xa": [(2, 2, 2), (2, 3, 3), (3, 3, 3), (2, 3, 4), (2, 4, 4), (3, 3, 4), (2, 4, 5)],
        "xb": [
            (2, 2, 2, 2),
            (3, 2, 2, 2),
            (5, 1, 2, 1),
            (2, 2, 3, 2),
            (5, 2, 2, 1),
            (3, 2, 3, 1),
            (2, 3, 3, 3),
            (5, 2, 2, 2),
        ],
    },
}
LOCAL_CONFLUENCE_MAX_STATES = 40000


# ---------------------------------------------------------------------------------------------- fields


def field(p):
    inv = [0] + [pow(x, p - 2, p) for x in range(1, p)]
    g0 = next(g for g in range(1, p) if len({pow(g, e, p) for e in range(1, p)}) == p - 1) if p > 2 else 1
    return inv, g0


# ---------------------------------------------------------------------------------------------- X-A


def rows_of(A, c):
    return [A[i * c : (i + 1) * c] for i in range(len(A) // c)]


def cols_sorted_matrix(rows, r, c):
    cols = sorted(tuple(rows[i][j] for i in range(r)) for j in range(c))
    return tuple(cols[j][i] for i in range(r) for j in range(c))


def xa_naive(A, r, c):
    return cols_sorted_matrix(sorted(rows_of(A, c)), r, c)


def xa_signature(row):
    return tuple(sorted(row))


def xa_certificate(A, r, c):
    rows = rows_of(A, c)
    for i in range(r):
        for j in range(i + 1, r):
            if xa_signature(rows[i]) == xa_signature(rows[j]) and rows[i] != rows[j]:
                return False
    return True


def xa_sig(A, r, c):
    """The signature procedure; None (unknown) when the certificate fails."""
    if not xa_certificate(A, r, c):
        return None
    rows = sorted(rows_of(A, c), key=lambda row: (xa_signature(row), row))
    return cols_sorted_matrix(rows, r, c)


def xa_sigtie(A, r, c):
    """The signature procedure without the certificate: ties between rows of equal signature keep the input order."""
    rows = sorted(rows_of(A, c), key=xa_signature)
    return cols_sorted_matrix(rows, r, c)


def xa(p, r, c):
    t0 = time.time()
    rperms = list(itertools.permutations(range(r)))
    cperms = list(itertools.permutations(range(c)))
    seen = set()
    orbits = []
    for A in itertools.product(range(p), repeat=r * c):
        if A in seen:
            continue
        members = set()
        for pi in rperms:
            rows = [A[i * c : (i + 1) * c] for i in pi]
            for tau in cperms:
                members.add(tuple(row[j] for row in rows for j in tau))
        seen |= members
        orbits.append(members)
    out = {
        "p": p,
        "r": r,
        "c": c,
        "matrices": p ** (r * c),
        "orbits": len(orbits),
        "cert_matrices": 0,
        "cert_orbits": 0,
        "uncert_orbits": 0,
        "naive_noncanonical_orbits": 0,
        "naive_noncanonical_cert_orbits": 0,
        "sig_noncanonical_cert_orbits": 0,
        "sig_not_in_orbit": 0,
        "sig_collisions": 0,
        "sigtie_noncanonical_uncert_orbits": 0,
        "sigtie_canonical_uncert_orbits": 0,
    }
    sig_owner = {}
    for oid, members in enumerate(orbits):
        flags = {xa_certificate(m, r, c) for m in members}
        assert len(flags) == 1, "the certificate is not constant on an orbit"
        cert = flags.pop()
        naive_vals = {xa_naive(m, r, c) for m in members}
        if len(naive_vals) > 1:
            out["naive_noncanonical_orbits"] += 1
            if cert:
                out["naive_noncanonical_cert_orbits"] += 1
        if cert:
            out["cert_orbits"] += 1
            out["cert_matrices"] += len(members)
            vals = {xa_sig(m, r, c) for m in members}
            assert None not in vals, "the signature procedure answers exactly when its certificate holds"
            if len(vals) > 1:
                out["sig_noncanonical_cert_orbits"] += 1
            v = next(iter(vals))
            if v is not None:
                if v not in members:
                    out["sig_not_in_orbit"] += 1
                if v in sig_owner and sig_owner[v] != oid:
                    out["sig_collisions"] += 1
                sig_owner[v] = oid
        else:
            out["uncert_orbits"] += 1
            assert all(xa_sig(m, r, c) is None for m in members), "the signature procedure answers exactly when its certificate holds"
            vals = {xa_sigtie(m, r, c) for m in members}
            if len(vals) > 1:
                out["sigtie_noncanonical_uncert_orbits"] += 1
            else:
                out["sigtie_canonical_uncert_orbits"] += 1
    out["seconds"] = round(time.time() - t0, 2)
    return out


# ---------------------------------------------------------------------------------------------- X-B


class Bond:
    def __init__(self, p, r, K, s):
        self.p, self.r, self.K, self.s = p, r, K, s
        self.inv, self.g0 = field(p)
        self.ga = r * K

    def slices(self, st):
        r, K, s = self.r, self.K, self.s
        return tuple((tuple(st[i * K + k] for i in range(r)), tuple(st[self.ga + k * s + j] for j in range(s))) for k in range(K))

    def state(self, sl):
        r, K, s = self.r, self.K, self.s
        A = [sl[k][0][i] for i in range(r) for k in range(K)]
        B = [sl[k][1][j] for k in range(K) for j in range(s)]
        return tuple(A + B)

    def product(self, st):
        r, K, s, p = self.r, self.K, self.s, self.p
        return tuple(sum(st[i * K + k] * st[self.ga + k * s + j] for k in range(K)) % p for i in range(r) for j in range(s))

    # --- generators of the three groups, acting on slice tuples
    def gen_swap(self, sl, k):
        t = list(sl)
        t[k], t[k + 1] = t[k + 1], t[k]
        return tuple(t)

    def gen_scale(self, sl, k):
        p, c = self.p, self.g0
        ci = self.inv[c]
        a, b = sl[k]
        t = list(sl)
        t[k] = (tuple(c * x % p for x in a), tuple(ci * x % p for x in b))
        return tuple(t)

    def gen_transvection(self, sl, i, j):
        """b_i += b_j and a_j -= a_i: (A g^-1)(g B) = AB with g = I + E_ij."""
        p = self.p
        t = list(sl)
        t[i] = (sl[i][0], tuple((x + y) % p for x, y in zip(sl[i][1], sl[j][1], strict=True)))
        t[j] = (tuple((x - y) % p for x, y in zip(sl[j][0], sl[i][0], strict=True)), sl[j][1])
        return tuple(t)

    def neighbours(self, st, group):
        sl = self.slices(st)
        K = self.K
        out = [self.gen_swap(sl, k) for k in range(K - 1)]
        if group in ("mono", "gl") and self.p > 2:
            out += [self.gen_scale(sl, k) for k in range(K)]
        if group == "gl":
            out += [self.gen_transvection(sl, i, j) for i in range(K) for j in range(K) if i != j]
        return [self.state(x) for x in out]

    # --- partitions
    def orbit_ids(self, states, group):
        oid = {}
        n = 0
        for st in states:
            if st in oid:
                continue
            oid[st] = n
            todo = [st]
            while todo:
                x = todo.pop()
                for y in self.neighbours(x, group):
                    if y not in oid:
                        oid[y] = n
                        todo.append(y)
            n += 1
        return oid, n

    # --- canonical forms
    def norm_slice(self, sl):
        p, inv = self.p, self.inv
        a, b = sl
        for x in a:
            if x:
                c = inv[x]
                return (tuple(c * y % p for y in a), tuple(x * y % p for y in b))
        for x in b:
            if x:
                c = inv[x]
                return (a, tuple(c * y % p for y in b))
        return sl

    def n_mono(self, st):
        return self.state(tuple(sorted(self.norm_slice(x) for x in self.slices(st))))

    def rank(self, rows):
        p, inv = self.p, self.inv
        rows = [list(x) for x in rows]
        rk = 0
        for col in range(len(rows[0]) if rows else 0):
            piv = next((i for i in range(rk, len(rows)) if rows[i][col] % p), None)
            if piv is None:
                continue
            rows[rk], rows[piv] = rows[piv], rows[rk]
            iv = inv[rows[rk][col]]
            rows[rk] = [x * iv % p for x in rows[rk]]
            for i in range(len(rows)):
                if i != rk and rows[i][col] % p:
                    f = rows[i][col]
                    rows[i] = [(x - f * y) % p for x, y in zip(rows[i], rows[rk], strict=True)]
            rk += 1
        return rk

    def ranks(self, st):
        r, K, s = self.r, self.K, self.s
        A = [st[i * K : (i + 1) * K] for i in range(r)]
        B = [st[self.ga + k * s : self.ga + (k + 1) * s] for k in range(K)]
        return self.rank(A), self.rank(B)

    def n_gl(self, st):
        """The reduced row echelon form of B, with the inverse column operations applied to A; None unless B has full row rank."""
        p, inv, r, K, s = self.p, self.inv, self.r, self.K, self.s
        A = [list(st[i * K : (i + 1) * K]) for i in range(r)]
        B = [list(st[self.ga + k * s : self.ga + (k + 1) * s]) for k in range(K)]
        rk = 0
        for col in range(s):
            piv = next((i for i in range(rk, K) if B[i][col]), None)
            if piv is None:
                continue
            if piv != rk:
                B[rk], B[piv] = B[piv], B[rk]
                for row in A:
                    row[rk], row[piv] = row[piv], row[rk]
            c = B[rk][col]
            iv = inv[c]
            B[rk] = [x * iv % p for x in B[rk]]
            for row in A:
                row[rk] = row[rk] * c % p
            for i in range(K):
                if i != rk and B[i][col]:
                    f = B[i][col]
                    B[i] = [(x - f * y) % p for x, y in zip(B[i], B[rk], strict=True)]
                    for row in A:
                        row[rk] = (row[rk] + f * row[i]) % p
            rk += 1
        if rk < K:
            return None
        return tuple(x for row in A for x in row) + tuple(x for row in B for x in row)

    # --- the bag of tensor keys (batch 0011's identity): a matrix and its transpose are one key
    def bag(self, st):
        r, K, s = self.r, self.K, self.s
        A = [tuple(st[i * K : (i + 1) * K]) for i in range(r)]
        B = [tuple(st[self.ga + k * s : self.ga + (k + 1) * s]) for k in range(K)]

        def key(M, rows, cols):
            flat = tuple(x for row in M for x in row)
            T = tuple(M[i][j] for j in range(cols) for i in range(rows))
            return min(((rows, cols), flat), ((cols, rows), T))

        return tuple(sorted({key(A, r, K), key(B, K, s)}))

    # --- the rewriting system
    def good_steps(self, S):
        out = []
        for k in range(len(S)):
            n = self.norm_slice(S[k])
            if n != S[k]:
                t = list(S)
                t[k] = n
                out.append(tuple(t))
        for j in range(len(S)):
            for k in range(j + 1, len(S)):
                if S[j] > S[k]:
                    t = list(S)
                    t[j], t[k] = t[k], t[j]
                    out.append(tuple(t))
        return out

    def bad_steps(self, S):
        """The same scale rule; the swap compares the vectors a_k alone."""
        out = []
        for k in range(len(S)):
            n = self.norm_slice(S[k])
            if n != S[k]:
                t = list(S)
                t[k] = n
                out.append(tuple(t))
        for j in range(len(S)):
            for k in range(j + 1, len(S)):
                if S[j][0] > S[k][0]:
                    t = list(S)
                    t[j], t[k] = t[k], t[j]
                    out.append(tuple(t))
        return out

    def measure(self, S):
        u = sum(1 for x in S if self.norm_slice(x) != x)
        inv = sum(1 for j in range(len(S)) for k in range(j + 1, len(S)) if S[j] > S[k])
        return (u, inv)


def xb(p, r, K, s):
    t0 = time.time()
    G = Bond(p, r, K, s)
    n = r * K + K * s
    states = list(itertools.product(range(p), repeat=n))
    out = {"p": p, "r": r, "k": K, "s": s, "states": len(states)}
    perm, np_ = G.orbit_ids(states, "perm")
    mono, nm = G.orbit_ids(states, "mono")
    gl, ng = G.orbit_ids(states, "gl")
    fid, bid = {}, {}
    fcount = {}
    bcount = {}
    func = {}
    for st in states:
        m = G.product(st)
        func[st] = m
        fid[st] = fcount.setdefault(m, len(fcount))
        bid[st] = bcount.setdefault(G.bag(st), len(bcount))
    out.update(
        {
            "perm_orbits": np_,
            "mono_orbits": nm,
            "gl_orbits": ng,
            "function_classes": len(fcount),
            "bag_classes": len(bcount),
        }
    )

    def refines(fine, coarse):
        m = {}
        return all(m.setdefault(fine[st], coarse[st]) == coarse[st] for st in states)

    out["chain_perm_in_mono"] = refines(perm, mono)
    out["chain_mono_in_gl"] = refines(mono, gl)
    out["chain_gl_in_function"] = refines(gl, fid)

    def split_count(a, b):
        """The number of classes of a that meet more than one class of b."""
        seen = {}
        for st in states:
            seen.setdefault(a[st], set()).add(b[st])
        return sum(1 for v in seen.values() if len(v) > 1)

    out["function_split_by_bag"] = split_count(fid, bid)
    out["bag_split_by_function"] = split_count(bid, fid)
    out["mono_split_by_bag"] = split_count(mono, bid)
    out["bag_split_by_mono"] = split_count(bid, mono)
    out["gl_split_by_mono"] = split_count(gl, mono)
    out["function_split_by_gl"] = split_count(fid, gl)

    # completeness of the GL orbit where A has full column rank and B has full row rank
    mins = [st for st in states if G.ranks(st) == (K, K)]
    first = {}
    viol = 0
    for st in mins:
        if first.setdefault(func[st], gl[st]) != gl[st]:
            viol += 1
    out.update(
        {
            "min_states": len(mins),
            "min_function_classes": len(first),
            "min_gl_orbits": len({gl[st] for st in mins}),
            "min_violations": viol,
        }
    )

    # canonical form of the monomial group
    vals = {}
    bad = idem = 0
    for st in states:
        v = G.n_mono(st)
        if G.n_mono(v) != v:
            idem += 1
        if vals.setdefault(v, mono[st]) != mono[st]:
            bad += 1
    constant = {}
    for st in states:
        if constant.setdefault(mono[st], G.n_mono(st)) != G.n_mono(st):
            bad += 1
    out.update({"nmono_values": len(vals), "nmono_violations": bad, "nmono_idempotence_violations": idem})

    # canonical form of GL where B has full row rank
    dom = [st for st in states if G.ranks(st)[1] == K]
    vals, bad, constant = {}, 0, {}
    for st in dom:
        v = G.n_gl(st)
        if v is None or G.product(v) != func[st]:
            bad += 1
            continue
        if vals.setdefault(v, gl[st]) != gl[st]:
            bad += 1
        if constant.setdefault(gl[st], v) != v:
            bad += 1
    out.update(
        {
            "ngl_domain_states": len(dom),
            "ngl_values": len(vals),
            "ngl_domain_gl_orbits": len({gl[st] for st in dom}),
            "ngl_violations": bad,
            "ngl_unknown_states": len(states) - len(dom),
        }
    )

    # normalizing each tensor on its own
    changed = joint_changed = 0
    for st in states:
        A = [st[i * K : (i + 1) * K] for i in range(r)]
        B = [st[G.ga + k * s : G.ga + (k + 1) * s] for k in range(K)]
        cols = sorted(tuple(A[i][k] for i in range(r)) for k in range(K))
        A2 = [[cols[k][i] for k in range(K)] for i in range(r)]
        B2 = sorted(tuple(x) for x in B)
        st2 = tuple(x for row in A2 for x in row) + tuple(x for row in B2 for x in row)
        if G.product(st2) != func[st]:
            changed += 1
        if G.product(G.n_mono(st)) != func[st]:
            joint_changed += 1
    out.update({"tensorwise_changed_function": changed, "joint_changed_function": joint_changed})

    # the rewriting system on tuples of slices
    memo_good, memo_bad = {}, {}

    def irr(S, steps, memo):
        if S in memo:
            return memo[S]
        nxt = steps(S)
        res = frozenset([S]) if not nxt else frozenset().union(*(irr(T, steps, memo) for T in nxt))
        memo[S] = res
        return res

    meas_viol = un_fail = irr_not_nf = 0
    for st in states:
        S = G.slices(st)
        for T in G.good_steps(S):
            if not G.measure(T) < G.measure(S):
                meas_viol += 1
        res = irr(S, G.good_steps, memo_good)
        assert res, "the search for irreducible terms returns a non-empty set"
        if len(res) > 1 or next(iter(res)) != G.slices(G.n_mono(st)):
            un_fail += 1
    for S in memo_good:
        if not G.good_steps(S):
            normalized = all(G.norm_slice(x) == x for x in S)
            sorted_ = all(S[j] <= S[j + 1] for j in range(len(S) - 1))
            if not (normalized and sorted_):
                irr_not_nf += 1
    out.update(
        {
            "ars_states": len(states),
            "ars_measure_violations": meas_viol,
            "ars_un_failures": un_fail,
            "ars_irreducible_not_normal": irr_not_nf,
        }
    )

    if len(states) <= LOCAL_CONFLUENCE_MAX_STATES:
        memo_all = {}

        def reach(S):
            if S in memo_all:
                return memo_all[S]
            res = {S}
            for T in G.good_steps(S):
                res |= reach(T)
            memo_all[S] = frozenset(res)
            return memo_all[S]

        peaks = lc_fail = 0
        for st in states:
            S = G.slices(st)
            nxt = sorted(set(G.good_steps(S)))
            for x, y in itertools.combinations(nxt, 2):
                peaks += 1
                if not (reach(x) & reach(y)):
                    lc_fail += 1
        out.update({"ars_lc_peaks": peaks, "ars_lc_failures": lc_fail})
    else:
        out.update({"ars_lc_peaks": None, "ars_lc_failures": None})

    # the system with the bad key
    multi = cert_states = cert_multi = uncert_multi = uncert_single = 0
    for st in states:
        S = G.slices(st)
        res = irr(S, G.bad_steps, memo_bad)
        normed = [G.norm_slice(x) for x in S]
        cert = all(not (normed[j][0] == normed[k][0] and normed[j] != normed[k]) for j in range(len(S)) for k in range(j + 1, len(S)))
        if cert:
            cert_states += 1
        if len(res) > 1:
            multi += 1
            if cert:
                cert_multi += 1
            else:
                uncert_multi += 1
        elif not cert:
            uncert_single += 1
    out.update(
        {
            "bad_ars_multi_irreducible_states": multi,
            "bad_ars_cert_states": cert_states,
            "bad_ars_cert_multi": cert_multi,
            "bad_ars_uncert_multi": uncert_multi,
            "bad_ars_uncert_single": uncert_single,
        }
    )
    out["seconds"] = round(time.time() - t0, 2)
    return out


def main(argv):
    tier, path = argv
    cfg = TIERS[tier]
    res = {"tier": tier, "xa": [], "xb": []}
    for c in cfg["xa"]:
        res["xa"].append(xa(*c))
        print("xa", c, res["xa"][-1]["orbits"], "orbits", res["xa"][-1]["seconds"], "s", flush=True)
    for c in cfg["xb"]:
        res["xb"].append(xb(*c))
        print("xb", c, res["xb"][-1]["states"], "states", res["xb"][-1]["seconds"], "s", flush=True)
    with open(path, "w") as f:
        json.dump(res, f, indent=1, sort_keys=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
