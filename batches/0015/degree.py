# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
# An upper bound on the degree of the rational functions that a claim of claims.py compares, so that "equal on random elements of F_p" becomes a proof with an
# error bound. The same forward pass and the same claim code are run in an arithmetic whose elements are not numbers but pairs (n, d): a rational function
# whose numerator has degree at most n and whose denominator has degree at most d in the indeterminates (the parameters, the entries of the transformation, and the
# constants of the rotary table, each of degree 1 or, for the rotary table, 2 per step). Sums multiply denominators and products add degrees, which over-counts when
# denominators coincide and so only raises the bound.
# The guarantee. Let f1, f2 be the two results a claim compares, each a vector of rational functions. If f1 - f2 is not the zero function, its numerator after clearing
# denominators is a nonzero polynomial of degree at most N = max over entries of max(n1 + d2, n2 + d1); by the Schwartz-Zippel lemma a uniformly random point of F_p^k
# is a root of it with probability at most N / p, and is a root of one of the denominators (the point is then rejected and redrawn) with probability at most
# (d1 + d2) / p per entry. So one trial that finds f1 = f2 at a random point and no zero denominator proves f1 = f2 as rational functions except with probability at most
# (N + D) / p with (N, D) as bound() returns them. For the claims that expect a difference the same bound is the probability that a trial misses it.
# Not covered: the transformations of which entries are drawn from a subset (permutations, signs) are fixed per trial, and the identity is proved for the draw of the
# continuous parameters; the claim "for every permutation" is the statement for each of the finitely many permutations, one of which is drawn.
import sys
import numpy as np
from ops import P61


FACTORS = {}          # id -> degree of that denominator polynomial
_next = [0]


def new_factor(degree):
    _next[0] += 1
    FACTORS[_next[0]] = degree
    return _next[0]


def _dden(den):
    return sum(m * FACTORS[i] for i, m in den.items())


class Deg:
    """A rational function: numerator degree n, denominator the product of named factors (id -> multiplicity)."""
    __slots__ = ("n", "den")

    def __init__(self, n=0, den=None):
        self.n, self.den = n, den or {}

    def dden(self):
        return _dden(self.den)

    def __add__(self, o):
        if not isinstance(o, Deg):
            return Deg(max(self.n, self.dden()), self.den)
        L = dict(self.den)
        for i, m in o.den.items():
            L[i] = max(L.get(i, 0), m)
        dL = _dden(L)
        return Deg(max(self.n + dL - self.dden(), o.n + dL - o.dden()), L)

    __radd__ = __add__
    __sub__ = __add__
    __rsub__ = __add__

    def __neg__(self):
        return self

    def __mul__(self, o):
        if not isinstance(o, Deg):
            return self
        den = dict(self.den)
        for i, m in o.den.items():
            den[i] = den.get(i, 0) + m
        return Deg(self.n + o.n, den)

    __rmul__ = __mul__

    def __mod__(self, p):
        return self

    def inverse(self):
        return Deg(self.dden(), {new_factor(self.n): 1})

    def __repr__(self):
        return f"Deg({self.n}/{self.dden()})"


def _arr(shape, make):
    a = np.empty(shape, dtype=object)
    for idx in (np.ndindex(*shape) if len(shape) else [()]):
        a[idx] = make()
    return a


class DegOps:
    exact = True
    p = P61
    name = "degrees"

    def asarray(self, x):
        return np.array(x, dtype=object)

    def red(self, x):
        return x

    def mm(self, a, b):
        return np.matmul(a, b)

    def inv(self, x):
        if isinstance(x, Deg):
            return x.inverse()
        out = np.empty(np.shape(x), dtype=object)
        for idx in np.ndindex(*np.shape(x)):
            out[idx] = x[idx].inverse() if isinstance(x[idx], Deg) else x[idx]
        return out

    def act_exp(self, s):
        return s * s + s + 1

    def act_silu(self, s):
        return s * s + 2 * s

    def norm_scale(self, msq):
        return self.inv(msq * msq + 3)

    def attn_scale(self, hd):
        return 7

    def const(self, c):
        return int(c)

    def const_inv(self, n):
        return int(n)

    def rope_tables(self, T, hd, seed=None):
        """Plane j has an element c + i s = ((1 - t^2) + 2 i t) / (1 + t^2) with its own indeterminate t_j; position n has its n-th power: numerator 2n, denominator (1 + t_j^2)^n."""
        C = np.empty((T, hd // 2), dtype=object)
        S = np.empty((T, hd // 2), dtype=object)
        for j in range(hd // 2):
            g = new_factor(2)
            for n in range(T):
                C[n, j] = Deg(2 * n, {g: n} if n else {})
                S[n, j] = Deg(2 * n, {g: n} if n else {})
        return C, S

    def rand(self, rng, shape):
        return _arr(shape, lambda: Deg(1))

    def rand_nonzero(self, rng, shape):
        return _arr(shape, lambda: Deg(1))

    def _orth(self, n):
        den = {new_factor(2): 1 for _ in range(0)}
        for _ in range(n):
            den[new_factor(2)] = 1                  # one factor v^T v of degree 2 per Householder reflection
        return _arr((n, n), lambda: Deg(2 * n, dict(den)))

    def rand_orthogonal(self, n, rng):
        return self._orth(n)

    def rand_orthogonal_fixing_ones(self, n, rng):
        return self._orth(n)

    def rand_invertible(self, n, rng, spread=None):
        return _arr((n, n), lambda: Deg(1))

    def rand_plane_scalars(self, shape, rng):
        return _arr(shape, lambda: Deg(1)), _arr(shape, lambda: Deg(1))

    def matinv(self, A):
        """Entries of the inverse are cofactors over the determinant: numerator degree (n - 1) times the entries' degree, one shared denominator of n times it."""
        n = len(A)
        k = max(int(A[i][j].n) for i in range(n) for j in range(n))
        f = new_factor(n * k)
        return _arr((n, n), lambda: Deg((n - 1) * k, {f: 1}))

    def diff(self, a, b):
        return None


class CutOps(DegOps):
    """The same, with exp, SiLU and the norm function as uninterpreted functions: every call is a cut point. The argument of the call is recorded (its degree), and the
    output is a fresh indeterminate of degree 1 that nothing before it can cancel. This is the interpretation under which OracleFieldOps proves identities, and
    it is why the degree does not compound from layer to layer."""

    name = "degrees, cut points"

    def __init__(self):
        self.calls = {"exp": [], "silu": [], "norm": []}
        self.denominators = []

    def _cut(self, name, x):
        flat = np.ravel(x)
        out = np.empty(flat.shape, dtype=object)
        for i, v in enumerate(flat):
            self.calls[name].append(v.n + v.dden() if isinstance(v, Deg) else 0)
            out[i] = Deg(1)
        return out.reshape(np.shape(x))

    def act_exp(self, s):
        return self._cut("exp", s)

    def act_silu(self, s):
        return self._cut("silu", s)

    def norm_scale(self, msq):
        return self._cut("norm", msq)

    def inv(self, x):
        for v in (np.ravel(x) if not isinstance(x, Deg) else [x]):
            if isinstance(v, Deg):
                self.denominators.append(v.n)
        return super().inv(x)


def oracle_bound(c, base):
    """The union bound on the probability that a trial of the claim in OracleFieldOps accepts although the two sides differ, or rejects a true identity because a denominator
    vanishes: (the collisions between distinct arguments of the same oracle + the final difference + the denominators) / p, with p = 2^127 - 1. Returns the probability
    and the pieces."""
    from ops import P127
    ops = CutOps()
    y1, y2 = c.build(ops, np.random.default_rng(0), base.with_(**c.scope))
    final = sum((u - v).n + (u - v).dden() for u, v in zip(np.ravel(y1), np.ravel(y2)))
    coll = sum((len(v) - 1) * sum(v) for v in ops.calls.values() if v)
    zero = sum(ops.denominators)
    return (coll + final + zero) / P127, {"calls": {k: len(v) for k, v in ops.calls.items()}, "collisions": coll, "final": final, "denominators": zero}


def bound(y1, y2):
    """(N, D): over all entries of f1 - f2, the sum of the numerator degrees and the sum of the denominator degrees: a union bound for the event that a random point
    is a root of the numerator of some entry or of a denominator."""
    N = D = 0
    for u, v in zip(np.ravel(y1), np.ravel(y2)):
        z = u - v
        N += z.n
        D += z.dden()
    return N, D


def claim_bound(c, base):
    arch = base.with_(**c.scope)
    y1, y2 = c.build(DegOps(), np.random.default_rng(0), arch)
    return bound(y1, y2)


if __name__ == "__main__":
    import claims as C
    mode = sys.argv[1] if len(sys.argv) > 1 else "oracle"
    worst = 0.0
    for c in C.CLAIMS:
        if mode == "poly":
            N, D = claim_bound(c, C.BASE)
            pr = (N + D) / P61
            print(f"{c.id:4s} numerator degree <= {N:>14,d}   denominators <= {D:>14,d}   failure probability per trial <= {pr:.2e}")
        else:
            pr, parts = oracle_bound(c, C.BASE)
            print(f"{c.id:4s} calls {sum(parts['calls'].values()):>5d}   collisions <= {parts['collisions']:>12,d}   final <= {parts['final']:>8,d}   denominators <= {parts['denominators']:>8,d}   failure probability per trial <= {pr:.2e}")
        worst = max(worst, pr)
    print(f"largest failure probability per trial over the claims ({mode}): {worst:.2e}")
