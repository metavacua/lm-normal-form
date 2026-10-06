# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
# The two arithmetics in which the one definition of a Transformer (model.py) is run.
#
#   FloatOps  float64 with the real nonlinearities: exp, SiLU, 1/sqrt(mean square + eps), cos and sin for RoPE. This is the arithmetic of a
#             real language model, carried out in double precision.
#   FieldOps  the prime field F_p, p = 2^61 - 1, with polynomial surrogates for the nonlinearities and a field inverse for the normalizer.
#
# Why a field. An identity f(g . theta) = f(theta) between two rational functions of the parameters theta (and of the parameters of the
# transformation g) that holds for random values in F_p holds as an identity of rational functions with probability at least 1 - D/p,
# D the total degree of the numerator of the difference (Schwartz-Zippel). Over F_p there is no rounding: agreement is exact, and a
# failure is a failure.
# Three field arithmetics, which differ in what the nonlinearities are:
#   FieldOps        p = 2^61 - 1, with polynomial surrogates for exp, SiLU and the norm function. A pass proves the identity *for those polynomials*.
#                   It does not prove it for every nonlinearity.
#   OracleFieldOps  p = 2^127 - 1, with each of exp, SiLU and the norm function a lazily sampled random function F_p -> F_p (a memoized table, one per
#                   function, shared by both sides of a comparison): an uninterpreted function. A pass proves, with the error bound of degree.py, that the arguments
#                   of every call on one side are the arguments of the matching call on the other (otherwise the random outputs differ), and so that the identity
#                   holds for every elementwise function in the place of the nonlinearity. This is the arithmetic of the claims "for every sigma".
#   FloatOps        float64 with the real nonlinearities, which checks the numbers.
# What none of them proves: anything that needs a particular property of exp, SiLU or 1/sqrt (positivity, homogeneity, exp(a + b) = exp(a) exp(b)); anything
# about float32 or bfloat16; anything that depends on the order of the sums. The program of model.py is checked against Hugging Face's separately (conform.py).
import random
import numpy as np

P61 = (1 << 61) - 1
P127 = (1 << 127) - 1


class FloatOps:
    exact = False
    name = "float64"

    def __init__(self, eps=0.0, base=10000.0):
        self.eps, self.base = eps, base

    def asarray(self, x):
        return np.asarray(x, dtype=np.float64)

    def red(self, x):
        return x

    def mm(self, a, b):
        return np.matmul(a, b)

    def inv(self, x):
        return 1.0 / x

    def act_exp(self, s):
        return np.exp(s)

    def act_silu(self, s):
        return s / (1.0 + np.exp(-s))

    def norm_scale(self, msq):
        return 1.0 / np.sqrt(msq + self.eps)

    def attn_scale(self, hd):
        return hd ** -0.5

    def const(self, c):
        return float(c)

    def const_inv(self, n):
        return 1.0 / n

    def rope_tables(self, T, hd):
        """(cos, sin), each (T, hd/2): the rotation of plane j at position n is by n * base^(-2j/hd)."""
        ang = np.arange(T)[:, None] * (self.base ** (-2.0 * np.arange(hd // 2) / hd))[None, :]
        return np.cos(ang), np.sin(ang)

    def rand(self, rng, shape):
        return rng.standard_normal(shape) * 0.5

    def rand_nonzero(self, rng, shape):
        s = np.where(rng.random(shape) < 0.5, -1.0, 1.0)
        return s * np.exp(rng.uniform(-0.7, 0.7, shape))

    def rand_orthogonal(self, n, rng):
        q, r = np.linalg.qr(rng.standard_normal((n, n)))
        return q * np.sign(np.diag(r))[None, :]

    def rand_orthogonal_fixing_ones(self, n, rng):
        """Orthogonal Q with Q 1 = 1: a product of Householder reflections in vectors orthogonal to the all-ones vector."""
        Q = np.identity(n)
        for _ in range(n):
            v = rng.standard_normal(n)
            v = v - v.mean()
            Q = Q @ (np.identity(n) - 2.0 * np.outer(v, v) / (v @ v))
        return Q

    def rand_invertible(self, n, rng, spread=0.5):
        """A well-conditioned random invertible matrix: U diag(exp(u)) V with orthogonal U, V and u uniform in [-spread, spread]."""
        return self.rand_orthogonal(n, rng) @ np.diag(np.exp(rng.uniform(-spread, spread, n))) @ self.rand_orthogonal(n, rng)

    def rand_plane_scalars(self, shape, rng):
        """(a, b) for the matrices [[a, -b], [b, a]]: a random complex number of modulus in [0.5, 2]."""
        mod, ang = np.exp(rng.uniform(-0.7, 0.7, shape)), rng.uniform(0, 2 * np.pi, shape)
        return mod * np.cos(ang), mod * np.sin(ang)

    def matinv(self, A):
        return np.linalg.inv(A)

    def diff(self, a, b):
        """The size of the difference between two arrays of results: the largest absolute difference."""
        return float(np.max(np.abs(np.asarray(a) - np.asarray(b)))) if np.size(a) else 0.0

    def same(self, a, b, tol=1e-9):
        return self.diff(a, b) <= tol * max(1.0, float(np.max(np.abs(a))))


class FieldOps:
    exact = True

    def __init__(self, p=P61, rope_seed=1234):
        self.p = p
        self.rope_seed = rope_seed
        self.name = "F_p, p = 2^61 - 1, polynomial surrogates" if p == P61 else f"F_p, p = {p}"

    def asarray(self, x):
        return np.array(x, dtype=object) % self.p

    def red(self, x):
        return x % self.p

    def mm(self, a, b):
        return self.red(np.matmul(a, b))

    def inv(self, x):
        p = self.p
        flat = [pow(int(v), p - 2, p) if int(v) % p else _raise_zero() for v in np.ravel(x)]
        return np.array(flat, dtype=object).reshape(np.shape(x))

    # polynomial surrogates. Any polynomials do: the claims proved hold for every elementwise function, and these make the failures of the
    # claims that need a particular property (homogeneity of the gate nonlinearity) visible, since neither is homogeneous.
    def act_exp(self, s):
        return self.red(s * s + s + 1)

    def act_silu(self, s):
        return self.red(s * s + 2 * s)

    def norm_scale(self, msq):
        return self.inv(self.red(msq * msq + 3))

    def attn_scale(self, hd):
        return 7

    def const(self, c):
        return int(c) % self.p

    def const_inv(self, n):
        return pow(int(n), self.p - 2, self.p)

    def rope_tables(self, T, hd, seed=None):
        """(cos, sin), each (T, hd/2) as object arrays: plane j rotates by a fixed element r_j of SO(2, F_p) per position; r_j^n by repeated
        multiplication. The elements r_j = (c_j, s_j), c^2 + s^2 = 1, come from the rational parametrization of the circle, are fixed by
        `seed` (not by the parameters being tested) and differ between planes."""
        p, rng = self.p, random.Random(self.rope_seed if seed is None else seed)
        C = np.empty((T, hd // 2), dtype=object)
        S = np.empty((T, hd // 2), dtype=object)
        for j in range(hd // 2):
            t = rng.randrange(2, p - 2)
            den = pow(1 + t * t, p - 2, p)
            c, s = (1 - t * t) * den % p, 2 * t * den % p
            cn, sn = 1, 0
            for n in range(T):
                C[n, j], S[n, j] = cn, sn
                cn, sn = (c * cn - s * sn) % p, (s * cn + c * sn) % p
        return C, S

    def rand(self, rng, shape):
        return self._draw(rng, shape)

    def _draw(self, rng, shape):
        """Uniform elements of F_p: 64 more bits than p has, reduced (the bias is below 2^-64)."""
        n = int(np.prod(shape)) if len(shape) else 1
        nb = (self.p.bit_length() + 7) // 8 + 8
        return np.array([int.from_bytes(rng.bytes(nb), "little") % self.p for _ in range(n)], dtype=object).reshape(shape)

    def rand_nonzero(self, rng, shape):
        x = self._draw(rng, shape)
        flat = np.ravel(x)
        for i in range(len(flat)):
            while int(flat[i]) == 0:
                flat[i] = int(self._draw(rng, (1,))[0])
        return flat.reshape(shape)

    def rand_orthogonal(self, n, rng):
        """A product of n Householder reflections I - 2 v v^T / (v^T v), v with v^T v != 0: orthogonal over F_p, Q^T Q = I."""
        p, Q = self.p, np.identity(n, dtype=object)
        for _ in range(n):
            while True:
                v = self._draw(rng, (n, 1))
                vv = int(self.red(np.matmul(v.T, v))[0, 0])
                if vv:
                    break
            H = (np.identity(n, dtype=object) - 2 * pow(vv, p - 2, p) * np.matmul(v, v.T)) % p
            Q = self.mm(Q, H)
        return Q

    def rand_orthogonal_fixing_ones(self, n, rng):
        """Orthogonal Q over F_p with Q 1 = 1: Householder reflections in vectors v with sum(v) = 0 and v^T v != 0."""
        p, Q, dinv = self.p, np.identity(n, dtype=object), pow(n, self.p - 2, self.p)
        for _ in range(n):
            while True:
                v = self._draw(rng, (n, 1))
                v = (v - (int(v.sum()) % p) * dinv) % p
                vv = int(self.red(np.matmul(v.T, v))[0, 0])
                if vv:
                    break
            H = (np.identity(n, dtype=object) - 2 * pow(vv, p - 2, p) * np.matmul(v, v.T)) % p
            Q = self.mm(Q, H)
        return Q

    def rand_invertible(self, n, rng, spread=None):
        while True:
            A = self._draw(rng, (n, n))
            try:
                self.matinv(A)
                return A
            except ZeroDivisionError:
                continue

    def rand_plane_scalars(self, shape, rng):
        """(a, b) with a^2 + b^2 != 0, so that [[a, -b], [b, a]] is invertible."""
        a, b = self._draw(rng, shape), self._draw(rng, shape)
        fa, fb = np.ravel(a), np.ravel(b)
        for i in range(len(fa)):
            while (int(fa[i]) ** 2 + int(fb[i]) ** 2) % self.p == 0:
                fa[i] = int(self._draw(rng, (1,))[0])
        return fa.reshape(shape), fb.reshape(shape)

    def matinv(self, A):
        """Gauss-Jordan over F_p; ZeroDivisionError if A is singular."""
        p, n = self.p, len(A)
        M = [[int(A[i][j]) % p for j in range(n)] + [1 if i == j else 0 for j in range(n)] for i in range(n)]
        for c in range(n):
            piv = next((r for r in range(c, n) if M[r][c]), None)
            if piv is None:
                raise ZeroDivisionError("singular")
            M[c], M[piv] = M[piv], M[c]
            iv = pow(M[c][c], p - 2, p)
            M[c] = [x * iv % p for x in M[c]]
            for r in range(n):
                if r != c and M[r][c]:
                    f = M[r][c]
                    M[r] = [(x - f * y) % p for x, y in zip(M[r], M[c])]
        return np.array([row[n:] for row in M], dtype=object)

    def diff(self, a, b):
        """The number of entries in which two arrays of results differ (0 means they are equal)."""
        a, b = np.asarray(a, dtype=object), np.asarray(b, dtype=object)
        return int(np.count_nonzero((a - b) % self.p)) if a.size else 0

    def same(self, a, b, tol=None):
        return self.diff(a, b) == 0


class OracleFieldOps(FieldOps):
    """F_p, p = 2^127 - 1, with exp, SiLU and the norm function as lazily sampled random functions, one memoized table each. The same object has to run both sides of a
    comparison, so that equal arguments give equal outputs on both sides and different arguments give independent outputs."""

    def __init__(self, p=P127, seed=0, rope_seed=1234):
        super().__init__(p, rope_seed)
        self.name = "F_p with random oracles, p = 2^127 - 1" if p == P127 else f"F_p with random oracles, p = {p}"
        self._tables = {}
        self._rand = random.Random(seed)

    def _oracle(self, name, x):
        table = self._tables.setdefault(name, {})
        flat = np.ravel(x)
        out = np.empty(flat.shape, dtype=object)
        for i, v in enumerate(flat):
            k = int(v) % self.p
            if k not in table:
                table[k] = self._rand.randrange(self.p)
            out[i] = table[k]
        return out.reshape(np.shape(x))

    def act_exp(self, s):
        return self._oracle("exp", s)

    def act_silu(self, s):
        return self._oracle("silu", s)

    def norm_scale(self, msq):
        return self._oracle("norm", msq)

    def calls(self):
        """The number of distinct arguments each oracle has been given."""
        return {k: len(v) for k, v in self._tables.items()}


def _raise_zero():
    raise ZeroDivisionError("a denominator of the model is zero: draw the parameters again")
