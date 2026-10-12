/-
SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
SPDX-License-Identifier: AGPL-3.0-or-later
-/
import Mathlib

/-!
# The canonical form of a function on a finite field: the reduced polynomial

Over a finite field `K` with `q` elements, every function `σ → K` (for `σ` finite) is the function
of exactly one polynomial in which every variable has degree below `q`.  Mathlib has the two halves
(`MvPolynomial.range_evalᵢ`, `MvPolynomial.ker_evalₗ`); this file says them as the one statement
that a canonical form needs: the evaluation of the reduced polynomials is a bijection onto the
functions.  For `K = ZMod 3` the reduced polynomials have degree at most two in each variable, and
the three digits are the symmetric residues `-1, 0, 1`.
-/

universe u

namespace LmnfNormal

/-- The reduced polynomials of a finite field in the variables `σ` are in bijection with the
functions `σ → K`, by evaluation. -/
theorem reduced_polynomial_canonical (σ K : Type u) [Finite σ] [Field K] [Fintype K] :
    Function.Bijective (MvPolynomial.evalᵢ σ K) :=
  ⟨LinearMap.ker_eq_bot.mp (MvPolynomial.ker_evalₗ σ K),
    LinearMap.range_eq_top.mp (MvPolynomial.range_evalᵢ σ K)⟩

/-- The case of the three symmetric residues. -/
theorem reduced_polynomial_canonical_F3 (σ : Type) [Finite σ] :
    Function.Bijective (MvPolynomial.evalᵢ σ (ZMod 3)) := by
  have : Fact (Nat.Prime 3) := ⟨Nat.prime_three⟩
  exact reduced_polynomial_canonical σ (ZMod 3)

end LmnfNormal
