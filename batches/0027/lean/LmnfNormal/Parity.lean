/-
SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
SPDX-License-Identifier: AGPL-3.0-or-later
-/
import Mathlib
import LmnfNormal.Unit

/-!
# What the parity of the nonlinearity adds to the gauge of a gated unit

The gauge of `Unit.lean` (a scalar on the up row, its inverse on the down column) holds for every
nonlinearity `s`.  A symmetry of `s` gives more:

* if `s` is even, `s (-t) = s t`, negating the gate row of a unit does not change the function;
* if `s` is odd, `s (-t) = -s t`, negating the gate row and the down column of a unit does not
  change the function.

`ternary.py` finds exactly these extra symmetries among the 27 nonlinearities of `F_3` (8 even, 2
odd, 16 neither, and the zero function, which is both) and no other one; SiLU and the squared ReLU
of BitNet are neither, which is why the gauge of their units is only the sign pair and the
permutations.
-/

namespace LmnfNormal

namespace GBlock

variable {K : Type*} [Field K] {d e n : ℕ}

/-- If `s` is even, negating the gate row of every unit does not change the function. -/
theorem F_neg_gate_even (s : K → K) (hs : ∀ t, s (-t) = s t) (θ : Fin n → GUnit K d e) :
    F s (fun i => (⟨-(θ i).g, (θ i).u, (θ i).w⟩ : GUnit K d e)) = F s θ := by
  funext x
  unfold F
  refine Finset.sum_congr rfl ?_
  intro i _
  show (s ((-(θ i).g) ⬝ᵥ x) * ((θ i).u ⬝ᵥ x)) • (θ i).w
    = (s ((θ i).g ⬝ᵥ x) * ((θ i).u ⬝ᵥ x)) • (θ i).w
  rw [neg_dotProduct, hs]

/-- If `s` is odd, negating the gate row and the down column of every unit does not change the
function. -/
theorem F_neg_gate_down_odd (s : K → K) (hs : ∀ t, s (-t) = -s t) (θ : Fin n → GUnit K d e) :
    F s (fun i => (⟨-(θ i).g, (θ i).u, -(θ i).w⟩ : GUnit K d e)) = F s θ := by
  funext x
  unfold F
  refine Finset.sum_congr rfl ?_
  intro i _
  show (s ((-(θ i).g) ⬝ᵥ x) * ((θ i).u ⬝ᵥ x)) • (-(θ i).w)
    = (s ((θ i).g ⬝ᵥ x) * ((θ i).u ⬝ᵥ x)) • (θ i).w
  rw [neg_dotProduct, hs, neg_mul, neg_smul_neg]

end GBlock

end LmnfNormal
