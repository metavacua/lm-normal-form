/-
SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
SPDX-License-Identifier: AGPL-3.0-or-later
-/
import Mathlib

/-!
# The gated feed-forward block

* `swiglu_scale`: scaling the up-branch rows by nonzero `c i` and the down columns by `(c i)⁻¹` leaves the block unchanged, for every function `f` applied to the gate (the role `mono`).
  It is false for the gate branch itself, whose nonlinearity is not homogeneous (the role `nl`: permutations only).
* `unit_perm`: a permutation of the units, applied to the activations and to the columns of the down matrix, leaves the block unchanged.
-/

namespace Lmnf

open Matrix

variable {K : Type*} [Field K]

theorem swiglu_scale {n r : ℕ} (f : K → K) (Wd : Matrix (Fin r) (Fin n) K) (c : Fin n → K)
    (hc : ∀ i, c i ≠ 0) (g u : Fin n → K) :
    (Wd * Matrix.diagonal (fun i => (c i)⁻¹)) *ᵥ (fun i => f (g i) * (c i * u i))
      = Wd *ᵥ (fun i => f (g i) * u i) := by
  rw [← Matrix.mulVec_mulVec]
  congr 1
  funext i
  rw [Matrix.mulVec_diagonal]
  have := hc i
  field_simp

theorem unit_perm {n r : ℕ} (σ : Equiv.Perm (Fin n)) (Wd : Matrix (Fin r) (Fin n) K)
    (m : Fin n → K) : (Wd.submatrix id σ) *ᵥ (fun i => m (σ i)) = Wd *ᵥ m := by
  ext j
  simp only [Matrix.mulVec, dotProduct, Matrix.submatrix_apply, id]
  exact Equiv.sum_comp σ (fun i => Wd j i * m i)

end Lmnf
