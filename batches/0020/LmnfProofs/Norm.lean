/-
SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
SPDX-License-Identifier: AGPL-3.0-or-later
-/
import Mathlib

/-!
# Normalization and the orthogonal gauge

The root-mean-square normalization of a Llama-style Transformer, with an *arbitrary* function `s` in the place of `m ↦ 1 / √(m + ε)`, is equivariant under every matrix that
preserves the standard dot product, whatever `s` is; this is the role `rmsnorm` and the group `O` of the library of `batches/0019/states.dl`. With `ε = 0` and a homogeneous `s`
it is also invariant under a nonzero scalar (the role `rmsnorm0` and the group `CO`).
-/

namespace Lmnf

open Matrix

variable {K : Type*} [Field K]

/-- `x ↦ s (⟨x, x⟩ / d) • x`: the normalization, with `s` arbitrary. -/
def rms {d : ℕ} (s : K → K) (x : Fin d → K) : Fin d → K :=
  s ((x ⬝ᵥ x) / (d : K)) • x

/-- The matrix `Q` preserves the dot product of a vector with itself. -/
def PreservesDot {d : ℕ} (Q : Matrix (Fin d) (Fin d) K) : Prop :=
  ∀ x : Fin d → K, (Q *ᵥ x) ⬝ᵥ (Q *ᵥ x) = x ⬝ᵥ x

/-- An orthogonal matrix (`Qᵀ Q = 1`) preserves the dot product. -/
theorem preservesDot_of_transpose_mul {d : ℕ} {Q : Matrix (Fin d) (Fin d) K} (hQ : Qᵀ * Q = 1) :
    PreservesDot Q := by
  intro x
  rw [Matrix.dotProduct_mulVec, ← Matrix.mulVec_transpose, Matrix.mulVec_mulVec, hQ, Matrix.one_mulVec]

/-- The normalization commutes with every matrix that preserves the dot product, for every function `s`. -/
theorem rms_equivariant {d : ℕ} (s : K → K) {Q : Matrix (Fin d) (Fin d) K} (hQ : PreservesDot Q)
    (x : Fin d → K) : rms s (Q *ᵥ x) = Q *ᵥ rms s x := by
  unfold rms
  rw [hQ x, Matrix.mulVec_smul]

/-- With `ε = 0` (a homogeneous `s`, `s (c² m) = c⁻¹ s m`) the normalization does not see a nonzero scalar. -/
theorem rms_scale {d : ℕ} (s : K → K) (hs : ∀ c m : K, c ≠ 0 → s (c ^ 2 * m) = c⁻¹ * s m)
    {c : K} (hc : c ≠ 0) (x : Fin d → K) : rms s (c • x) = rms s x := by
  unfold rms
  have h1 : ((c • x) ⬝ᵥ (c • x)) / (d : K) = c ^ 2 * ((x ⬝ᵥ x) / (d : K)) := by
    rw [smul_dotProduct, dotProduct_smul, smul_eq_mul, smul_eq_mul]
    ring
  have h2 : c⁻¹ * s ((x ⬝ᵥ x) / (d : K)) * c = s ((x ⬝ᵥ x) / (d : K)) := by
    field_simp
  rw [h1, hs c _ hc, smul_smul, h2]

end Lmnf
