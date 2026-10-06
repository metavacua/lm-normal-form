/-
SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
SPDX-License-Identifier: AGPL-3.0-or-later
-/
import Mathlib

/-!
# The attention gauges

* `ov_gauge`: the value/output gauge. For any invertible `A` (with left inverse `B`), the value matrix `A Wv` and the output matrix `Wo B` compute the same attention output, for any
  attention weights (the roles `lin`, the group `GL`).
* `bilinear_gauge`: the pairing of queries and keys, `A` on the keys and `A⁻ᵀ` on the queries (the role `bilinear`).
* `cplx`, `cplx_comm`, `rope_score_gauge`: with rotary embeddings every rotation is a complex scalar `[[c, -s], [s, c]]` on its plane; complex scalars commute with it, so a complex scalar on
  the keys and its inverse on the queries leave every score unchanged (the role `rope`, the group `Cplx`); `rope_gauge_complex` is the same statement in `ℂ`.
* `head_perm`: a permutation of the heads (or of the groups) does not change a sum over them (the role `replica`).
-/

namespace Lmnf

open Matrix

variable {K : Type*} [Field K]

/-- A matrix applied to a weighted sum of vectors is the weighted sum of its images. -/
theorem mulVec_weighted_sum {ι : Type*} [Fintype ι] {m n : ℕ} (M : Matrix (Fin m) (Fin n) K)
    (w : ι → K) (v : ι → Fin n → K) :
    M *ᵥ (∑ i, w i • v i) = ∑ i, w i • (M *ᵥ v i) := by
  have h := map_sum (Matrix.mulVecLin M) (fun i => w i • v i) Finset.univ
  simpa [Matrix.mulVecLin_apply, Matrix.mulVec_smul] using h

/-- The value/output gauge: `A Wv` and `Wo B` with `B A = 1` give the same attention output for every weighting of the positions. -/
theorem ov_gauge {ι : Type*} [Fintype ι] {n h r : ℕ} (Wv : Matrix (Fin h) (Fin n) K)
    (Wo : Matrix (Fin r) (Fin h) K) (A B : Matrix (Fin h) (Fin h) K) (hBA : B * A = 1)
    (w : ι → K) (x : ι → Fin n → K) :
    (Wo * B) *ᵥ (∑ i, w i • ((A * Wv) *ᵥ x i)) = Wo *ᵥ (∑ i, w i • (Wv *ᵥ x i)) := by
  have e : (Wo * B) * (A * Wv) = Wo * Wv := by
    rw [Matrix.mul_assoc, ← Matrix.mul_assoc B A Wv, hBA, Matrix.one_mul]
  rw [mulVec_weighted_sum, mulVec_weighted_sum]
  refine Finset.sum_congr rfl fun i _ => ?_
  simp only [Matrix.mulVec_mulVec, e]

/-- The pairing of a query and a key is unchanged by `A` on the keys and `(B)ᵀ` on the queries when `B A = 1`. -/
theorem bilinear_gauge {n : ℕ} (A B : Matrix (Fin n) (Fin n) K) (hBA : B * A = 1)
    (q k : Fin n → K) : (Bᵀ *ᵥ q) ⬝ᵥ (A *ᵥ k) = q ⬝ᵥ k := by
  rw [Matrix.dotProduct_mulVec, Matrix.mulVec_transpose, Matrix.vecMul_vecMul, hBA, Matrix.vecMul_one]

/-- The complex scalar `a + b i` as the matrix `[[a, -b], [b, a]]` acting on a rotary plane. -/
def cplx (a b : K) : Matrix (Fin 2) (Fin 2) K := !![a, -b; b, a]

/-- Complex scalars commute: in particular with every rotation of a rotary plane. -/
theorem cplx_comm (a b c d : K) : cplx a b * cplx c d = cplx c d * cplx a b := by
  ext i j
  fin_cases i <;> fin_cases j <;> simp [cplx, Matrix.mul_apply, Fin.sum_univ_two] <;> ring

/-- The transpose of a complex scalar is its conjugate. -/
theorem cplx_transpose (a b : K) : (cplx a b)ᵀ = cplx a (-b) := by
  ext i j
  fin_cases i <;> fin_cases j <;> simp [cplx]

/-- The inverse of the complex scalar `a + b i` (when its norm is not zero). -/
theorem cplx_inv_mul {a b : K} (h : a ^ 2 + b ^ 2 ≠ 0) :
    ((a ^ 2 + b ^ 2)⁻¹ • cplx a (-b)) * cplx a b = 1 := by
  ext i j
  fin_cases i <;> fin_cases j <;> simp [cplx, Matrix.mul_apply, Fin.sum_univ_two, Matrix.one_apply] <;> field_simp <;> ring

/-- The score of a rotated query and a rotated key is unchanged by a complex scalar on the keys and its inverse transpose on the queries,
for any two rotations of the plane (matrices of the form `[[c, -s], [s, c]]`). -/
theorem rope_score_gauge {a b : K} (h : a ^ 2 + b ^ 2 ≠ 0) {Rm Rn : Matrix (Fin 2) (Fin 2) K}
    (hm : ∃ c s, Rm = cplx c s) (hn : ∃ c s, Rn = cplx c s) (q k : Fin 2 → K) :
    (Rm *ᵥ (((a ^ 2 + b ^ 2)⁻¹ • cplx a (-b))ᵀ *ᵥ q)) ⬝ᵥ (Rn *ᵥ (cplx a b *ᵥ k)) = (Rm *ᵥ q) ⬝ᵥ (Rn *ᵥ k) := by
  obtain ⟨c, s, rfl⟩ := hm
  obtain ⟨c', s', rfl⟩ := hn
  have hT : ((a ^ 2 + b ^ 2)⁻¹ • cplx a (-b))ᵀ = (a ^ 2 + b ^ 2)⁻¹ • cplx a b := by
    rw [Matrix.transpose_smul, cplx_transpose]
    simp
  have h1 : cplx c s *ᵥ (((a ^ 2 + b ^ 2)⁻¹ • cplx a b) *ᵥ q)
      = ((a ^ 2 + b ^ 2)⁻¹ • cplx a b) *ᵥ (cplx c s *ᵥ q) := by
    rw [Matrix.mulVec_mulVec, Matrix.mulVec_mulVec, Matrix.mul_smul, Matrix.smul_mul, cplx_comm]
  have h2 : cplx c' s' *ᵥ (cplx a b *ᵥ k) = cplx a b *ᵥ (cplx c' s' *ᵥ k) := by
    rw [Matrix.mulVec_mulVec, Matrix.mulVec_mulVec, cplx_comm]
  rw [hT, h1, h2]
  have hB := bilinear_gauge (cplx a b) ((a ^ 2 + b ^ 2)⁻¹ • cplx a (-b)) (cplx_inv_mul h) (cplx c s *ᵥ q) (cplx c' s' *ᵥ k)
  rw [hT] at hB
  exact hB

/-- The same, in the plane `ℂ`: a nonzero complex scalar `z` on the keys and `1 / conj z` on the queries leaves
`conj (ρₘ q) * (ρₙ k)` unchanged, whatever the complex factors `ρₘ`, `ρₙ` of the two rotations. -/
theorem rope_gauge_complex (z ρm ρn q k : ℂ) (hz : z ≠ 0) :
    (starRingEnd ℂ) (ρm * (q / (starRingEnd ℂ) z)) * (ρn * (z * k)) = (starRingEnd ℂ) (ρm * q) * (ρn * k) := by
  have hz' : (starRingEnd ℂ) z ≠ 0 := by simpa using hz
  simp only [map_mul, map_div₀, Complex.conj_conj]
  field_simp

/-- A permutation of the heads, or of the groups, does not change a sum over them. -/
theorem head_perm {ι : Type*} [Fintype ι] {V : Type*} [AddCommMonoid V] (σ : Equiv.Perm ι)
    (F : ι → V) : ∑ i, F (σ i) = ∑ i, F i :=
  Equiv.sum_comp σ F

end Lmnf
