/-
SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
SPDX-License-Identifier: AGPL-3.0-or-later
-/
import Mathlib
import LmnfProofs.GaugeStream

/-! The two instances of the gauge of the stream that the experiments use. (1) H1, a signed permutation of the hidden coordinates with the norm weights permuted along: for every model, every
nonlinearity, every permutation `σ` and signs `s` with `s i * s i = 1`, the logits are unchanged. (2) H4 with the norm weights folded away (every norm weight constant): every orthogonal
matrix. -/

set_option linter.unusedSectionVars false

namespace Lmnf

open Matrix

section

variable {K : Type*} [Field K] {T d hv nu : ℕ} {Hh Gr P V : Type*} [Fintype Hh] [Fintype Gr] [Fintype P]

/-- The matrix of `x ↦ (i ↦ s i * x (σ i))`. -/
def sp (σ : Equiv.Perm (Fin d)) (s : Fin d → K) : Matrix (Fin d) (Fin d) K :=
  Matrix.diagonal s * (1 : Matrix (Fin d) (Fin d) K).submatrix σ id

theorem sp_mulVec (σ : Equiv.Perm (Fin d)) (s : Fin d → K) (x : Fin d → K) :
    sp σ s *ᵥ x = fun i => s i * x (σ i) := by
  unfold sp
  rw [← Matrix.mulVec_mulVec, submatrix_mulVec_row, Matrix.one_mulVec]
  funext i
  exact Matrix.mulVec_diagonal s _ i

theorem sp_preservesDot (σ : Equiv.Perm (Fin d)) (s : Fin d → K) (hs : ∀ i, s i * s i = 1) :
    PreservesDot (sp σ s) := by
  intro x
  rw [sp_mulVec]
  show ∑ i, s i * x (σ i) * (s i * x (σ i)) = ∑ i, x i * x i
  calc ∑ i, s i * x (σ i) * (s i * x (σ i)) = ∑ i, x (σ i) * x (σ i) :=
        Finset.sum_congr rfl fun i _ => by linear_combination (x (σ i) * x (σ i)) * hs i
    _ = ∑ i, x i * x i := Equiv.sum_comp σ (fun i => x i * x i)

theorem sp_leftInv (σ : Equiv.Perm (Fin d)) (s : Fin d → K) (hs : ∀ i, s i * s i = 1) :
    LeftInv (sp σ s) (sp σ.symm (fun k => s (σ.symm k))) := by
  intro x
  rw [sp_mulVec, sp_mulVec]
  funext k
  simp only [Equiv.apply_symm_apply]
  have h1 := hs (σ.symm k)
  linear_combination (x k) * h1

theorem sp_compat (σ : Equiv.Perm (Fin d)) (s : Fin d → K) (γ : Fin d → K) :
    Compat (sp σ s) γ (fun i => γ (σ i)) := by
  intro r
  simp only [sp_mulVec]
  funext i
  ring

/-- H1: for every model, every nonlinearity, every permutation of the hidden coordinates and every choice of signs, the weights permuted (and signed) and the norm weights permuted along
compute the same logits. -/
theorem logits_signedPerm (E : Env K T Hh Gr P) (M : Model K d hv nu Hh Gr P V) (toks : Fin T → V)
    (σ : Equiv.Perm (Fin d)) (s : Fin d → K) (hs : ∀ i, s i * s i = 1) :
    logits E (M.streamGauge (sp σ s) (sp σ.symm (fun k => s (σ.symm k))) (fun γ i => γ (σ i))) toks
      = logits E M toks :=
  logits_stream E M toks (sp_preservesDot σ s hs) (sp_leftInv σ s hs) (fun γ i => γ (σ i))
    (fun L _ => ⟨sp_compat σ s L.ga, sp_compat σ s L.gm⟩) (sp_compat σ s M.gf)

theorem compat_const (Q : Matrix (Fin d) (Fin d) K) (c : K) : Compat Q (fun _ => c) (fun _ => c) := by
  intro r
  funext i
  show c * ∑ j, Q i j * r j = ∑ j, Q i j * (c * r j)
  rw [Finset.mul_sum]
  exact Finset.sum_congr rfl fun j _ => by ring

theorem orth_leftInv {Q : Matrix (Fin d) (Fin d) K} (hQ : Qᵀ * Q = 1) : LeftInv Q Qᵀ := by
  intro x
  rw [Matrix.mulVec_mulVec, hQ, Matrix.one_mulVec]

/-- H4 for models whose norm weights, including the final one, are constant vectors: every orthogonal matrix, for every such model and nonlinearity. Folding the norm weights of a general model into the neighbouring matrices (which needs the norm gauge on the final norm as well) is not formalized, so a model with non-constant weights is not covered by this statement. -/
theorem logits_orthogonal (E : Env K T Hh Gr P) (M : Model K d hv nu Hh Gr P V) (toks : Fin T → V)
    {Q : Matrix (Fin d) (Fin d) K} (hQ : Qᵀ * Q = 1)
    (hc : ∀ L ∈ M.layers, (∃ c, L.ga = fun _ => c) ∧ (∃ c, L.gm = fun _ => c)) (hf : ∃ c, M.gf = fun _ => c) :
    logits E (M.streamGauge Q Qᵀ id) toks = logits E M toks := by
  refine logits_stream E M toks (preservesDot_of_transpose_mul hQ) (orth_leftInv hQ) id ?_ ?_
  · intro L hL
    obtain ⟨⟨c1, h1⟩, ⟨c2, h2⟩⟩ := hc L hL
    refine ⟨?_, ?_⟩
    · rw [h1]; exact compat_const Q c1
    · rw [h2]; exact compat_const Q c2
  · obtain ⟨c, h⟩ := hf
    rw [h]
    exact compat_const Q c

end

end Lmnf
