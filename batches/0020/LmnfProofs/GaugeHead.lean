/-
SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
SPDX-License-Identifier: AGPL-3.0-or-later
-/
import Mathlib
import LmnfProofs.Glue
import LmnfProofs.Attention

/-! A permutation `σ` of the heads and a permutation `τ` of the key/value groups, compatible with the map of heads to groups (the generator A6), in the model. The function of the layer is
unchanged; the queries are permuted by `σ`, the keys and the values by `τ`. -/

set_option linter.unusedSectionVars false

namespace Lmnf

open Matrix

section

variable {K : Type*} [Field K] {T d hv nu : ℕ} {Hh Gr P : Type*} [Fintype Hh] [Fintype Gr] [Fintype P]

def headPerm (L : Layer K d hv nu Hh Gr P) (σ : Equiv.Perm Hh) (τ : Equiv.Perm Gr) : Layer K d hv nu Hh Gr P :=
  { L with
    Wq := fun hd p => L.Wq (σ hd) p
    Wo := fun hd => L.Wo (σ hd)
    Wk := fun g p => L.Wk (τ g) p
    Wv := fun g => L.Wv (τ g) }

/-- Definitional: unfolds `headPerm` and `qv`. -/
theorem qv_head (E : Env K T Hh Gr P) (L : Layer K d hv nu Hh Gr P) (σ : Equiv.Perm Hh) (τ : Equiv.Perm Gr)
    (h : St K T d) (t : Fin T) (hd : Hh) (p : P) : qv E (headPerm L σ τ) h t hd p = qv E L h t (σ hd) p := rfl

/-- Definitional: unfolds `headPerm` and `kv`. -/
theorem kv_head (E : Env K T Hh Gr P) (L : Layer K d hv nu Hh Gr P) (σ : Equiv.Perm Hh) (τ : Equiv.Perm Gr)
    (h : St K T d) (s : Fin T) (g : Gr) (p : P) : kv E (headPerm L σ τ) h s g p = kv E L h s (τ g) p := rfl

/-- Definitional: unfolds `headPerm` and `vv`. -/
theorem vv_head (E : Env K T Hh Gr P) (L : Layer K d hv nu Hh Gr P) (σ : Equiv.Perm Hh) (τ : Equiv.Perm Gr)
    (h : St K T d) (s : Fin T) (g : Gr) : vv E (headPerm L σ τ) h s g = vv E L h s (τ g) := rfl

theorem score_head (E : Env K T Hh Gr P) (L : Layer K d hv nu Hh Gr P) (σ : Equiv.Perm Hh) (τ : Equiv.Perm Gr)
    (hστ : ∀ hd, E.gr (σ hd) = τ (E.gr hd)) (h : St K T d) (t s : Fin T) (hd : Hh) :
    score E (headPerm L σ τ) h t s hd = score E L h t s (σ hd) := by
  unfold score
  simp only [qv_head, kv_head, hστ]

theorem att_head (E : Env K T Hh Gr P) (L : Layer K d hv nu Hh Gr P) (σ : Equiv.Perm Hh) (τ : Equiv.Perm Gr)
    (hστ : ∀ hd, E.gr (σ hd) = τ (E.gr hd)) (h : St K T d) (t s : Fin T) (hd : Hh) :
    att E (headPerm L σ τ) h t s hd = att E L h t s (σ hd) := by
  unfold att wt
  simp only [score_head E L σ τ hστ h]

theorem attnOut_headPerm (E : Env K T Hh Gr P) (L : Layer K d hv nu Hh Gr P) (σ : Equiv.Perm Hh) (τ : Equiv.Perm Gr)
    (hστ : ∀ hd, E.gr (σ hd) = τ (E.gr hd)) (h : St K T d) (t : Fin T) :
    attnOut E (headPerm L σ τ) h t = attnOut E L h t := by
  have hWo : ∀ hd, (headPerm L σ τ).Wo hd = L.Wo (σ hd) := fun _ => rfl
  have step : ∀ hd, L.Wo (σ hd) *ᵥ (∑ s, att E L h t s (σ hd) • vv E L h s (τ (E.gr hd)))
      = (fun hd' => L.Wo hd' *ᵥ (∑ s, att E L h t s hd' • vv E L h s (E.gr hd'))) (σ hd) := by
    intro hd
    simp only [hστ]
  calc attnOut E (headPerm L σ τ) h t
      = ∑ hd, L.Wo (σ hd) *ᵥ (∑ s, att E L h t s (σ hd) • vv E L h s (τ (E.gr hd))) := by
        unfold attnOut
        refine Finset.sum_congr rfl fun hd _ => ?_
        simp only [att_head E L σ τ hστ h, vv_head, hWo]
    _ = ∑ hd, (fun hd' => L.Wo hd' *ᵥ (∑ s, att E L h t s hd' • vv E L h s (E.gr hd'))) (σ hd) :=
        Finset.sum_congr rfl fun hd _ => step hd
    _ = ∑ hd, L.Wo hd *ᵥ (∑ s, att E L h t s hd • vv E L h s (E.gr hd)) :=
        head_perm σ (fun hd' => L.Wo hd' *ᵥ (∑ s, att E L h t s hd' • vv E L h s (E.gr hd')))

/-- The function of a layer is unchanged by a permutation of the heads and of the groups. -/
theorem layerFn_headPerm (E : Env K T Hh Gr P) (L : Layer K d hv nu Hh Gr P) (σ : Equiv.Perm Hh) (τ : Equiv.Perm Gr)
    (hστ : ∀ hd, E.gr (σ hd) = τ (E.gr hd)) (h : St K T d) : layerFn E (headPerm L σ τ) h = layerFn E L h :=
  layerFn_congr (fun h t => afterAttn_congr (fun t => attnOut_headPerm E L σ τ hστ h t) t) (fun _ => rfl) h

end

end Lmnf
