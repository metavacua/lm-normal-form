/-
SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
SPDX-License-Identifier: AGPL-3.0-or-later
-/
import Mathlib
import LmnfProofs.Glue
import LmnfProofs.Attention

/-! The query/key gauge of rotary embeddings (the generator A3), in the model: a complex scalar `a g p + b g p i` on the keys of the group `g` at the plane `p`, and its inverse transpose on
the queries of the heads of that group. The rotation of every position and plane is a complex scalar (a hypothesis on the environment, true of every rotary embedding), and a complex scalar
commutes with it. The function of the layer is unchanged; of the cache, the keys move by the scalar, the queries by the scalar's inverse transpose, the values do not move. -/

set_option linter.unusedSectionVars false

namespace Lmnf

open Matrix

section

variable {K : Type*} [Field K] {T d hv nu : ℕ} {Hh Gr P : Type*} [Fintype Hh] [Fintype Gr] [Fintype P]

def qkGauge (E : Env K T Hh Gr P) (L : Layer K d hv nu Hh Gr P) (a b : Gr → P → K) : Layer K d hv nu Hh Gr P :=
  { L with
    Wk := fun g p => cplx (a g p) (b g p) * L.Wk g p
    Wq := fun hd p => (((a (E.gr hd) p ^ 2 + b (E.gr hd) p ^ 2)⁻¹ • cplx (a (E.gr hd) p) (-(b (E.gr hd) p)))ᵀ) * L.Wq hd p }

theorem cplx_neg_transpose (a b : K) : ((a ^ 2 + b ^ 2)⁻¹ • cplx a (-b))ᵀ = (a ^ 2 + b ^ 2)⁻¹ • cplx a b := by
  rw [Matrix.transpose_smul, cplx_transpose]
  simp

theorem rot_comm (R C : Matrix (Fin 2) (Fin 2) K) (hc : R * C = C * R) (y : Fin 2 → K) :
    R *ᵥ (C *ᵥ y) = C *ᵥ (R *ᵥ y) := by
  rw [Matrix.mulVec_mulVec, Matrix.mulVec_mulVec, hc]

theorem cplx_comm_smul (s a b c d : K) : cplx c d * (s • cplx a b) = (s • cplx a b) * cplx c d := by
  rw [Matrix.mul_smul, Matrix.smul_mul, cplx_comm]

/-- The keys move by the complex scalar. -/
theorem kv_qk (E : Env K T Hh Gr P) (L : Layer K d hv nu Hh Gr P) (a b : Gr → P → K)
    (hrot : ∀ t p, ∃ c s, E.rot t p = cplx c s) (h : St K T d) (s : Fin T) (g : Gr) (p : P) :
    kv E (qkGauge E L a b) h s g p = cplx (a g p) (b g p) *ᵥ kv E L h s g p := by
  obtain ⟨c, sn, hR⟩ := hrot s p
  show E.rot s p *ᵥ ((cplx (a g p) (b g p) * L.Wk g p) *ᵥ nrm E.F L.ga (h s))
      = cplx (a g p) (b g p) *ᵥ (E.rot s p *ᵥ (L.Wk g p *ᵥ nrm E.F L.ga (h s)))
  rw [← Matrix.mulVec_mulVec, hR]
  exact rot_comm _ _ (cplx_comm c sn (a g p) (b g p)) _

/-- The queries move by the inverse transpose of the complex scalar of the group of their head. -/
theorem qv_qk (E : Env K T Hh Gr P) (L : Layer K d hv nu Hh Gr P) (a b : Gr → P → K)
    (hrot : ∀ t p, ∃ c s, E.rot t p = cplx c s) (h : St K T d) (t : Fin T) (hd : Hh) (p : P) :
    qv E (qkGauge E L a b) h t hd p
      = ((a (E.gr hd) p ^ 2 + b (E.gr hd) p ^ 2)⁻¹ • cplx (a (E.gr hd) p) (b (E.gr hd) p)) *ᵥ qv E L h t hd p := by
  obtain ⟨c, sn, hR⟩ := hrot t p
  show E.rot t p *ᵥ ((((a (E.gr hd) p ^ 2 + b (E.gr hd) p ^ 2)⁻¹ • cplx (a (E.gr hd) p) (-(b (E.gr hd) p)))ᵀ * L.Wq hd p) *ᵥ nrm E.F L.ga (h t))
      = ((a (E.gr hd) p ^ 2 + b (E.gr hd) p ^ 2)⁻¹ • cplx (a (E.gr hd) p) (b (E.gr hd) p)) *ᵥ (E.rot t p *ᵥ (L.Wq hd p *ᵥ nrm E.F L.ga (h t)))
  rw [cplx_neg_transpose, ← Matrix.mulVec_mulVec, hR]
  exact rot_comm _ _ (cplx_comm_smul _ _ _ _ _) _

theorem vv_qk (E : Env K T Hh Gr P) (L : Layer K d hv nu Hh Gr P) (a b : Gr → P → K)
    (h : St K T d) (s : Fin T) (g : Gr) : vv E (qkGauge E L a b) h s g = vv E L h s g := rfl

theorem score_qk (E : Env K T Hh Gr P) (L : Layer K d hv nu Hh Gr P) (a b : Gr → P → K)
    (hnz : ∀ g p, a g p ^ 2 + b g p ^ 2 ≠ 0) (hrot : ∀ t p, ∃ c s, E.rot t p = cplx c s)
    (h : St K T d) (t s : Fin T) (hd : Hh) :
    score E (qkGauge E L a b) h t s hd = score E L h t s hd := by
  simp only [score, qv, kv, qkGauge, ← Matrix.mulVec_mulVec]
  congr 1
  refine Finset.sum_congr rfl fun p _ => ?_
  exact rope_score_gauge (hnz (E.gr hd) p) (hrot t p) (hrot s p) _ _

/-- The function of a layer is unchanged by the query/key gauge of the rotary planes. -/
theorem layerFn_qk (E : Env K T Hh Gr P) (L : Layer K d hv nu Hh Gr P) (a b : Gr → P → K)
    (hnz : ∀ g p, a g p ^ 2 + b g p ^ 2 ≠ 0) (hrot : ∀ t p, ∃ c s, E.rot t p = cplx c s) (h : St K T d) :
    layerFn E (qkGauge E L a b) h = layerFn E L h :=
  layerFn_congr (fun h t => afterAttn_congr (fun t =>
      attnOut_congr (fun t s hd => att_congr (fun t s hd => score_qk E L a b hnz hrot h t s hd) t s hd)
        (fun _ _ => rfl) (fun _ => rfl) t) t) (fun _ => rfl) h

end

end Lmnf
