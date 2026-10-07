/-
SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
SPDX-License-Identifier: AGPL-3.0-or-later
-/
import Mathlib
import LmnfProofs.Glue
import LmnfProofs.Attention

/-! The value/output gauge (the generator A1 of the Datalog program), in the model: `A g` on the values of the key/value group `g`, `B g` on the output columns of the heads of that group,
with `B g * A g = 1`. The function of the layer is unchanged; of the cache, the values move by `A g` and the queries and keys do not move. -/

set_option linter.unusedSectionVars false

namespace Lmnf

open Matrix

section

variable {K : Type*} [Field K] {T d hv nu : ℕ} {Hh Gr P : Type*} [Fintype Hh] [Fintype Gr] [Fintype P]

def ovGauge (E : Env K T Hh Gr P) (L : Layer K d hv nu Hh Gr P) (A B : Gr → Matrix (Fin hv) (Fin hv) K) :
    Layer K d hv nu Hh Gr P :=
  { L with Wv := fun g => A g * L.Wv g, Wo := fun hd => L.Wo hd * B (E.gr hd) }

theorem qv_ov (E : Env K T Hh Gr P) (L : Layer K d hv nu Hh Gr P) (A B : Gr → Matrix (Fin hv) (Fin hv) K)
    (h : St K T d) (t : Fin T) (hd : Hh) (p : P) : qv E (ovGauge E L A B) h t hd p = qv E L h t hd p := rfl

theorem kv_ov (E : Env K T Hh Gr P) (L : Layer K d hv nu Hh Gr P) (A B : Gr → Matrix (Fin hv) (Fin hv) K)
    (h : St K T d) (s : Fin T) (g : Gr) (p : P) : kv E (ovGauge E L A B) h s g p = kv E L h s g p := rfl

/-- The values move by `A g`. -/
theorem vv_ov (E : Env K T Hh Gr P) (L : Layer K d hv nu Hh Gr P) (A B : Gr → Matrix (Fin hv) (Fin hv) K)
    (h : St K T d) (s : Fin T) (g : Gr) : vv E (ovGauge E L A B) h s g = A g *ᵥ vv E L h s g := by
  show (A g * L.Wv g) *ᵥ nrm E.F L.ga (h s) = A g *ᵥ (L.Wv g *ᵥ nrm E.F L.ga (h s))
  rw [Matrix.mulVec_mulVec]

theorem attnOut_ov (E : Env K T Hh Gr P) (L : Layer K d hv nu Hh Gr P) (A B : Gr → Matrix (Fin hv) (Fin hv) K)
    (hBA : ∀ g, B g * A g = 1) (h : St K T d) (t : Fin T) :
    attnOut E (ovGauge E L A B) h t = attnOut E L h t := by
  have ha : ∀ t s hd, att E (ovGauge E L A B) h t s hd = att E L h t s hd :=
    fun t s hd => att_congr (fun t s hd => score_of_qk (fun t hd p => qv_ov E L A B h t hd p)
      (fun s g p => kv_ov E L A B h s g p) t s hd) t s hd
  unfold attnOut
  refine Finset.sum_congr rfl fun hd _ => ?_
  simp only [ha]
  exact ov_gauge (L.Wv (E.gr hd)) (L.Wo hd) (A (E.gr hd)) (B (E.gr hd)) (hBA _)
    (fun s => att E L h t s hd) (fun s => nrm E.F L.ga (h s))

/-- The function of a layer is unchanged by the value/output gauge, for every environment. -/
theorem layerFn_ov (E : Env K T Hh Gr P) (L : Layer K d hv nu Hh Gr P) (A B : Gr → Matrix (Fin hv) (Fin hv) K)
    (hBA : ∀ g, B g * A g = 1) (h : St K T d) : layerFn E (ovGauge E L A B) h = layerFn E L h :=
  layerFn_congr (fun h t => afterAttn_congr (fun t => attnOut_ov E L A B hBA h t) t) (fun _ => rfl) h

end

end Lmnf
