/-
SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
SPDX-License-Identifier: AGPL-3.0-or-later
-/
import Mathlib
import LmnfProofs.Model

/-! The function of a layer depends on its parameters only through the caches (queries, keys, values), the output matrices and the feed-forward block: congruence lemmas for each step. -/

namespace Lmnf

open Matrix

section

variable {K : Type*} [Field K] {T d hv nu : ℕ} {Hh Gr P : Type*} [Fintype Hh] [Fintype Gr] [Fintype P]

theorem submatrix_mulVec_row {m n : ℕ} (M : Matrix (Fin m) (Fin n) K) (σ : Equiv.Perm (Fin m)) (x : Fin n → K) :
    (M.submatrix σ id) *ᵥ x = fun i => (M *ᵥ x) (σ i) := by
  funext i
  first
    | rfl
    | (simp only [Matrix.mulVec, dotProduct, Matrix.submatrix_apply, id]; done)
    | (simp [Matrix.mulVec, dotProduct, Matrix.submatrix_apply]; done)

theorem score_of_qk {E : Env K T Hh Gr P} {L L' : Layer K d hv nu Hh Gr P} {h : St K T d}
    (hq : ∀ t hd p, qv E L' h t hd p = qv E L h t hd p) (hk : ∀ s g p, kv E L' h s g p = kv E L h s g p)
    (t s : Fin T) (hd : Hh) : score E L' h t s hd = score E L h t s hd := by
  unfold score
  simp only [hq, hk]

theorem att_congr {E : Env K T Hh Gr P} {L L' : Layer K d hv nu Hh Gr P} {h : St K T d}
    (hs : ∀ t s hd, score E L' h t s hd = score E L h t s hd) (t s : Fin T) (hd : Hh) :
    att E L' h t s hd = att E L h t s hd := by
  unfold att wt
  simp only [hs]

theorem attnOut_congr {E : Env K T Hh Gr P} {L L' : Layer K d hv nu Hh Gr P} {h : St K T d}
    (hatt : ∀ t s hd, att E L' h t s hd = att E L h t s hd) (hv' : ∀ s g, vv E L' h s g = vv E L h s g)
    (hWo : ∀ hd, L'.Wo hd = L.Wo hd) (t : Fin T) : attnOut E L' h t = attnOut E L h t := by
  unfold attnOut
  simp only [hatt, hv', hWo]

theorem attnOut_of_qkv {E : Env K T Hh Gr P} {L L' : Layer K d hv nu Hh Gr P} {h : St K T d}
    (hq : ∀ t hd p, qv E L' h t hd p = qv E L h t hd p) (hk : ∀ s g p, kv E L' h s g p = kv E L h s g p)
    (hv' : ∀ s g, vv E L' h s g = vv E L h s g) (hWo : ∀ hd, L'.Wo hd = L.Wo hd) (t : Fin T) :
    attnOut E L' h t = attnOut E L h t :=
  attnOut_congr (fun t s hd => att_congr (fun t s hd => score_of_qk hq hk t s hd) t s hd) hv' hWo t

theorem afterAttn_congr {E : Env K T Hh Gr P} {L L' : Layer K d hv nu Hh Gr P} {h : St K T d}
    (h1 : ∀ t, attnOut E L' h t = attnOut E L h t) (t : Fin T) : afterAttn E L' h t = afterAttn E L h t := by
  unfold afterAttn
  rw [h1 t]

/-- The function of a layer is determined by the residual after the attention block and by the feed-forward block with its norm. -/
theorem layerFn_congr {E : Env K T Hh Gr P} {L L' : Layer K d hv nu Hh Gr P}
    (hA : ∀ h t, afterAttn E L' h t = afterAttn E L h t)
    (hM : ∀ y, mlpOut E L' (nrm E.F L'.gm y) = mlpOut E L (nrm E.F L.gm y)) (h : St K T d) :
    layerFn E L' h = layerFn E L h := by
  funext t
  show afterAttn E L' h t + mlpOut E L' (nrm E.F L'.gm (afterAttn E L' h t))
      = afterAttn E L h t + mlpOut E L (nrm E.F L.gm (afterAttn E L h t))
  rw [hA h t, hM]

end

end Lmnf
