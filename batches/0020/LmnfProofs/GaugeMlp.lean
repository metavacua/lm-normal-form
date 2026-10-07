/-
SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
SPDX-License-Identifier: AGPL-3.0-or-later
-/
import Mathlib
import LmnfProofs.Glue
import LmnfProofs.Mlp

/-! The unit gauges of the gated feed-forward block (the generator M1), in the model: the up branch scaled by nonzero `c` and the down columns by `c⁻¹`, and the units permuted. The
function of the layer is unchanged, and the cache (which belongs to the attention block) is untouched: neither the residual stream nor the cache moves, only the activations of the block. -/

namespace Lmnf

open Matrix

section

variable {K : Type*} [Field K] {T d hv nu : ℕ} {Hh Gr P : Type*} [Fintype Hh] [Fintype Gr] [Fintype P]

def mlpScale (L : Layer K d hv nu Hh Gr P) (c : Fin nu → K) : Layer K d hv nu Hh Gr P :=
  { L with Wu := Matrix.diagonal c * L.Wu, Wd := L.Wd * Matrix.diagonal (fun i => (c i)⁻¹) }

theorem mlpOut_scale (E : Env K T Hh Gr P) (L : Layer K d hv nu Hh Gr P) (c : Fin nu → K) (hc : ∀ i, c i ≠ 0)
    (x : Fin d → K) : mlpOut E (mlpScale L c) x = mlpOut E L x := by
  have hu : (Matrix.diagonal c * L.Wu) *ᵥ x = fun i => c i * (L.Wu *ᵥ x) i := by
    rw [← Matrix.mulVec_mulVec]
    funext i
    exact Matrix.mulVec_diagonal c _ i
  simp only [mlpOut, mlpScale, hu]
  exact swiglu_scale E.F.silu L.Wd c hc (L.Wg *ᵥ x) (L.Wu *ᵥ x)

theorem layerFn_mlpScale (E : Env K T Hh Gr P) (L : Layer K d hv nu Hh Gr P) (c : Fin nu → K) (hc : ∀ i, c i ≠ 0)
    (h : St K T d) : layerFn E (mlpScale L c) h = layerFn E L h :=
  layerFn_congr (fun _ _ => rfl) (fun y => mlpOut_scale E L c hc _) h

def unitPerm (L : Layer K d hv nu Hh Gr P) (σ : Equiv.Perm (Fin nu)) : Layer K d hv nu Hh Gr P :=
  { L with Wg := L.Wg.submatrix σ id, Wu := L.Wu.submatrix σ id, Wd := L.Wd.submatrix id σ }

theorem mlpOut_unitPerm (E : Env K T Hh Gr P) (L : Layer K d hv nu Hh Gr P) (σ : Equiv.Perm (Fin nu))
    (x : Fin d → K) : mlpOut E (unitPerm L σ) x = mlpOut E L x := by
  simp only [mlpOut, unitPerm, submatrix_mulVec_row]
  exact unit_perm σ L.Wd (fun j => E.F.silu ((L.Wg *ᵥ x) j) * (L.Wu *ᵥ x) j)

theorem layerFn_unitPerm (E : Env K T Hh Gr P) (L : Layer K d hv nu Hh Gr P) (σ : Equiv.Perm (Fin nu))
    (h : St K T d) : layerFn E (unitPerm L σ) h = layerFn E L h :=
  layerFn_congr (fun _ _ => rfl) (fun y => mlpOut_unitPerm E L σ _) h

end

end Lmnf
