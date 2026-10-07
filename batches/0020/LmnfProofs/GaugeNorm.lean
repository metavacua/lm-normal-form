/-
SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
SPDX-License-Identifier: AGPL-3.0-or-later
-/
import Mathlib
import LmnfProofs.Glue

/-! The gauge of a norm weight (the generator N1), in the model: the weight of a norm multiplied coordinate by coordinate by nonzero `c`, the columns of every matrix that reads the normalized
vector multiplied by `c⁻¹`. The function of the layer is unchanged, and so is the cache: the vector that the readers see is changed (it is multiplied by `c`), the readers compensate. -/

namespace Lmnf

open Matrix

section

variable {K : Type*} [Field K] {T d hv nu : ℕ} {Hh Gr P : Type*} [Fintype Hh] [Fintype Gr] [Fintype P]

theorem reader_scale {m : Type*} (W : Matrix m (Fin d) K) (c : Fin d → K) (hc : ∀ i, c i ≠ 0) (z : Fin d → K) :
    (W * Matrix.diagonal (fun i => (c i)⁻¹)) *ᵥ (fun i => c i * z i) = W *ᵥ z := by
  rw [← Matrix.mulVec_mulVec]
  congr 1
  funext i
  rw [Matrix.mulVec_diagonal]
  have := hc i
  field_simp

theorem nrm_scale (F : Fns K) (γ c x : Fin d → K) : nrm F (fun i => c i * γ i) x = fun i => c i * nrm F γ x i := by
  funext i
  simp only [nrm]
  ring

def attnNorm (L : Layer K d hv nu Hh Gr P) (c : Fin d → K) : Layer K d hv nu Hh Gr P :=
  { L with
    ga := fun i => c i * L.ga i
    Wq := fun hd p => L.Wq hd p * Matrix.diagonal (fun i => (c i)⁻¹)
    Wk := fun g p => L.Wk g p * Matrix.diagonal (fun i => (c i)⁻¹)
    Wv := fun g => L.Wv g * Matrix.diagonal (fun i => (c i)⁻¹) }

theorem qv_attnNorm (E : Env K T Hh Gr P) (L : Layer K d hv nu Hh Gr P) (c : Fin d → K) (hc : ∀ i, c i ≠ 0)
    (h : St K T d) (t : Fin T) (hd : Hh) (p : P) : qv E (attnNorm L c) h t hd p = qv E L h t hd p := by
  show E.rot t p *ᵥ ((L.Wq hd p * Matrix.diagonal (fun i => (c i)⁻¹)) *ᵥ nrm E.F (fun i => c i * L.ga i) (h t))
      = E.rot t p *ᵥ (L.Wq hd p *ᵥ nrm E.F L.ga (h t))
  rw [nrm_scale E.F L.ga c (h t), reader_scale _ c hc (nrm E.F L.ga (h t))]

theorem kv_attnNorm (E : Env K T Hh Gr P) (L : Layer K d hv nu Hh Gr P) (c : Fin d → K) (hc : ∀ i, c i ≠ 0)
    (h : St K T d) (s : Fin T) (g : Gr) (p : P) : kv E (attnNorm L c) h s g p = kv E L h s g p := by
  show E.rot s p *ᵥ ((L.Wk g p * Matrix.diagonal (fun i => (c i)⁻¹)) *ᵥ nrm E.F (fun i => c i * L.ga i) (h s))
      = E.rot s p *ᵥ (L.Wk g p *ᵥ nrm E.F L.ga (h s))
  rw [nrm_scale E.F L.ga c (h s), reader_scale _ c hc (nrm E.F L.ga (h s))]

theorem vv_attnNorm (E : Env K T Hh Gr P) (L : Layer K d hv nu Hh Gr P) (c : Fin d → K) (hc : ∀ i, c i ≠ 0)
    (h : St K T d) (s : Fin T) (g : Gr) : vv E (attnNorm L c) h s g = vv E L h s g := by
  show (L.Wv g * Matrix.diagonal (fun i => (c i)⁻¹)) *ᵥ nrm E.F (fun i => c i * L.ga i) (h s) = L.Wv g *ᵥ nrm E.F L.ga (h s)
  rw [nrm_scale E.F L.ga c (h s), reader_scale _ c hc (nrm E.F L.ga (h s))]

theorem layerFn_attnNorm (E : Env K T Hh Gr P) (L : Layer K d hv nu Hh Gr P) (c : Fin d → K) (hc : ∀ i, c i ≠ 0)
    (h : St K T d) : layerFn E (attnNorm L c) h = layerFn E L h :=
  layerFn_congr (fun h t => afterAttn_congr (fun t =>
      attnOut_of_qkv (fun t hd p => qv_attnNorm E L c hc h t hd p) (fun s g p => kv_attnNorm E L c hc h s g p)
        (fun s g => vv_attnNorm E L c hc h s g) (fun _ => rfl) t) t) (fun _ => rfl) h

def mlpNorm (L : Layer K d hv nu Hh Gr P) (c : Fin d → K) : Layer K d hv nu Hh Gr P :=
  { L with
    gm := fun i => c i * L.gm i
    Wg := L.Wg * Matrix.diagonal (fun i => (c i)⁻¹)
    Wu := L.Wu * Matrix.diagonal (fun i => (c i)⁻¹) }

theorem mlpOut_mlpNorm (E : Env K T Hh Gr P) (L : Layer K d hv nu Hh Gr P) (c : Fin d → K) (hc : ∀ i, c i ≠ 0)
    (y : Fin d → K) : mlpOut E (mlpNorm L c) (nrm E.F (fun i => c i * L.gm i) y) = mlpOut E L (nrm E.F L.gm y) := by
  rw [nrm_scale E.F L.gm c y]
  simp only [mlpOut, mlpNorm, reader_scale _ c hc (nrm E.F L.gm y)]

theorem layerFn_mlpNorm (E : Env K T Hh Gr P) (L : Layer K d hv nu Hh Gr P) (c : Fin d → K) (hc : ∀ i, c i ≠ 0)
    (h : St K T d) : layerFn E (mlpNorm L c) h = layerFn E L h :=
  layerFn_congr (fun _ _ => rfl) (fun y => mlpOut_mlpNorm E L c hc y) h

end

end Lmnf
