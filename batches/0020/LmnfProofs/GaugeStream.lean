/-
SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
SPDX-License-Identifier: AGPL-3.0-or-later
-/
import Mathlib
import LmnfProofs.Glue

/-! The gauge of the residual stream (the generators H1 and H4): a matrix `Q` that preserves the dot product, with a left inverse `Qi`, acts on the stream of every position; the matrices
that read the stream are multiplied by `Qi` on the right, those that write to it by `Q` on the left; the embedding is carried by `Q` and the head reads through `Qi`; the weight of every
norm is replaced by a weight `ν` that is compatible with `Q` (`Compat`: for a scalar weight any `Q`, for a general weight a signed permutation with the weight permuted).

The stream is not invariant (`Q` moves it) but equivariant (`layerFn_stream`); the cache is invariant (`qv_stream`, `kv_stream`, `vv_stream`); the logits are invariant (`logits_stream`). -/

set_option linter.unusedSectionVars false

namespace Lmnf

open Matrix

section

variable {K : Type*} [Field K] {T d hv nu : ℕ} {Hh Gr P V : Type*} [Fintype Hh] [Fintype Gr] [Fintype P]

/-- The stream of every position moved by `Q`. -/
def mapSt (Q : Matrix (Fin d) (Fin d) K) (h : St K T d) : St K T d := fun t => Q *ᵥ h t

/-- The norm weight `γ'` is `γ` as seen through `Q`: `γ' ⊙ (Q r) = Q (γ ⊙ r)` for every `r`. -/
def Compat (Q : Matrix (Fin d) (Fin d) K) (γ γ' : Fin d → K) : Prop :=
  ∀ r : Fin d → K, (fun i => γ' i * (Q *ᵥ r) i) = Q *ᵥ (fun i => γ i * r i)

/-- `Qi` is a left inverse of `Q`. -/
def LeftInv (Q Qi : Matrix (Fin d) (Fin d) K) : Prop := ∀ x : Fin d → K, Qi *ᵥ (Q *ᵥ x) = x

def streamGauge (L : Layer K d hv nu Hh Gr P) (Q Qi : Matrix (Fin d) (Fin d) K) (ga gm : Fin d → K) :
    Layer K d hv nu Hh Gr P where
  Wq := fun hd p => L.Wq hd p * Qi
  Wk := fun g p => L.Wk g p * Qi
  Wv := fun g => L.Wv g * Qi
  Wo := fun hd => Q * L.Wo hd
  Wg := L.Wg * Qi
  Wu := L.Wu * Qi
  Wd := Q * L.Wd
  ga := ga
  gm := gm

theorem nrm_stream (F : Fns K) {Q : Matrix (Fin d) (Fin d) K} (hQ : PreservesDot Q) {γ γ' : Fin d → K}
    (hc : Compat Q γ γ') (x : Fin d → K) : nrm F γ' (Q *ᵥ x) = Q *ᵥ nrm F γ x := by
  have h1 : rms F.rs (Q *ᵥ x) = Q *ᵥ rms F.rs x := rms_equivariant F.rs hQ x
  funext i
  have h2 := congrFun (hc (rms F.rs x)) i
  show γ' i * rms F.rs (Q *ᵥ x) i = (Q *ᵥ fun j => γ j * rms F.rs x j) i
  rw [h1]
  exact h2

theorem reader_stream {m : Type*} (W : Matrix m (Fin d) K) {Q Qi : Matrix (Fin d) (Fin d) K} (hi : LeftInv Q Qi)
    (z : Fin d → K) : (W * Qi) *ᵥ (Q *ᵥ z) = W *ᵥ z := by
  rw [← Matrix.mulVec_mulVec, hi z]

/-- The queries do not move. -/
theorem qv_stream (E : Env K T Hh Gr P) (L : Layer K d hv nu Hh Gr P) {Q Qi : Matrix (Fin d) (Fin d) K}
    (hQ : PreservesDot Q) (hi : LeftInv Q Qi) {ga gm : Fin d → K} (hca : Compat Q L.ga ga) (h : St K T d)
    (t : Fin T) (hd : Hh) (p : P) :
    qv E (streamGauge L Q Qi ga gm) (mapSt Q h) t hd p = qv E L h t hd p := by
  show E.rot t p *ᵥ ((L.Wq hd p * Qi) *ᵥ nrm E.F ga (Q *ᵥ h t)) = E.rot t p *ᵥ (L.Wq hd p *ᵥ nrm E.F L.ga (h t))
  rw [nrm_stream E.F hQ hca, reader_stream _ hi]

/-- The keys do not move. -/
theorem kv_stream (E : Env K T Hh Gr P) (L : Layer K d hv nu Hh Gr P) {Q Qi : Matrix (Fin d) (Fin d) K}
    (hQ : PreservesDot Q) (hi : LeftInv Q Qi) {ga gm : Fin d → K} (hca : Compat Q L.ga ga) (h : St K T d)
    (s : Fin T) (g : Gr) (p : P) :
    kv E (streamGauge L Q Qi ga gm) (mapSt Q h) s g p = kv E L h s g p := by
  show E.rot s p *ᵥ ((L.Wk g p * Qi) *ᵥ nrm E.F ga (Q *ᵥ h s)) = E.rot s p *ᵥ (L.Wk g p *ᵥ nrm E.F L.ga (h s))
  rw [nrm_stream E.F hQ hca, reader_stream _ hi]

/-- The values do not move. -/
theorem vv_stream (E : Env K T Hh Gr P) (L : Layer K d hv nu Hh Gr P) {Q Qi : Matrix (Fin d) (Fin d) K}
    (hQ : PreservesDot Q) (hi : LeftInv Q Qi) {ga gm : Fin d → K} (hca : Compat Q L.ga ga) (h : St K T d)
    (s : Fin T) (g : Gr) :
    vv E (streamGauge L Q Qi ga gm) (mapSt Q h) s g = vv E L h s g := by
  show (L.Wv g * Qi) *ᵥ nrm E.F ga (Q *ᵥ h s) = L.Wv g *ᵥ nrm E.F L.ga (h s)
  rw [nrm_stream E.F hQ hca, reader_stream _ hi]

theorem attnOut_stream (E : Env K T Hh Gr P) (L : Layer K d hv nu Hh Gr P) {Q Qi : Matrix (Fin d) (Fin d) K}
    (hQ : PreservesDot Q) (hi : LeftInv Q Qi) {ga gm : Fin d → K} (hca : Compat Q L.ga ga) (h : St K T d)
    (t : Fin T) :
    attnOut E (streamGauge L Q Qi ga gm) (mapSt Q h) t = Q *ᵥ attnOut E L h t := by
  have ha : ∀ t s hd, att E (streamGauge L Q Qi ga gm) (mapSt Q h) t s hd = att E L h t s hd :=
    fun t s hd => att_congr (fun t s hd => score_of_qk (fun t hd p => qv_stream E L hQ hi hca h t hd p)
      (fun s g p => kv_stream E L hQ hi hca h s g p) t s hd) t s hd
  have hv : ∀ s g, vv E (streamGauge L Q Qi ga gm) (mapSt Q h) s g = vv E L h s g :=
    fun s g => vv_stream E L hQ hi hca h s g
  have hWo : ∀ hd, (streamGauge L Q Qi ga gm).Wo hd = Q * L.Wo hd := fun _ => rfl
  unfold attnOut
  simp only [ha, hv, hWo]
  rw [Matrix.mulVec_sum]
  refine Finset.sum_congr rfl fun hd _ => ?_
  rw [Matrix.mulVec_mulVec]

theorem afterAttn_stream (E : Env K T Hh Gr P) (L : Layer K d hv nu Hh Gr P) {Q Qi : Matrix (Fin d) (Fin d) K}
    (hQ : PreservesDot Q) (hi : LeftInv Q Qi) {ga gm : Fin d → K} (hca : Compat Q L.ga ga) (h : St K T d)
    (t : Fin T) :
    afterAttn E (streamGauge L Q Qi ga gm) (mapSt Q h) t = Q *ᵥ afterAttn E L h t := by
  show Q *ᵥ h t + attnOut E (streamGauge L Q Qi ga gm) (mapSt Q h) t = Q *ᵥ (h t + attnOut E L h t)
  rw [attnOut_stream E L hQ hi hca h t, Matrix.mulVec_add]

theorem mlpOut_stream (E : Env K T Hh Gr P) (L : Layer K d hv nu Hh Gr P) {Q Qi : Matrix (Fin d) (Fin d) K}
    (hQ : PreservesDot Q) (hi : LeftInv Q Qi) {ga gm : Fin d → K} (hcm : Compat Q L.gm gm) (y : Fin d → K) :
    mlpOut E (streamGauge L Q Qi ga gm) (nrm E.F gm (Q *ᵥ y)) = Q *ᵥ mlpOut E L (nrm E.F L.gm y) := by
  rw [nrm_stream E.F hQ hcm]
  simp only [mlpOut, streamGauge, reader_stream _ hi]
  rw [Matrix.mulVec_mulVec]

/-- The stream is equivariant: the layer of the transformed model, applied to the moved stream, gives the moved output of the original layer. -/
theorem layerFn_stream (E : Env K T Hh Gr P) (L : Layer K d hv nu Hh Gr P) {Q Qi : Matrix (Fin d) (Fin d) K}
    (hQ : PreservesDot Q) (hi : LeftInv Q Qi) {ga gm : Fin d → K} (hca : Compat Q L.ga ga)
    (hcm : Compat Q L.gm gm) (h : St K T d) :
    layerFn E (streamGauge L Q Qi ga gm) (mapSt Q h) = mapSt Q (layerFn E L h) := by
  funext t
  show afterAttn E (streamGauge L Q Qi ga gm) (mapSt Q h) t
        + mlpOut E (streamGauge L Q Qi ga gm) (nrm E.F gm (afterAttn E (streamGauge L Q Qi ga gm) (mapSt Q h) t))
      = Q *ᵥ (afterAttn E L h t + mlpOut E L (nrm E.F L.gm (afterAttn E L h t)))
  rw [afterAttn_stream E L hQ hi hca h t, mlpOut_stream E L hQ hi hcm (afterAttn E L h t), Matrix.mulVec_add]

/-- The transformation of a whole layer, for a transformation `ν` of the norm weights. -/
def streamLayer (Q Qi : Matrix (Fin d) (Fin d) K) (ν : (Fin d → K) → (Fin d → K)) (L : Layer K d hv nu Hh Gr P) :
    Layer K d hv nu Hh Gr P :=
  streamGauge L Q Qi (ν L.ga) (ν L.gm)

def Model.streamGauge (M : Model K d hv nu Hh Gr P V) (Q Qi : Matrix (Fin d) (Fin d) K) (ν : (Fin d → K) → (Fin d → K)) :
    Model K d hv nu Hh Gr P V where
  emb := fun v => Q *ᵥ M.emb v
  head := M.head * Qi
  gf := ν M.gf
  layers := M.layers.map (streamLayer Q Qi ν)

theorem runLayers_stream (E : Env K T Hh Gr P) {Q Qi : Matrix (Fin d) (Fin d) K} (hQ : PreservesDot Q)
    (hi : LeftInv Q Qi) (ν : (Fin d → K) → (Fin d → K)) :
    ∀ (Ls : List (Layer K d hv nu Hh Gr P)),
      (∀ L ∈ Ls, Compat Q L.ga (ν L.ga) ∧ Compat Q L.gm (ν L.gm)) →
      ∀ h : St K T d, runLayers E (Ls.map (streamLayer Q Qi ν)) (mapSt Q h) = mapSt Q (runLayers E Ls h) := by
  intro Ls
  induction Ls with
  | nil => intro _ h; rfl
  | cons L Ls ih =>
    intro hν h
    have hL := hν L (by simp)
    have hν' : ∀ L' ∈ Ls, Compat Q L'.ga (ν L'.ga) ∧ Compat Q L'.gm (ν L'.gm) :=
      fun L' hL' => hν L' (by simp [hL'])
    have e : layerFn E (streamLayer Q Qi ν L) (mapSt Q h) = mapSt Q (layerFn E L h) :=
      layerFn_stream E L hQ hi hL.1 hL.2 h
    show runLayers E (Ls.map (streamLayer Q Qi ν)) (layerFn E (streamLayer Q Qi ν L) (mapSt Q h))
      = mapSt Q (runLayers E Ls (layerFn E L h))
    rw [e]
    exact ih hν' (layerFn E L h)

/-- The logits are invariant under the gauge of the stream, for every environment (every nonlinearity), when the norm weights are compatible. -/
theorem logits_stream (E : Env K T Hh Gr P) (M : Model K d hv nu Hh Gr P V) (toks : Fin T → V)
    {Q Qi : Matrix (Fin d) (Fin d) K} (hQ : PreservesDot Q) (hi : LeftInv Q Qi) (ν : (Fin d → K) → (Fin d → K))
    (hν : ∀ L ∈ M.layers, Compat Q L.ga (ν L.ga) ∧ Compat Q L.gm (ν L.gm)) (hf : Compat Q M.gf (ν M.gf)) :
    logits E (M.streamGauge Q Qi ν) toks = logits E M toks := by
  funext t
  have hr : runLayers E (M.layers.map (streamLayer Q Qi ν)) (fun t => Q *ᵥ M.emb (toks t)) t
      = Q *ᵥ runLayers E M.layers (fun t => M.emb (toks t)) t :=
    congrFun (runLayers_stream E hQ hi ν M.layers hν (fun t => M.emb (toks t))) t
  calc logits E (M.streamGauge Q Qi ν) toks t
      = (M.head * Qi) *ᵥ nrm E.F (ν M.gf)
          (runLayers E (M.layers.map (streamLayer Q Qi ν)) (fun t => Q *ᵥ M.emb (toks t)) t) := rfl
    _ = (M.head * Qi) *ᵥ nrm E.F (ν M.gf) (Q *ᵥ runLayers E M.layers (fun t => M.emb (toks t)) t) := by rw [hr]
    _ = (M.head * Qi) *ᵥ (Q *ᵥ nrm E.F M.gf (runLayers E M.layers (fun t => M.emb (toks t)) t)) := by
        rw [nrm_stream E.F hQ hf]
    _ = M.head *ᵥ nrm E.F M.gf (runLayers E M.layers (fun t => M.emb (toks t)) t) := reader_stream M.head hi _
    _ = logits E M toks t := rfl

end

end Lmnf
