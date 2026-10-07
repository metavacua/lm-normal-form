/-
SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
SPDX-License-Identifier: AGPL-3.0-or-later
-/
import Mathlib
import LmnfProofs.GaugeOV
import LmnfProofs.GaugeQK
import LmnfProofs.GaugeHead
import LmnfProofs.GaugeMlp
import LmnfProofs.GaugeNorm
import LmnfProofs.GaugeStream

/-! All the gauges that leave the function of a layer unchanged, composed in any order, leave the logits of the model unchanged (`logits_gauged`); and the table of what each gauge does to the
stream and to the cache (`table_*`): the stream gauge moves the stream and leaves the cache, the value/output, query/key and head gauges move the cache and leave the stream, the feed-forward
and norm-weight gauges move neither, a composition of the stream gauge and the value gauge moves both (`both_moved`). -/

namespace Lmnf

open Matrix

section

variable {K : Type*} [Field K] {T d hv nu : ℕ} {Hh Gr P V : Type*} [Fintype Hh] [Fintype Gr] [Fintype P]

/-- The layers that can be reached from a layer by a sequence of the gauges that leave the function of the layer unchanged. -/
inductive Gauged (E : Env K T Hh Gr P) : Layer K d hv nu Hh Gr P → Layer K d hv nu Hh Gr P → Prop
  | refl (L : Layer K d hv nu Hh Gr P) : Gauged E L L
  | trans {L L' L'' : Layer K d hv nu Hh Gr P} : Gauged E L L' → Gauged E L' L'' → Gauged E L L''
  | viaOv (L : Layer K d hv nu Hh Gr P) (A B : Gr → Matrix (Fin hv) (Fin hv) K) (hBA : ∀ g, B g * A g = 1) :
      Gauged E L (ovGauge E L A B)
  | viaQk (L : Layer K d hv nu Hh Gr P) (a b : Gr → P → K) (hnz : ∀ g p, a g p ^ 2 + b g p ^ 2 ≠ 0)
      (hrot : ∀ t p, ∃ c s, E.rot t p = cplx c s) : Gauged E L (qkGauge E L a b)
  | viaHead (L : Layer K d hv nu Hh Gr P) (σ : Equiv.Perm Hh) (τ : Equiv.Perm Gr)
      (hστ : ∀ hd, E.gr (σ hd) = τ (E.gr hd)) : Gauged E L (headPerm L σ τ)
  | viaMlpScale (L : Layer K d hv nu Hh Gr P) (c : Fin nu → K) (hc : ∀ i, c i ≠ 0) : Gauged E L (mlpScale L c)
  | viaUnitPerm (L : Layer K d hv nu Hh Gr P) (σ : Equiv.Perm (Fin nu)) : Gauged E L (unitPerm L σ)
  | viaAttnNorm (L : Layer K d hv nu Hh Gr P) (c : Fin d → K) (hc : ∀ i, c i ≠ 0) : Gauged E L (attnNorm L c)
  | viaMlpNorm (L : Layer K d hv nu Hh Gr P) (c : Fin d → K) (hc : ∀ i, c i ≠ 0) : Gauged E L (mlpNorm L c)

theorem layerFn_gauged {E : Env K T Hh Gr P} {L L' : Layer K d hv nu Hh Gr P} (hG : Gauged E L L') :
    ∀ h : St K T d, layerFn E L' h = layerFn E L h := by
  induction hG with
  | refl L => intro h; rfl
  | trans _ _ ih1 ih2 => intro h; rw [ih2 h, ih1 h]
  | viaOv L A B hBA => exact layerFn_ov E L A B hBA
  | viaQk L a b hnz hrot => exact layerFn_qk E L a b hnz hrot
  | viaHead L σ τ hστ => exact layerFn_headPerm E L σ τ hστ
  | viaMlpScale L c hc => exact layerFn_mlpScale E L c hc
  | viaUnitPerm L σ => exact layerFn_unitPerm E L σ
  | viaAttnNorm L c hc => exact layerFn_attnNorm E L c hc
  | viaMlpNorm L c hc => exact layerFn_mlpNorm E L c hc

/-- The logits of a model are unchanged when every layer is replaced by a layer reached from it by any sequence of the gauges (with parameters chosen independently for every layer). -/
theorem logits_gauged (E : Env K T Hh Gr P) (M M' : Model K d hv nu Hh Gr P V)
    (hemb : M'.emb = M.emb) (hhead : M'.head = M.head) (hgf : M'.gf = M.gf)
    (hL : List.Forall₂ (Gauged E) M.layers M'.layers) (toks : Fin T → V) :
    logits E M' toks = logits E M toks :=
  logits_of_forall₂ E M M' hemb hhead hgf (fun _ _ hG => layerFn_gauged hG) hL toks

/-- The stream gauge: the stream moves (equivariantly), the cache does not. -/
theorem table_stream (E : Env K T Hh Gr P) (L : Layer K d hv nu Hh Gr P) {Q Qi : Matrix (Fin d) (Fin d) K}
    (hQ : PreservesDot Q) (hi : LeftInv Q Qi) {ga gm : Fin d → K} (hca : Compat Q L.ga ga)
    (hcm : Compat Q L.gm gm) (h : St K T d) :
    layerFn E (streamGauge L Q Qi ga gm) (mapSt Q h) = mapSt Q (layerFn E L h) ∧
    (∀ t hd p, qv E (streamGauge L Q Qi ga gm) (mapSt Q h) t hd p = qv E L h t hd p) ∧
    (∀ s g p, kv E (streamGauge L Q Qi ga gm) (mapSt Q h) s g p = kv E L h s g p) ∧
    (∀ s g, vv E (streamGauge L Q Qi ga gm) (mapSt Q h) s g = vv E L h s g) :=
  ⟨layerFn_stream E L hQ hi hca hcm h, fun t hd p => qv_stream E L hQ hi hca h t hd p,
    fun s g p => kv_stream E L hQ hi hca h s g p, fun s g => vv_stream E L hQ hi hca h s g⟩

/-- The value/output gauge: the stream does not move, the values move by `A`, the queries and keys do not. -/
theorem table_ov (E : Env K T Hh Gr P) (L : Layer K d hv nu Hh Gr P) (A B : Gr → Matrix (Fin hv) (Fin hv) K)
    (hBA : ∀ g, B g * A g = 1) (h : St K T d) :
    layerFn E (ovGauge E L A B) h = layerFn E L h ∧
    (∀ t hd p, qv E (ovGauge E L A B) h t hd p = qv E L h t hd p) ∧
    (∀ s g p, kv E (ovGauge E L A B) h s g p = kv E L h s g p) ∧
    (∀ s g, vv E (ovGauge E L A B) h s g = A g *ᵥ vv E L h s g) :=
  ⟨layerFn_ov E L A B hBA h, qv_ov E L A B h, kv_ov E L A B h, vv_ov E L A B h⟩

/-- The query/key gauge: the stream does not move, the keys move by the complex scalar, the queries by its inverse transpose, the values do not. -/
theorem table_qk (E : Env K T Hh Gr P) (L : Layer K d hv nu Hh Gr P) (a b : Gr → P → K)
    (hnz : ∀ g p, a g p ^ 2 + b g p ^ 2 ≠ 0) (hrot : ∀ t p, ∃ c s, E.rot t p = cplx c s) (h : St K T d) :
    layerFn E (qkGauge E L a b) h = layerFn E L h ∧
    (∀ t hd p, qv E (qkGauge E L a b) h t hd p
      = ((a (E.gr hd) p ^ 2 + b (E.gr hd) p ^ 2)⁻¹ • cplx (a (E.gr hd) p) (b (E.gr hd) p)) *ᵥ qv E L h t hd p) ∧
    (∀ s g p, kv E (qkGauge E L a b) h s g p = cplx (a g p) (b g p) *ᵥ kv E L h s g p) ∧
    (∀ s g, vv E (qkGauge E L a b) h s g = vv E L h s g) :=
  ⟨layerFn_qk E L a b hnz hrot h, qv_qk E L a b hrot h, kv_qk E L a b hrot h, vv_qk E L a b h⟩

/-- The permutation of heads and groups: the stream does not move, the cache is permuted. -/
theorem table_head (E : Env K T Hh Gr P) (L : Layer K d hv nu Hh Gr P) (σ : Equiv.Perm Hh) (τ : Equiv.Perm Gr)
    (hστ : ∀ hd, E.gr (σ hd) = τ (E.gr hd)) (h : St K T d) :
    layerFn E (headPerm L σ τ) h = layerFn E L h ∧
    (∀ t hd p, qv E (headPerm L σ τ) h t hd p = qv E L h t (σ hd) p) ∧
    (∀ s g p, kv E (headPerm L σ τ) h s g p = kv E L h s (τ g) p) ∧
    (∀ s g, vv E (headPerm L σ τ) h s g = vv E L h s (τ g)) :=
  ⟨layerFn_headPerm E L σ τ hστ h, qv_head E L σ τ h, kv_head E L σ τ h, vv_head E L σ τ h⟩

/-- The scaling of the up branch: neither the stream nor the cache moves. -/
theorem table_mlpScale (E : Env K T Hh Gr P) (L : Layer K d hv nu Hh Gr P) (c : Fin nu → K) (hc : ∀ i, c i ≠ 0)
    (h : St K T d) :
    layerFn E (mlpScale L c) h = layerFn E L h ∧
    (∀ t hd p, qv E (mlpScale L c) h t hd p = qv E L h t hd p) ∧
    (∀ s g p, kv E (mlpScale L c) h s g p = kv E L h s g p) ∧
    (∀ s g, vv E (mlpScale L c) h s g = vv E L h s g) :=
  ⟨layerFn_mlpScale E L c hc h, fun _ _ _ => rfl, fun _ _ _ => rfl, fun _ _ => rfl⟩

/-- The permutation of the units: neither the stream nor the cache moves. -/
theorem table_unitPerm (E : Env K T Hh Gr P) (L : Layer K d hv nu Hh Gr P) (σ : Equiv.Perm (Fin nu)) (h : St K T d) :
    layerFn E (unitPerm L σ) h = layerFn E L h ∧
    (∀ t hd p, qv E (unitPerm L σ) h t hd p = qv E L h t hd p) ∧
    (∀ s g p, kv E (unitPerm L σ) h s g p = kv E L h s g p) ∧
    (∀ s g, vv E (unitPerm L σ) h s g = vv E L h s g) :=
  ⟨layerFn_unitPerm E L σ h, fun _ _ _ => rfl, fun _ _ _ => rfl, fun _ _ => rfl⟩

/-- The scaling of the weight of the attention norm: neither the stream nor the cache moves (what the readers see is multiplied by `c`, and they compensate). -/
theorem table_attnNorm (E : Env K T Hh Gr P) (L : Layer K d hv nu Hh Gr P) (c : Fin d → K) (hc : ∀ i, c i ≠ 0)
    (h : St K T d) :
    layerFn E (attnNorm L c) h = layerFn E L h ∧
    (∀ t hd p, qv E (attnNorm L c) h t hd p = qv E L h t hd p) ∧
    (∀ s g p, kv E (attnNorm L c) h s g p = kv E L h s g p) ∧
    (∀ s g, vv E (attnNorm L c) h s g = vv E L h s g) :=
  ⟨layerFn_attnNorm E L c hc h, qv_attnNorm E L c hc h, kv_attnNorm E L c hc h, vv_attnNorm E L c hc h⟩

/-- The scaling of the weight of the feed-forward norm: neither the stream nor the cache moves. -/
theorem table_mlpNorm (E : Env K T Hh Gr P) (L : Layer K d hv nu Hh Gr P) (c : Fin d → K) (hc : ∀ i, c i ≠ 0)
    (h : St K T d) :
    layerFn E (mlpNorm L c) h = layerFn E L h ∧
    (∀ t hd p, qv E (mlpNorm L c) h t hd p = qv E L h t hd p) ∧
    (∀ s g p, kv E (mlpNorm L c) h s g p = kv E L h s g p) ∧
    (∀ s g, vv E (mlpNorm L c) h s g = vv E L h s g) :=
  ⟨layerFn_mlpNorm E L c hc h, fun _ _ _ => rfl, fun _ _ _ => rfl, fun _ _ => rfl⟩

/-- A composition of the stream gauge and the value/output gauge moves both the stream and the values, and still leaves the keys (and the function of the layer, up to the stream gauge) alone. -/
theorem both_moved (E : Env K T Hh Gr P) (L : Layer K d hv nu Hh Gr P) {Q Qi : Matrix (Fin d) (Fin d) K}
    (hQ : PreservesDot Q) (hi : LeftInv Q Qi) {ga gm : Fin d → K} (hca : Compat Q L.ga ga)
    (hcm : Compat Q L.gm gm) (A B : Gr → Matrix (Fin hv) (Fin hv) K) (hBA : ∀ g, B g * A g = 1) (h : St K T d) :
    layerFn E (ovGauge E (streamGauge L Q Qi ga gm) A B) (mapSt Q h) = mapSt Q (layerFn E L h) ∧
    (∀ s g, vv E (ovGauge E (streamGauge L Q Qi ga gm) A B) (mapSt Q h) s g = A g *ᵥ vv E L h s g) ∧
    (∀ s g p, kv E (ovGauge E (streamGauge L Q Qi ga gm) A B) (mapSt Q h) s g p = kv E L h s g p) := by
  refine ⟨?_, ?_, ?_⟩
  · rw [layerFn_ov E (streamGauge L Q Qi ga gm) A B hBA (mapSt Q h)]
    exact layerFn_stream E L hQ hi hca hcm h
  · intro s g
    rw [vv_ov E (streamGauge L Q Qi ga gm) A B (mapSt Q h) s g, vv_stream E L hQ hi hca h s g]
  · intro s g p
    rw [kv_ov E (streamGauge L Q Qi ga gm) A B (mapSt Q h) s g p, kv_stream E L hQ hi hca h s g p]

end

end Lmnf
