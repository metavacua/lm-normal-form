/-
SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
SPDX-License-Identifier: AGPL-3.0-or-later
-/
import Mathlib
import LmnfProofs.Norm

/-!
# A Llama-style decoder, over an arbitrary field, with arbitrary nonlinearities

The definitions of the forward pass of `batches/0015/model.py` in the form that a proof wants. The heads and the key/value groups are arbitrary finite types with a map from the heads to
the groups; the head dimension of the queries and keys is a finite type `P` of rotary planes with a `2 × 2` block each; the rotation of every position and plane is given; and the three
functions that make the model nonlinear (the exponential of the attention, the gate of the feed-forward block, the scale of the normalization) are *variables* (`Fns`). Nothing is said
about their values, so a theorem about this model is a theorem for every nonlinearity. The cache of a layer is the triple `qv`, `kv`, `vv` (queries, keys and values after the rotation
where there is one).

The correspondence with `model.py` (the flattened layout of Hugging Face's Llama, in which the queries of all heads are one matrix) is by reading; it is tested, in exact arithmetic,
by a later batch.
-/

namespace Lmnf

open Matrix

/-- The three nonlinear functions of the model, arbitrary. -/
structure Fns (K : Type*) where
  expo : K → K
  silu : K → K
  rs : K → K

/-- One layer: the projections of the attention block (per head and per plane for the queries and keys, per group for the keys and values, per head for the output), the gated
feed-forward block, the weights of the two norms. -/
structure Layer (K : Type*) (d hv nu : ℕ) (Hh Gr P : Type*) where
  Wq : Hh → P → Matrix (Fin 2) (Fin d) K
  Wk : Gr → P → Matrix (Fin 2) (Fin d) K
  Wv : Gr → Matrix (Fin hv) (Fin d) K
  Wo : Hh → Matrix (Fin d) (Fin hv) K
  Wg : Matrix (Fin nu) (Fin d) K
  Wu : Matrix (Fin nu) (Fin d) K
  Wd : Matrix (Fin d) (Fin nu) K
  ga : Fin d → K
  gm : Fin d → K

/-- What is not a parameter: the nonlinearities, the group of each head, the rotation of each rotary plane at each position, the scale of the scores. -/
structure Env (K : Type*) (T : ℕ) (Hh Gr P : Type*) where
  F : Fns K
  gr : Hh → Gr
  rot : Fin T → P → Matrix (Fin 2) (Fin 2) K
  c : K

/-- The states of the positions: one vector of the hidden dimension for each. -/
abbrev St (K : Type*) (T d : ℕ) := Fin T → Fin d → K

/-- A model: the embedding of each token, the head (one row per token of the vocabulary), the weight of the final norm, the layers. -/
structure Model (K : Type*) (d hv nu : ℕ) (Hh Gr P V : Type*) where
  emb : V → Fin d → K
  head : Matrix V (Fin d) K
  gf : Fin d → K
  layers : List (Layer K d hv nu Hh Gr P)

section Defs

variable {K : Type*} [Field K] {T d hv nu : ℕ} {Hh Gr P V : Type*} [Fintype Hh] [Fintype Gr] [Fintype P]

/-- Normalization, then the weight. -/
def nrm (F : Fns K) (γ x : Fin d → K) (i : Fin d) : K := γ i * rms F.rs x i

def qv (E : Env K T Hh Gr P) (L : Layer K d hv nu Hh Gr P) (h : St K T d) (t : Fin T) (hd : Hh) (p : P) : Fin 2 → K :=
  E.rot t p *ᵥ (L.Wq hd p *ᵥ nrm E.F L.ga (h t))

def kv (E : Env K T Hh Gr P) (L : Layer K d hv nu Hh Gr P) (h : St K T d) (s : Fin T) (g : Gr) (p : P) : Fin 2 → K :=
  E.rot s p *ᵥ (L.Wk g p *ᵥ nrm E.F L.ga (h s))

def vv (E : Env K T Hh Gr P) (L : Layer K d hv nu Hh Gr P) (h : St K T d) (s : Fin T) (g : Gr) : Fin hv → K :=
  L.Wv g *ᵥ nrm E.F L.ga (h s)

def score (E : Env K T Hh Gr P) (L : Layer K d hv nu Hh Gr P) (h : St K T d) (t s : Fin T) (hd : Hh) : K :=
  E.c * ∑ p, qv E L h t hd p ⬝ᵥ kv E L h s (E.gr hd) p

/-- The weight of position `s` for the query at `t`: the exponential of the score, zero after `t` (the causal mask). -/
def wt (E : Env K T Hh Gr P) (L : Layer K d hv nu Hh Gr P) (h : St K T d) (t s : Fin T) (hd : Hh) : K :=
  if s ≤ t then E.F.expo (score E L h t s hd) else 0

/-- The attention weights: the normalized weights. -/
def att (E : Env K T Hh Gr P) (L : Layer K d hv nu Hh Gr P) (h : St K T d) (t s : Fin T) (hd : Hh) : K :=
  wt E L h t s hd / ∑ r, wt E L h t r hd

def attnOut (E : Env K T Hh Gr P) (L : Layer K d hv nu Hh Gr P) (h : St K T d) (t : Fin T) : Fin d → K :=
  ∑ hd, L.Wo hd *ᵥ (∑ s, att E L h t s hd • vv E L h s (E.gr hd))

def mlpOut (E : Env K T Hh Gr P) (L : Layer K d hv nu Hh Gr P) (x : Fin d → K) : Fin d → K :=
  L.Wd *ᵥ (fun i => E.F.silu ((L.Wg *ᵥ x) i) * (L.Wu *ᵥ x) i)

def afterAttn (E : Env K T Hh Gr P) (L : Layer K d hv nu Hh Gr P) (h : St K T d) (t : Fin T) : Fin d → K :=
  h t + attnOut E L h t

def layerFn (E : Env K T Hh Gr P) (L : Layer K d hv nu Hh Gr P) (h : St K T d) (t : Fin T) : Fin d → K :=
  afterAttn E L h t + mlpOut E L (nrm E.F L.gm (afterAttn E L h t))

def runLayers (E : Env K T Hh Gr P) (Ls : List (Layer K d hv nu Hh Gr P)) (h : St K T d) : St K T d :=
  Ls.foldl (fun h L => layerFn E L h) h

/-- The logits of position `t` for the tokens `toks`. -/
def logits (E : Env K T Hh Gr P) (M : Model K d hv nu Hh Gr P V) (toks : Fin T → V) (t : Fin T) : V → K :=
  M.head *ᵥ nrm E.F M.gf (runLayers E M.layers (fun t => M.emb (toks t)) t)

/-- If every layer of one model has the same function as the corresponding layer of another, and the embedding, the head and the final weight are the same, the logits are the same. -/
theorem runLayers_forall₂ (E : Env K T Hh Gr P) {R : Layer K d hv nu Hh Gr P → Layer K d hv nu Hh Gr P → Prop}
    (hR : ∀ L L', R L L' → ∀ h, layerFn E L' h = layerFn E L h) {Ls Ls' : List (Layer K d hv nu Hh Gr P)}
    (hF : List.Forall₂ R Ls Ls') : ∀ h : St K T d, runLayers E Ls' h = runLayers E Ls h := by
  induction hF with
  | nil => intro h; rfl
  | cons hr _ ih =>
    intro h
    simp only [runLayers, List.foldl_cons]
    rw [hR _ _ hr h]
    exact ih _

theorem logits_of_forall₂ (E : Env K T Hh Gr P) (M M' : Model K d hv nu Hh Gr P V)
    (hemb : M'.emb = M.emb) (hhead : M'.head = M.head) (hgf : M'.gf = M.gf)
    {R : Layer K d hv nu Hh Gr P → Layer K d hv nu Hh Gr P → Prop}
    (hR : ∀ L L', R L L' → ∀ h, layerFn E L' h = layerFn E L h)
    (hL : List.Forall₂ R M.layers M'.layers) (toks : Fin T → V) :
    logits E M' toks = logits E M toks := by
  funext t
  show M'.head *ᵥ nrm E.F M'.gf (runLayers E M'.layers (fun t => M'.emb (toks t)) t)
      = M.head *ᵥ nrm E.F M.gf (runLayers E M.layers (fun t => M.emb (toks t)) t)
  rw [hemb, hhead, hgf, runLayers_forall₂ E hR hL]

end Defs

end Lmnf
