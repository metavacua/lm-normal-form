/-
SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
SPDX-License-Identifier: AGPL-3.0-or-later
-/
import Mathlib

/-!
# From the blocks to the model

If a change of variables `φ` carries every block of a stack to the corresponding block of another stack (`f' (φ x) = φ (f x)`), carries the embedding, and the readout does not see it,
the two models compute the same function. This is the step that turns the equivariance of every block, which the lemmas of `Norm`, `Attention` and `Mlp` give for the operations of a block,
into the invariance of the logits. The residual connection preserves equivariance when `φ` is additive.
-/

namespace Lmnf

theorem stack_equivariant {α β : Type*} (φ : α → β) :
    ∀ (L : List (α → α)) (L' : List (β → β)),
      List.Forall₂ (fun f f' => ∀ x, f' (φ x) = φ (f x)) L L' →
      ∀ x, L'.foldl (fun a f => f a) (φ x) = φ (L.foldl (fun a f => f a) x) := by
  intro L L' h
  induction h with
  | nil => intro x; rfl
  | cons hf _ ih =>
    intro x
    simp only [List.foldl_cons]
    rw [hf x]
    exact ih _

theorem residual_equivariant {V W : Type*} [AddCommMonoid V] [AddCommMonoid W] (φ : V →+ W)
    (f : V → V) (f' : W → W) (hf : ∀ x, f' (φ x) = φ (f x)) :
    ∀ x, φ x + f' (φ x) = φ (x + f x) := by
  intro x
  rw [hf x, map_add]

theorem model_invariant {T α β Out : Type*} (embed : T → α) (embed' : T → β) (φ : α → β)
    (hE : ∀ t, embed' t = φ (embed t)) (blocks : List (α → α)) (blocks' : List (β → β))
    (hB : List.Forall₂ (fun f f' => ∀ x, f' (φ x) = φ (f x)) blocks blocks')
    (readout : α → Out) (readout' : β → Out) (hR : ∀ x, readout' (φ x) = readout x) (t : T) :
    readout' (blocks'.foldl (fun a f => f a) (embed' t))
      = readout (blocks.foldl (fun a f => f a) (embed t)) := by
  rw [hE t, stack_equivariant φ blocks blocks' hB (embed t), hR]

end Lmnf
