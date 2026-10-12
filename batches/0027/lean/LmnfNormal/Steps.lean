/-
SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
SPDX-License-Identifier: AGPL-3.0-or-later
-/
import Mathlib
import LmnfNormal.Canon
import LmnfNormal.Rewriting
import LmnfNormal.TupleNF

/-!
# The oriented rewriting system whose normal forms are the sorted tuples of normalized items

`Step key norm` has two kinds of step on a tuple `f : Fin n → U`:

* `scale`: replace the item `f i` by `norm (f i)` when this changes it;
* `swap`: exchange the items at positions `i < j` when the key of the later item is smaller.

A tuple is irreducible exactly when all its items are normalized and the keys are monotone
(`isNF_iff`).  The map `NF key norm` of `LmnfNormal.TupleNF` is constant along the steps and fixes
the irreducible tuples, so the system has unique normal forms (`step_unr`) and the normal form
reached from `f` is `NF key norm f` (`unique_normal_form`), by `eq_of_reach_nf` of
`LmnfNormal.Rewriting`, without any critical-pair analysis.
-/

namespace LmnfNormal

variable {n : ℕ} {U Γ β : Type*}

/-- The steps: normalize one item, or swap two items that are out of order. -/
inductive Step [LinearOrder β] (key : U → β) (norm : U → U) :
    (Fin n → U) → (Fin n → U) → Prop
  | scale (f : Fin n → U) (i : Fin n) (h : norm (f i) ≠ f i) :
      Step key norm f (Function.update f i (norm (f i)))
  | swap (f : Fin n → U) (i j : Fin n) (hij : i < j) (h : key (f j) < key (f i)) :
      Step key norm f (f ∘ Equiv.swap i j)

/-- A tuple is irreducible exactly when its items are normalized and its keys are monotone. -/
theorem isNF_iff [LinearOrder β] {key : U → β} {norm : U → U} (f : Fin n → U) :
    IsNF (Step key norm) f ↔ (∀ i, norm (f i) = f i) ∧ Monotone (key ∘ f) := by
  unfold IsNF
  constructor
  · intro h
    refine ⟨fun i => ?_, ?_⟩
    · by_contra hi
      exact h ⟨_, Step.scale f i hi⟩
    · refine monotone_iff_forall_lt.mpr ?_
      intro i j hij
      by_contra hlt
      exact h ⟨_, Step.swap f i j hij (not_le.mp hlt)⟩
  · rintro ⟨h1, h2⟩ ⟨g, hg⟩
    cases hg with
    | scale i hi => exact hi (h1 i)
    | swap i j hij hlt => exact (not_le.mpr hlt) (h2 hij.le)

/-- Normalizing one item does not change the normalized items of the tuple. -/
theorem norm_comp_update {act : Γ → U → U} {norm : U → U} (hn : ItemNormalizer act norm)
    (f : Fin n → U) (i : Fin n) :
    norm ∘ Function.update f i (norm (f i)) = norm ∘ f := by
  funext j
  change norm (Function.update f i (norm (f i)) j) = norm (f j)
  by_cases hj : j = i
  · rw [hj, Function.update_self]
    exact hn.norm_idem _
  · rw [Function.update_of_ne hj]

/-- The normal form is constant along a step. -/
theorem NF_step [LinearOrder β] {key : U → β} (hkey : Function.Injective key)
    {act : Γ → U → U} {norm : U → U} (hn : ItemNormalizer act norm)
    {f g : Fin n → U} (h : Step key norm f g) : NF key norm f = NF key norm g := by
  cases h with
  | scale i _ =>
    exact Eq.symm (NF_eq_of_norm_comp_eq key norm (norm_comp_update hn _ i))
  | swap i j _ _ => exact Eq.symm (NF_perm hkey _ (Equiv.swap i j))

/-- An irreducible tuple is its own normal form. -/
theorem NF_of_isNF [LinearOrder β] {key : U → β} (_hkey : Function.Injective key)
    {act : Γ → U → U} {norm : U → U} (_hn : ItemNormalizer act norm)
    {f : Fin n → U} (h : IsNF (Step key norm) f) : NF key norm f = f := by
  obtain ⟨h1, h2⟩ := (isNF_iff (key := key) (norm := norm) f).1 h
  have e : norm ∘ f = f := funext h1
  have hs : Tuple.sort (key ∘ f) = Equiv.refl _ := Tuple.sort_eq_refl_iff_monotone.2 h2
  unfold NF
  rw [e, hs, Equiv.coe_refl, Function.comp_id]

/-- An irreducible tuple reached from `f` is `NF key norm f`. -/
theorem unique_normal_form [LinearOrder β] {key : U → β} (hkey : Function.Injective key)
    {act : Γ → U → U} {norm : U → U} (hn : ItemNormalizer act norm)
    {f g : Fin n → U} (hfg : Relation.ReflTransGen (Step key norm) f g)
    (hg : IsNF (Step key norm) g) : g = NF key norm f :=
  eq_of_reach_nf (Step key norm) (NF key norm) (fun _ _ h => NF_step hkey hn h)
    (fun _ h => NF_of_isNF hkey hn h) hfg hg

/-- The system `Step key norm` has unique normal forms. -/
theorem step_unr [LinearOrder β] {key : U → β} (hkey : Function.Injective key)
    {act : Γ → U → U} {norm : U → U} (hn : ItemNormalizer act norm) :
    UNr (Step (n := n) key norm) :=
  unr_of_invariant (Step key norm) (NF key norm) (fun _ _ h => NF_step hkey hn h)
    (fun _ h => NF_of_isNF hkey hn h)

end LmnfNormal
