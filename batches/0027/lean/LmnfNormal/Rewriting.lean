/-
SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
SPDX-License-Identifier: AGPL-3.0-or-later
-/
import Mathlib

/-!
# Normal forms in abstract rewriting

The vocabulary of the page "Normal form (abstract rewriting)" (Wikipedia), for a binary relation
`r` (a reduction) on a type `α`:

* `IsNF r x`: `x` is a normal form, that is irreducible, there is no `y` with `r x y`;
* `WN r`: weak normalization, every element reduces to a normal form;
* `UNr r`: unique normal forms with respect to reduction (the page writes `UN→`), two normal forms
  reachable from the same element are equal;
* `Confluent r`: any two reducts of an element have a common reduct.

The page states that weak normalization and `UN→` imply confluence (`confluent_of_wn_un`).
The two workhorses are `eq_of_reach_nf` and `unr_of_invariant`: a map `c` that is constant along
the steps of `r` and fixes the normal forms gives unique normal forms, without any analysis of
critical pairs.
-/

namespace LmnfNormal

variable {α : Type*}

/-- `x` is a normal form: it is irreducible. -/
def IsNF (r : α → α → Prop) (x : α) : Prop := ¬ ∃ y, r x y

/-- Weak normalization: every element reduces to a normal form. -/
def WN (r : α → α → Prop) : Prop :=
  ∀ a, ∃ b, Relation.ReflTransGen r a b ∧ IsNF r b

/-- Unique normal forms with respect to reduction (`UN→`): two normal forms reachable from the same
element are equal. -/
def UNr (r : α → α → Prop) : Prop :=
  ∀ a b c, Relation.ReflTransGen r a b → Relation.ReflTransGen r a c → IsNF r b → IsNF r c →
    b = c

/-- Confluence: two reducts of one element have a common reduct. -/
def Confluent (r : α → α → Prop) : Prop :=
  ∀ a b c, Relation.ReflTransGen r a b → Relation.ReflTransGen r a c →
    ∃ d, Relation.ReflTransGen r b d ∧ Relation.ReflTransGen r c d

/-- Weak normalization and unique normal forms imply confluence. -/
theorem confluent_of_wn_un {r : α → α → Prop} (hwn : WN r) (hun : UNr r) : Confluent r := by
  intro a b c hab hac
  obtain ⟨b', hbb', hb'⟩ := hwn b
  obtain ⟨c', hcc', hc'⟩ := hwn c
  have h : b' = c' := hun a b' c' (hab.trans hbb') (hac.trans hcc') hb' hc'
  subst h
  exact ⟨_, hbb', hcc'⟩

/-- A map that is constant along the steps of `r` is constant along its reflexive transitive
closure. -/
theorem invariant_of_reflTransGen (r : α → α → Prop) (c : α → α)
    (hinv : ∀ a b, r a b → c a = c b) {a b : α} (hab : Relation.ReflTransGen r a b) :
    c a = c b := by
  induction hab with
  | refl => rfl
  | tail _ hbc ih => exact ih.trans (hinv _ _ hbc)

/-- If `c` is constant along the steps of `r` and fixes the normal forms, then a normal form
reachable from `a` is `c a`. -/
theorem eq_of_reach_nf (r : α → α → Prop) (c : α → α)
    (hinv : ∀ a b, r a b → c a = c b) (hfix : ∀ a, IsNF r a → c a = a) {a b : α}
    (hab : Relation.ReflTransGen r a b) (hb : IsNF r b) : b = c a :=
  ((hfix b hb).symm).trans (invariant_of_reflTransGen r c hinv hab).symm

/-- If `c` is constant along the steps of `r` and fixes the normal forms, then `r` has unique
normal forms. -/
theorem unr_of_invariant (r : α → α → Prop) (c : α → α)
    (hinv : ∀ a b, r a b → c a = c b) (hfix : ∀ a, IsNF r a → c a = a) : UNr r := by
  intro a b d hab had hb hd
  exact (eq_of_reach_nf r c hinv hfix hab hb).trans (eq_of_reach_nf r c hinv hfix had hd).symm

end LmnfNormal
