/-
SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
SPDX-License-Identifier: AGPL-3.0-or-later
-/
import Mathlib

/-!
# Canonical form

The definition of the page "Canonical form" (Wikipedia, section Definition).
A *canonicalization* with respect to an equivalence relation `R` on a set `S` is a map
`c : S → S` that is

* idempotent: `c (c s) = c s`,
* decisive: `R s₁ s₂` holds if and only if `c s₁ = c s₂`,
* representative: `R s (c s)`.

The three conditions are a structure `Canonicalization R c`; the relation `R` is then an
equivalence relation, and conversely an invariant representative map of an equivalence
relation is a canonicalization.  A canonicalization decides `R` when `S` has decidable equality.
-/

namespace LmnfNormal

/-- `c` is a canonicalization with respect to `R`: idempotent, decisive and representative. -/
structure Canonicalization {S : Type*} (R : S → S → Prop) (c : S → S) : Prop where
  /-- Canonicalizing twice is canonicalizing once. -/
  idempotent : ∀ s, c (c s) = c s
  /-- Two elements are related exactly when they have the same canonical form. -/
  decisive : ∀ s₁ s₂, R s₁ s₂ ↔ c s₁ = c s₂
  /-- Every element is related to its canonical form. -/
  representative : ∀ s, R s (c s)

variable {S : Type*} {R : S → S → Prop} {c : S → S}

/-- The relation of a canonicalization is an equivalence relation. -/
theorem Canonicalization.equivalence (h : Canonicalization R c) : Equivalence R :=
  ⟨fun s => (h.decisive s s).2 rfl,
    fun {x y} hxy => (h.decisive y x).2 ((h.decisive x y).1 hxy).symm,
    fun {x y z} hxy hyz =>
      (h.decisive x z).2 (((h.decisive x y).1 hxy).trans ((h.decisive y z).1 hyz))⟩

/-- A map that is constant on the classes of an equivalence relation `R` and sends every element
to a related one is a canonicalization. -/
theorem Canonicalization.of_invariant_representative (hR : Equivalence R)
    (hinv : ∀ s t, R s t → c s = c t) (hrep : ∀ s, R s (c s)) : Canonicalization R c :=
  ⟨fun s => (hinv s (c s) (hrep s)).symm,
    fun s t => ⟨hinv s t, fun h =>
      hR.trans (Eq.mp (congrArg (R s) h) (hrep s)) (hR.symm (hrep t))⟩,
    hrep⟩

/-- A canonicalization decides its relation when the carrier has decidable equality. -/
def Canonicalization.decidable [DecidableEq S] (h : Canonicalization R c) : DecidableRel R :=
  fun a b => decidable_of_iff' (c a = c b) (h.decisive a b)

end LmnfNormal
