/-
SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
SPDX-License-Identifier: AGPL-3.0-or-later
-/
import Mathlib
import LmnfNormal.TupleNF
import LmnfNormal.InvolutionNF
import LmnfNormal.Unit

/-!
# The theorems that the plan of `nf.dl` depends on

`nf.dl` derives, for each index space of a model, the group of changes of basis that act on it and
the method that canonicalizes the orbits of that group. The method names a theorem; the theorem is
in this file or in `Unit.lean`:

* `canon_Perm_sort`: the group `Perm`; the items are not changed, only their order; the normal form
  is the sort by an injective key.
* `canon_SP_sign_sort`: the group `SP` of signed permutations; the item is mapped to the one of the
  item and its negative that has the smaller key, then the items are sorted. The unit of a ternary
  block is the instance (`canon_SP_units`), with the sign gauge `(g, u, w) ↦ (g, -u, -w)` and its
  soundness (`sound_SP`).
* `canon_Mono_scale_sort` (in `Unit.lean`): the group `Mono`; the item is scaled to its pivot, then
  the items are sorted.

Not here, and so unknown to the plan: `canon_GL_rref` (the reduced row echelon form, checked
exhaustively on a finite model by `ref.py` and not proved), and `canon_stratified` (canonicalizing
the spaces one after another in the order of their distance from the vocabulary).
-/

namespace LmnfNormal

/-- `Perm`: the gauge is trivial; the normal form is the sort of the items by an injective key. -/
theorem canon_Perm_sort {U β : Type*} {n : ℕ} [LinearOrder β] {key : U → β}
    (hkey : Function.Injective key) :
    Canonicalization (TEquiv (n := n) (fun (_ : Unit) (x : U) => x)) (NF (n := n) key id) :=
  canonicalization hkey ⟨fun _ _ => rfl, fun x => ⟨(), rfl⟩⟩

/-- `SP`: the gauge is an involution of the items (a sign change); the normal form takes the
smaller of an item and its negative, then sorts. -/
theorem canon_SP_sign_sort {U β : Type*} {n : ℕ} [LinearOrder β] {key : U → β} {neg : U → U}
    (hneg : ∀ x, neg (neg x) = x) (hkey : Function.Injective key) :
    Canonicalization (TEquiv (n := n) (fun (b : Bool) (x : U) => if b then neg x else x))
      (NF (n := n) key (minNorm key neg)) :=
  canonicalization hkey (minNorm_itemNormalizer hneg hkey)

namespace GUnit

variable {K : Type*} [Field K] {d e : ℕ}

/-- The sign gauge of a gated unit: negate the up row and the down column. -/
def negUW (x : GUnit K d e) : GUnit K d e := ⟨x.g, -x.u, -x.w⟩

theorem negUW_negUW (x : GUnit K d e) : negUW (negUW x) = x := by
  ext <;> simp [negUW]

/-- The action of `Bool` on a unit by the sign gauge. -/
def signAct (b : Bool) (x : GUnit K d e) : GUnit K d e := if b then negUW x else x

/-- One unit contributes the same to the function after the sign gauge, whatever `s` is. -/
theorem contrib_signAct (s : K → K) (x : Fin d → K) (b : Bool) (u : GUnit K d e) :
    (s ((signAct b u).g ⬝ᵥ x) * ((signAct b u).u ⬝ᵥ x)) • (signAct b u).w
      = (s (u.g ⬝ᵥ x) * (u.u ⬝ᵥ x)) • u.w := by
  cases b
  · rfl
  · change (s (u.g ⬝ᵥ x) * ((-u.u) ⬝ᵥ x)) • (-u.w) = (s (u.g ⬝ᵥ x) * (u.u ⬝ᵥ x)) • u.w
    rw [neg_dotProduct, mul_neg, neg_smul_neg]

end GUnit

section Sign

variable {K : Type*} [Field K] {d e n : ℕ}

theorem GBlock.F_signMove (s : K → K) {θ θ' : Fin n → GUnit K d e}
    (h : Move (GUnit.signAct (K := K) (d := d) (e := e)) θ θ') : GBlock.F s θ = GBlock.F s θ' := by
  cases h with
  | perm σ => exact (GBlock.F_perm s θ σ).symm
  | gauge c =>
    funext x
    unfold GBlock.F
    refine Finset.sum_congr rfl ?_
    intro i _
    exact (GUnit.contrib_signAct s x (c i) (θ i)).symm

/-- Soundness of the sign gauge with the permutations: equivalent blocks have the same function,
whatever `s` is. -/
theorem sound_SP (s : K → K) {θ θ' : Fin n → GUnit K d e}
    (h : TEquiv (GUnit.signAct (K := K) (d := d) (e := e)) θ θ') : GBlock.F s θ = GBlock.F s θ' :=
  GUnit.tequiv_invariant (GBlock.F s) (fun _ _ hm => GBlock.F_signMove s hm) h

/-- The canonical form of a block of gated units under the sign gauge and the permutations. -/
theorem canon_SP_units [LinearOrder K] :
    Canonicalization (TEquiv (n := n) (GUnit.signAct (K := K) (d := d) (e := e)))
      (NF (n := n) GUnit.key (minNorm GUnit.key GUnit.negUW)) :=
  canon_SP_sign_sort (neg := GUnit.negUW) GUnit.negUW_negUW GUnit.key_injective

end Sign

end LmnfNormal
