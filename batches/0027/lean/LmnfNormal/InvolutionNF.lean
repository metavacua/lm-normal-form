/-
SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
SPDX-License-Identifier: AGPL-3.0-or-later
-/
import Mathlib
import LmnfNormal.Canon
import LmnfNormal.TupleNF

/-!
# The orbit minimum of an involution

Let `neg : U → U` be an involution, `neg (neg x) = x`, and let `key : U → β` be injective into a
linear order.  The group `Bool` acts on `U` by `b • x = if b then neg x else x`.  The map
`minNorm key neg x`, which takes the element of `{x, neg x}` of the smaller key, is an
`ItemNormalizer` for this action (`minNorm_itemNormalizer`).  Injectivity of `key` is used only for
the case `key (neg x) = key x`, in which `neg x = x`.
-/

namespace LmnfNormal

variable {U β : Type*}

/-- The element of `{x, neg x}` of the smaller key, `x` itself when the keys are equal. -/
def minNorm [LinearOrder β] (key : U → β) (neg : U → U) (x : U) : U :=
  if key (neg x) < key x then neg x else x

/-- If `neg x` has the smaller key, `minNorm` is `neg x`. -/
theorem minNorm_of_lt [LinearOrder β] {key : U → β} {neg : U → U} {x : U}
    (h : key (neg x) < key x) : minNorm key neg x = neg x := by
  unfold minNorm
  exact if_pos h

/-- If `neg x` does not have the smaller key, `minNorm` is `x`. -/
theorem minNorm_of_not_lt [LinearOrder β] {key : U → β} {neg : U → U} {x : U}
    (h : ¬ key (neg x) < key x) : minNorm key neg x = x := by
  unfold minNorm
  exact if_neg h

/-- `minNorm` is constant on `{x, neg x}`. -/
theorem minNorm_neg [LinearOrder β] {key : U → β} {neg : U → U} (hneg : ∀ x, neg (neg x) = x)
    (hkey : Function.Injective key) (x : U) : minNorm key neg (neg x) = minNorm key neg x := by
  by_cases h : key (neg x) < key x
  · have h' : ¬ key (neg (neg x)) < key (neg x) := by
      rw [hneg x]
      exact not_lt.mpr h.le
    rw [minNorm_of_not_lt (key := key) (neg := neg) (x := neg x) h',
      minNorm_of_lt (key := key) (neg := neg) (x := x) h]
  · by_cases h2 : key x < key (neg x)
    · have h2' : key (neg (neg x)) < key (neg x) := by
        rw [hneg x]
        exact h2
      rw [minNorm_of_lt (key := key) (neg := neg) (x := neg x) h2',
        minNorm_of_not_lt (key := key) (neg := neg) (x := x) h, hneg x]
    · have hx : neg x = x := hkey (le_antisymm (not_lt.mp h2) (not_lt.mp h))
      rw [hx]

/-- `minNorm` is an `ItemNormalizer` for the action of `Bool` by `neg`. -/
theorem minNorm_itemNormalizer [LinearOrder β] {key : U → β} {neg : U → U}
    (hneg : ∀ x, neg (neg x) = x) (hkey : Function.Injective key) :
    ItemNormalizer (fun (b : Bool) (x : U) => if b then neg x else x) (minNorm key neg) := by
  refine ⟨fun γ x => ?_, fun x => ?_⟩
  · cases γ with
    | false => rfl
    | true => exact minNorm_neg hneg hkey x
  · by_cases h : key (neg x) < key x
    · exact ⟨true, minNorm_of_lt (key := key) (neg := neg) (x := x) h⟩
    · exact ⟨false, minNorm_of_not_lt (key := key) (neg := neg) (x := x) h⟩

end LmnfNormal
