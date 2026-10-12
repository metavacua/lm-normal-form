/-
SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
SPDX-License-Identifier: AGPL-3.0-or-later
-/
import Mathlib
import LmnfNormal.TupleNF
import LmnfNormal.InvolutionNF
import LmnfNormal.Unit

/-!
# Three equivalences that are not one

The sources keep three notions apart for a block of gated units `(g, u, w)` over `ℚ` with `d = e =
1`:

* the *tensor-level identity* (`bagOf`): the three tensors `G`, `U`, `W` of the block, as a bag;
  names and the order of the units are not part of it;
* the *orbit of the gauge group* (`TEquiv GUnit.act`): permutations of the units and a scalar `c` on
  the up row with `c⁻¹` on the down column;
* the *equality of the function* (`GBlock.F s`).

None of the three implies another.  The theorems of this file are examples over `ℚ`:

* `sep_bag_not_function`: the same bag of tensors, two different functions (for `s t = t ^ 2`:
  exchange the gate and the up tensor);
* `sep_function_not_bag`: the same function, for every `s`, two different bags (a unit and its image
  under the gauge);
* `sep_orbit_not_function`: the same function, two blocks that are not in one orbit (for `s t = max
  t 0`, because `max t 0 - max (-t) 0 = t`, a pair `(g, u, w), (-g, u, -w)` is symmetric in the gate
  and the up row, but the gates are an invariant of the orbit and they differ).

The bag `bagOf` is *coarser* than a real digest of the tensors: it keeps for each of the three
tensors only the multiset of its entries and ignores the positions inside a tensor.  A coarser bag
identifies more blocks, so the separation `sep_bag_not_function` is only stronger for it.
-/

namespace LmnfNormal

/-- The bag of the three tensors of a block of units with `d = e = 1`: the multiset of the three
multisets of entries of `G` (the gates), `U` (the up rows) and `W` (the down columns).  It is
coarser than the real digest: it ignores the positions inside a tensor, so it identifies more
blocks. -/
def bagOf {n : ℕ} (θ : Fin n → GUnit ℚ 1 1) : Multiset (Multiset ℚ) :=
  {Multiset.map (fun i => (θ i).g 0) (Finset.univ : Finset (Fin n)).val,
    Multiset.map (fun i => (θ i).u 0) (Finset.univ : Finset (Fin n)).val,
    Multiset.map (fun i => (θ i).w 0) (Finset.univ : Finset (Fin n)).val}

namespace Separ

/-- A unit with `d = e = 1`, from its three entries. -/
def mk11 (g u w : ℚ) : GUnit ℚ 1 1 :=
  ⟨fun _ => g, fun _ => u, fun _ => w⟩

theorem mk11_g (g u w : ℚ) : (mk11 g u w).g 0 = g := rfl

theorem mk11_u (g u w : ℚ) : (mk11 g u w).u 0 = u := rfl

theorem mk11_w (g u w : ℚ) : (mk11 g u w).w 0 = w := rfl

/-- The scalar `2`, as a unit of `ℚ`. -/
def twoU : ℚˣ := Units.mk0 2 (by norm_num)

theorem twoU_val : ((twoU : ℚˣ) : ℚ) = 2 := by
  first
    | exact Units.val_mk0 _
    | rfl

/-- The multiset over `Fin 1` of the values of `f` is the singleton of `f 0`. -/
theorem univ_map_fin1 {α : Type*} (f : Fin 1 → α) :
    Multiset.map f (Finset.univ : Finset (Fin 1)).val = {f 0} := by
  first
    | (rw [Finset.univ_unique, Finset.singleton_val, Multiset.map_singleton] <;> rfl)
    | simp
    | rfl

/-- The bag of a block of one unit with constant entries. -/
theorem bagOf_const (g u w : ℚ) :
    bagOf (fun _ : Fin 1 => mk11 g u w) = {{g}, {u}, {w}} := by
  unfold bagOf
  simp only [univ_map_fin1, mk11_g, mk11_u, mk11_w] <;> rfl

/-- Swapping the first two of three elements of a multiset. -/
theorem tri_swap {α : Type*} (a b c : α) :
    ({a, b, c} : Multiset α) = {b, a, c} := by
  first
    | exact Multiset.cons_swap a b {c}
    | (simp only [Multiset.insert_eq_cons]; exact Multiset.cons_swap a b {c})

/-- The sum of the sums of the three tensors of a block of one unit. -/
theorem bagTotal_bagOf (θ : Fin 1 → GUnit ℚ 1 1) :
    ((bagOf θ).map Multiset.sum).sum = (θ 0).g 0 + (θ 0).u 0 + (θ 0).w 0 := by
  unfold bagOf
  simp only [univ_map_fin1, Multiset.insert_eq_cons, Multiset.map_cons, Multiset.map_singleton,
    Multiset.sum_cons, Multiset.sum_singleton] <;> ring

/-- The sum of the sums of the three tensors of the image of a constant unit under a gauge. -/
theorem bagTotal_const_act (g u w : ℚ) (c : ℚˣ) :
    ((bagOf (fun _ : Fin 1 => (mk11 g u w).act c)).map Multiset.sum).sum
      = g + (c : ℚ) * u + (c : ℚ)⁻¹ * w := by
  refine (bagTotal_bagOf _).trans ?_
  show ((mk11 g u w).act c).g 0 + ((mk11 g u w).act c).u 0 + ((mk11 g u w).act c).w 0 = _
  rw [GUnit.act_g_apply, GUnit.act_u_apply, GUnit.act_w_apply, mk11_g, mk11_u, mk11_w] <;> rfl

/-- The dot product on `Fin 1 → ℚ`. -/
theorem dot_fin1 (a b : Fin 1 → ℚ) : a ⬝ᵥ b = a 0 * b 0 := by
  first
    | (unfold dotProduct; exact Fin.sum_univ_one (fun i => a i * b i))
    | simp [dotProduct]

/-- The value at the coordinate `0` of the function of a block, for `d = e = 1`. -/
theorem F_apply11 {n : ℕ} (s : ℚ → ℚ) (θ : Fin n → GUnit ℚ 1 1) (x : Fin 1 → ℚ) :
    GBlock.F s θ x 0 = ∑ i, s ((θ i).g 0 * x 0) * ((θ i).u 0 * x 0) * (θ i).w 0 := by
  unfold GBlock.F
  first
    | rw [Fintype.sum_apply]
    | simp only [Finset.sum_apply]
  refine Finset.sum_congr rfl (fun i _ => ?_)
  rw [Pi.smul_apply, smul_eq_mul, dot_fin1, dot_fin1] <;> rfl

/-- The same, for a block of one unit. -/
theorem F_one11 (s : ℚ → ℚ) (θ : Fin 1 → GUnit ℚ 1 1) (x : Fin 1 → ℚ) :
    GBlock.F s θ x 0 = s ((θ 0).g 0 * x 0) * ((θ 0).u 0 * x 0) * (θ 0).w 0 := by
  rw [F_apply11]
  exact Fin.sum_univ_one (fun i => s ((θ i).g 0 * x 0) * ((θ i).u 0 * x 0) * (θ i).w 0)

/-- The same, for a block of one unit with constant entries. -/
theorem F_const11 (s : ℚ → ℚ) (g u w : ℚ) (x : Fin 1 → ℚ) :
    GBlock.F s (fun _ : Fin 1 => mk11 g u w) x 0 = s (g * x 0) * (u * x 0) * w :=
  F_one11 s (fun _ : Fin 1 => mk11 g u w) x

/-- The pair of units `(g, u, w), (-g, u, -w)`. -/
def pairBlock (g u w : ℚ) : Fin 2 → GUnit ℚ 1 1 :=
  ![mk11 g u w, mk11 (-g) u (-w)]

/-- The value at the coordinate `0` of the function of a pair of units. -/
theorem F_pair (s : ℚ → ℚ) (g u w : ℚ) (x : Fin 1 → ℚ) :
    GBlock.F s (pairBlock g u w) x 0
      = s (g * x 0) * (u * x 0) * w + s ((-g) * x 0) * (u * x 0) * (-w) := by
  rw [F_apply11, Fin.sum_univ_two] <;> first | rfl | simp [pairBlock, mk11]

/-- The sum of the squares of the gates of a block with `d = 1`: a function of the multiset of
the gates. -/
def gateSq {n : ℕ} (θ : Fin n → GUnit ℚ 1 1) : ℚ :=
  ∑ i, ((θ i).g 0) ^ 2

/-- One move does not change the multiset of the gates, hence not `gateSq`: a permutation
reindexes the sum and a gauge does not touch the gates. -/
theorem gateSq_move {n : ℕ} {θ θ' : Fin n → GUnit ℚ 1 1} (h : Move GUnit.act θ θ') :
    gateSq θ = gateSq θ' := by
  cases h with
  | perm σ => exact (Equiv.sum_comp σ (fun i => ((θ i).g 0) ^ 2)).symm
  | gauge c => rfl

/-- The equivalence closure of the moves does not change `gateSq`. -/
theorem gateSq_tequiv {n : ℕ} {θ θ' : Fin n → GUnit ℚ 1 1} (h : TEquiv GUnit.act θ θ') :
    gateSq θ = gateSq θ' :=
  GUnit.tequiv_invariant gateSq (fun _ _ hm => gateSq_move hm) h

theorem gateSq_pair (g u w : ℚ) : gateSq (pairBlock g u w) = g ^ 2 + (-g) ^ 2 := by
  unfold gateSq
  rw [Fin.sum_univ_two] <;> first | rfl | simp [pairBlock, mk11]

/-- The symmetry that `max t 0 - max (-t) 0 = t` gives the pair of units, in the form of an
equation between reals. -/
theorem relu_pair (t : ℚ) :
    max (1 * t) 0 * (2 * t) * 1 + max ((-1) * t) 0 * (2 * t) * (-1)
      = max (2 * t) 0 * (1 * t) * 1 + max ((-2) * t) 0 * (1 * t) * (-1) := by
  rcases le_total 0 t with h | h
  · rw [max_eq_left (by linarith : (0 : ℚ) ≤ 1 * t),
      max_eq_right (by linarith : (-1) * t ≤ (0 : ℚ)),
      max_eq_left (by linarith : (0 : ℚ) ≤ 2 * t),
      max_eq_right (by linarith : (-2) * t ≤ (0 : ℚ))]
    ring
  · rw [max_eq_right (by linarith : 1 * t ≤ (0 : ℚ)),
      max_eq_left (by linarith : (0 : ℚ) ≤ (-1) * t),
      max_eq_right (by linarith : 2 * t ≤ (0 : ℚ)),
      max_eq_left (by linarith : (0 : ℚ) ≤ (-2) * t)]
    ring

end Separ

open Separ

/-- The same bag of tensors does not give the same function: exchanging the gate and the up
tensor keeps the bag, and for `s t = t ^ 2` it changes the function (at `x = 1` the values are
`2` and `4`). -/
theorem sep_bag_not_function :
    ∃ θ θ' : Fin 1 → GUnit ℚ 1 1, bagOf θ = bagOf θ' ∧
      GBlock.F (fun t : ℚ => t ^ 2) θ ≠ GBlock.F (fun t : ℚ => t ^ 2) θ' := by
  refine ⟨fun _ => mk11 1 2 1, fun _ => mk11 2 1 1, ?_, ?_⟩
  · rw [bagOf_const, bagOf_const]
    exact tri_swap _ _ _
  · intro h
    have h0 := congrFun (congrFun h (fun _ => (1 : ℚ))) 0
    rw [F_const11, F_const11] at h0
    norm_num at h0

/-- The same function does not give the same bag of tensors, for every `s`: a unit and its image
under the gauge `c = 2` have the same function and the bags `{{1}, {1}, {1}}` and `{{1}, {2},
{1/2}}`. -/
theorem sep_function_not_bag (s : ℚ → ℚ) :
    ∃ θ θ' : Fin 1 → GUnit ℚ 1 1, GBlock.F s θ = GBlock.F s θ' ∧ bagOf θ ≠ bagOf θ' := by
  refine ⟨fun _ => mk11 1 1 1, fun _ => (mk11 1 1 1).act twoU, ?_, ?_⟩
  · exact (GBlock.F_gauge s (fun _ : Fin 1 => mk11 1 1 1) (fun _ => twoU)).symm
  · intro hb
    have h1 : ((bagOf (fun _ : Fin 1 => mk11 1 1 1)).map Multiset.sum).sum
        = ((bagOf (fun _ : Fin 1 => (mk11 1 1 1).act twoU)).map Multiset.sum).sum :=
      congrArg (fun b : Multiset (Multiset ℚ) => (b.map Multiset.sum).sum) hb
    have e1 : ((bagOf (fun _ : Fin 1 => mk11 1 1 1)).map Multiset.sum).sum = 3 := by
      rw [bagTotal_bagOf, mk11_g, mk11_u, mk11_w] <;> norm_num
    have e2 : ((bagOf (fun _ : Fin 1 => (mk11 1 1 1).act twoU)).map Multiset.sum).sum = 7 / 2 := by
      rw [bagTotal_const_act, twoU_val] <;> norm_num
    rw [e1, e2] at h1
    norm_num at h1

/-- The same function does not give the same orbit: for `s t = max t 0` the blocks `(1, 2, 1),
(-1, 2, -1)` and `(2, 1, 1), (-2, 1, -1)` have the same function, and they are not in one orbit
of the moves, because the multiset of the gates, `{1, -1}` against `{2, -2}`, is invariant under
the moves (here through its sum of squares, `2` against `8`). -/
theorem sep_orbit_not_function :
    ∃ θ θ' : Fin 2 → GUnit ℚ 1 1,
      GBlock.F (fun t : ℚ => max t 0) θ = GBlock.F (fun t : ℚ => max t 0) θ' ∧
        ¬ TEquiv GUnit.act θ θ' := by
  refine ⟨pairBlock 1 2 1, pairBlock 2 1 1, ?_, ?_⟩
  · funext x i
    have hi : i = 0 := Subsingleton.elim i 0
    subst hi
    rw [F_pair, F_pair]
    exact relu_pair (x 0)
  · intro h
    have h1 := gateSq_tequiv h
    rw [gateSq_pair, gateSq_pair] at h1
    norm_num at h1

end LmnfNormal
