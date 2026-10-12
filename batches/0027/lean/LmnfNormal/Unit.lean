/-
SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
SPDX-License-Identifier: AGPL-3.0-or-later
-/
import Mathlib
import LmnfNormal.TupleNF
import LmnfNormal.InvolutionNF

/-!
# The gated unit block: the nonlinear instance of the tuple pattern

A *gated unit* is a triple `(g, u, w)`: a gate row `g`, an up row `u` and a down column `w`.  A
block is a family of `n` units, and its function is `x ↦ ∑ i, (s (g_i ⬝ᵥ x) * (u_i ⬝ᵥ x)) • w_i` for
an *arbitrary* nonlinearity `s` (the gated MLP of `batches/0015/model.py`).

Two things leave the function unchanged whatever `s` is:

* a permutation of the units (the sum is over a shared index);
* a scalar `c ≠ 0` on the up row together with `c⁻¹` on the down column (the gate is not touched,
  because `s` is not homogeneous).

These are the moves `Move.perm` and `Move.gauge` of the tuple pattern, with the group `Kˣ` acting on
a unit by `GUnit.act`.  This file

* defines the action, the normalizing scalar `GUnit.factor` and the normal form of one unit
  `GUnit.scale`, and proves that `GUnit.scale` is an `ItemNormalizer`;
* gives an injective key `GUnit.key` into `List K`, ordered lexicographically, so that `NF GUnit.key
  GUnit.scale` is a canonicalization of `TEquiv GUnit.act`;
* proves that the function of a block is invariant under `TEquiv` and under the normal form.

Nothing here is a statement about a particular `s`.
-/

namespace LmnfNormal

/-- A gated unit with `d` input coordinates and `e` output coordinates: the gate row `g`, the up
row `u` and the down column `w`. -/
@[ext]
structure GUnit (K : Type*) (d e : ℕ) where
  /-- the gate row -/
  g : Fin d → K
  /-- the up row -/
  u : Fin d → K
  /-- the down column -/
  w : Fin e → K

namespace GUnit

/-- Invariance under `TEquiv`, from invariance under one `Move`: if `P` does not change along a
single move, it does not change along the equivalence closure. -/
theorem tequiv_invariant {Γ U : Type*} {V : Sort*} {n : ℕ} {act : Γ → U → U}
    (P : (Fin n → U) → V) (hP : ∀ f g, Move act f g → P f = P g)
    {f g : Fin n → U} (h : TEquiv act f g) : P f = P g := by
  have h' : Relation.EqvGen (Move act) f g := h
  clear h
  induction h' with
  | rel x y hxy => exact hP x y hxy
  | refl x => rfl
  | symm x y _ ih => exact ih.symm
  | trans x y z _ _ ih1 ih2 => exact ih1.trans ih2

section Action

variable {K : Type*} [Field K] {d e : ℕ}

/-- The action of a unit of `K` on a gated unit: `c` multiplies the up row and `c⁻¹` multiplies
the down column; the gate is not touched. -/
def act (c : Kˣ) (x : GUnit K d e) : GUnit K d e :=
  ⟨x.g, (c : K) • x.u, ((c⁻¹ : Kˣ) : K) • x.w⟩

theorem act_g (c : Kˣ) (x : GUnit K d e) : (x.act c).g = x.g := rfl

theorem act_u (c : Kˣ) (x : GUnit K d e) : (x.act c).u = (c : K) • x.u := rfl

theorem act_w (c : Kˣ) (x : GUnit K d e) : (x.act c).w = (c : K)⁻¹ • x.w := by
  change ((c⁻¹ : Kˣ) : K) • x.w = (c : K)⁻¹ • x.w
  exact congrArg (fun a : K => a • x.w) (Units.val_inv_eq_inv_val c)

theorem act_g_apply (c : Kˣ) (x : GUnit K d e) (i : Fin d) : (x.act c).g i = x.g i := rfl

theorem act_u_apply (c : Kˣ) (x : GUnit K d e) (i : Fin d) : (x.act c).u i = (c : K) * x.u i :=
  rfl

theorem act_w_apply (c : Kˣ) (x : GUnit K d e) (j : Fin e) :
    (x.act c).w j = (c : K)⁻¹ * x.w j := by
  rw [act_w, Pi.smul_apply, smul_eq_mul]

/-- A scalar passes out of a dot product. -/
theorem smul_dot (c : K) (a b : Fin d → K) : (c • a) ⬝ᵥ b = c * (a ⬝ᵥ b) := by
  rw [smul_dotProduct, smul_eq_mul]

end Action

end GUnit

section Block

variable {K : Type*} [Field K] {d e n : ℕ}

/-- `a * (c * b) * c⁻¹ = a * b` for `c ≠ 0`. -/
theorem GUnit.gauge_cancel (a b c : K) (hc : c ≠ 0) : a * (c * b) * c⁻¹ = a * b := by
  calc a * (c * b) * c⁻¹ = a * b * (c * c⁻¹) := by ring
    _ = a * b := by rw [mul_inv_cancel₀ hc, mul_one]

namespace GBlock

/-- The function of a block of gated units, for an arbitrary nonlinearity `s`: `x ↦ ∑ i, (s (g_i
⬝ᵥ x) * (u_i ⬝ᵥ x)) • w_i`. -/
def F (s : K → K) (θ : Fin n → GUnit K d e) (x : Fin d → K) : Fin e → K :=
  ∑ i, (s ((θ i).g ⬝ᵥ x) * ((θ i).u ⬝ᵥ x)) • (θ i).w

/-- A permutation of the units does not change the function: the sum is over a shared index. -/
theorem F_perm (s : K → K) (θ : Fin n → GUnit K d e) (σ : Equiv.Perm (Fin n)) :
    F s (θ ∘ σ) = F s θ := by
  funext x
  unfold F
  exact Equiv.sum_comp σ (fun i => (s ((θ i).g ⬝ᵥ x) * ((θ i).u ⬝ᵥ x)) • (θ i).w)

/-- A scalar on the up row with its inverse on the down column does not change the function,
whatever `s` is. -/
theorem F_gauge (s : K → K) (θ : Fin n → GUnit K d e) (c : Fin n → Kˣ) :
    F s (fun i => (θ i).act (c i)) = F s θ := by
  funext x
  unfold F
  refine Finset.sum_congr rfl ?_
  intro i _
  change (s (((θ i).act (c i)).g ⬝ᵥ x) * (((θ i).act (c i)).u ⬝ᵥ x)) • ((θ i).act (c i)).w
    = (s ((θ i).g ⬝ᵥ x) * ((θ i).u ⬝ᵥ x)) • (θ i).w
  rw [GUnit.act_g, GUnit.act_u, GUnit.act_w, GUnit.smul_dot, smul_smul,
    GUnit.gauge_cancel _ _ _ (Units.ne_zero (c i))]

/-- One move does not change the function. -/
theorem F_move (s : K → K) {θ θ' : Fin n → GUnit K d e} (h : Move GUnit.act θ θ') :
    F s θ = F s θ' := by
  cases h with
  | perm σ => exact (F_perm s θ σ).symm
  | gauge c => exact (F_gauge s θ c).symm

/-- The equivalence closure of the moves does not change the function. -/
theorem F_tequiv (s : K → K) {θ θ' : Fin n → GUnit K d e} (h : TEquiv GUnit.act θ θ') :
    F s θ = F s θ' :=
  GUnit.tequiv_invariant (F s) (fun _ _ hm => F_move s hm) h

end GBlock

/-- Soundness of the equivalence of gated blocks: equivalent blocks have the same function, for
every nonlinearity `s`. -/
theorem sound_Mono (s : K → K) {θ θ' : Fin n → GUnit K d e} (h : TEquiv GUnit.act θ θ') :
    GBlock.F s θ = GBlock.F s θ' :=
  GBlock.F_tequiv s h

end Block

section Scale

variable {K : Type*} [Field K] [DecidableEq K] {d e : ℕ}

/-- The first non-zero coordinate of `v`, and `1` when `v = 0`. -/
def firstNonzero {m : ℕ} (v : Fin m → K) : K :=
  ((List.ofFn v).find? (fun a => decide (a ≠ 0))).getD 1

/-- A non-zero vector has a first non-zero coordinate. -/
theorem firstNonzero_find_exists {m : ℕ} {v : Fin m → K} (hv : v ≠ 0) :
    ∃ a, (List.ofFn v).find? (fun a => decide (a ≠ 0)) = some a := by
  cases hf : (List.ofFn v).find? (fun a => decide (a ≠ 0)) with
  | some a => exact ⟨a, rfl⟩
  | none =>
    exfalso
    apply hv
    funext i
    have h : ¬ (decide (v i ≠ 0) = true) :=
      List.find?_eq_none.mp hf (v i) (List.mem_ofFn.mpr ⟨i, rfl⟩)
    change v i = 0
    by_contra h'
    exact h (decide_eq_true h')

/-- `firstNonzero v` is never zero. -/
theorem firstNonzero_ne_zero {m : ℕ} (v : Fin m → K) : firstNonzero v ≠ 0 := by
  unfold firstNonzero
  cases hf : (List.ofFn v).find? (fun a => decide (a ≠ 0)) with
  | none =>
    change (1 : K) ≠ 0
    exact one_ne_zero
  | some a =>
    change a ≠ 0
    have h := List.find?_some hf
    exact of_decide_eq_true h

omit [DecidableEq K] in
/-- A non-zero scalar times a non-zero vector is non-zero. -/
theorem smul_vec_ne_zero {m : ℕ} {c : K} {v : Fin m → K} (hc : c ≠ 0) (hv : v ≠ 0) :
    c • v ≠ 0 := by
  intro h
  apply hv
  funext i
  have h' := congrFun h i
  simp only [Pi.smul_apply, smul_eq_mul, Pi.zero_apply] at h'
  exact (mul_eq_zero.mp h').resolve_left hc

/-- The first non-zero coordinate scales with the vector.  The hypothesis `v ≠ 0` is needed:
`firstNonzero 0 = 1`, and `1 ≠ c * 1` unless `c = 1`. -/
theorem firstNonzero_smul {m : ℕ} {c : K} {v : Fin m → K} (hv : v ≠ 0) (hc : c ≠ 0) :
    firstNonzero (c • v) = c * firstNonzero v := by
  obtain ⟨a, ha⟩ := firstNonzero_find_exists hv
  have hlist : List.ofFn (c • v) = (List.ofFn v).map (fun a => c * a) :=
    (congrArg List.ofFn (funext fun i => rfl : (c • v) = (fun a => c * a) ∘ v)).trans
      List.map_ofFn.symm
  have hp : ∀ a : K, decide (c * a ≠ 0) = decide (a ≠ 0) := fun a =>
    decide_eq_decide.mpr ⟨fun h h0 => h (by rw [h0, mul_zero]), fun h => mul_ne_zero hc h⟩
  have hpf : ((fun a : K => decide (a ≠ 0)) ∘ fun a => c * a) = fun a => decide (a ≠ 0) :=
    funext hp
  unfold firstNonzero
  rw [hlist, List.find?_map, hpf, ha]
  rfl

namespace GUnit

/-- The normalizing scalar of a unit: if `u ≠ 0`, the scalar `(firstNonzero u)⁻¹` that makes the
first non-zero coordinate of `c • u` equal to `1`; if `u = 0`, the scalar `firstNonzero w` that
makes the first non-zero coordinate of `c⁻¹ • w` equal to `1`; it is `1` when `u = 0` and `w =
0`. -/
def factor (x : GUnit K d e) : K :=
  if x.u ≠ 0 then (firstNonzero x.u)⁻¹ else firstNonzero x.w

/-- The normal form of one unit: the representative of its orbit under `Kˣ` obtained by the
scalar `factor`. -/
def scale (x : GUnit K d e) : GUnit K d e :=
  ⟨x.g, x.factor • x.u, x.factor⁻¹ • x.w⟩

theorem factor_ne_zero (x : GUnit K d e) : x.factor ≠ 0 := by
  unfold factor
  by_cases h : x.u ≠ 0
  · rw [ite_eq_left h]
    exact inv_ne_zero (firstNonzero_ne_zero _)
  · rw [ite_eq_right h]
    exact firstNonzero_ne_zero _

/-- The factor of the acted unit, when the unit is not zero.  For `u = 0` and `w = 0` the factor
is `1` on both sides, so the hypothesis cannot be dropped. -/
theorem factor_act (c : Kˣ) (x : GUnit K d e) (h : x.u ≠ 0 ∨ x.w ≠ 0) :
    (x.act c).factor = (c : K)⁻¹ * x.factor := by
  have hc : (c : K) ≠ 0 := Units.ne_zero c
  have hci : (c : K)⁻¹ ≠ 0 := inv_ne_zero hc
  by_cases hu : x.u ≠ 0
  · have hu' : (x.act c).u ≠ 0 := by
      rw [act_u]
      exact smul_vec_ne_zero hc hu
    unfold factor
    rw [ite_eq_left hu', ite_eq_left hu, act_u, firstNonzero_smul hu hc, mul_inv]
  · have hw : x.w ≠ 0 := h.resolve_left hu
    have hu0 : x.u = 0 := by
      by_contra h'
      exact hu h'
    have hu' : ¬ ((x.act c).u ≠ 0) := by
      intro h'
      apply h'
      rw [act_u, hu0, smul_zero]
    unfold factor
    rw [ite_eq_right hu', ite_eq_right hu, act_w, firstNonzero_smul hw hci]

omit [DecidableEq K] in
/-- `a⁻¹ * f * a = f` for `a ≠ 0`. -/
theorem cancel_up (a f : K) (ha : a ≠ 0) : a⁻¹ * f * a = f := by
  calc a⁻¹ * f * a = f * (a⁻¹ * a) := by ring
    _ = f := by rw [inv_mul_cancel₀ ha, mul_one]

omit [DecidableEq K] in
/-- `(a⁻¹ * f)⁻¹ * a⁻¹ = f⁻¹` for `a ≠ 0`. -/
theorem cancel_down (a f : K) (ha : a ≠ 0) : (a⁻¹ * f)⁻¹ * a⁻¹ = f⁻¹ := by
  rw [mul_inv, inv_inv]
  calc a * f⁻¹ * a⁻¹ = f⁻¹ * (a * a⁻¹) := by ring
    _ = f⁻¹ := by rw [mul_inv_cancel₀ ha, mul_one]

/-- The normal form of a unit is constant on its orbit. -/
theorem scale_act (c : Kˣ) (x : GUnit K d e) : (x.act c).scale = x.scale := by
  have hc : (c : K) ≠ 0 := Units.ne_zero c
  by_cases h : x.u ≠ 0 ∨ x.w ≠ 0
  · have hf := factor_act c x h
    refine GUnit.ext rfl ?_ ?_
    · change (x.act c).factor • (x.act c).u = x.factor • x.u
      rw [hf, act_u, smul_smul, cancel_up _ _ hc]
    · change (x.act c).factor⁻¹ • (x.act c).w = x.factor⁻¹ • x.w
      rw [hf, act_w, smul_smul, cancel_down _ _ hc]
  · have hu0 : x.u = 0 := by
      by_contra h'
      exact h (Or.inl h')
    have hw0 : x.w = 0 := by
      by_contra h'
      exact h (Or.inr h')
    refine GUnit.ext rfl ?_ ?_
    · change (x.act c).factor • (x.act c).u = x.factor • x.u
      simp only [act_u, hu0, smul_zero]
    · change (x.act c).factor⁻¹ • (x.act c).w = x.factor⁻¹ • x.w
      simp only [act_w, hw0, smul_zero]

/-- The normal form of a unit is a point of its orbit. -/
theorem scale_representative (x : GUnit K d e) : ∃ c : Kˣ, x.scale = x.act c := by
  refine ⟨Units.mk0 x.factor (factor_ne_zero x), GUnit.ext rfl ?_ ?_⟩
  · exact (act_u (Units.mk0 x.factor (factor_ne_zero x)) x).symm
  · exact (act_w (Units.mk0 x.factor (factor_ne_zero x)) x).symm

/-- `GUnit.scale` normalizes a unit under the action of `Kˣ`: it is invariant and it is a point
of the orbit. -/
theorem itemNormalizer :
    ItemNormalizer (GUnit.act (K := K) (d := d) (e := e)) GUnit.scale :=
  ⟨fun c x => scale_act c x, fun x => scale_representative x⟩

end GUnit

end Scale

section Key

variable {K : Type*} {d e : ℕ}

/-- An injective key of a unit: the three vectors, one after the other, as a list.  `List K` is
linearly ordered, lexicographically, when `K` is. -/
def GUnit.key (x : GUnit K d e) : List K :=
  List.ofFn x.g ++ List.ofFn x.u ++ List.ofFn x.w

/-- The key determines the unit: the lengths `d`, `d` and `e` of the three vectors are fixed by
the type. -/
theorem GUnit.key_injective : Function.Injective (GUnit.key (K := K) (d := d) (e := e)) := by
  intro x y h
  have h' : List.ofFn x.g ++ List.ofFn x.u ++ List.ofFn x.w
      = List.ofFn y.g ++ List.ofFn y.u ++ List.ofFn y.w := h
  have h1 : List.ofFn x.g ++ List.ofFn x.u = List.ofFn y.g ++ List.ofFn y.u ∧
      List.ofFn x.w = List.ofFn y.w :=
    List.append_inj h' (by simp)
  have h2 : List.ofFn x.g = List.ofFn y.g ∧ List.ofFn x.u = List.ofFn y.u :=
    List.append_inj h1.1 (by simp)
  exact GUnit.ext (List.ofFn_inj.mp h2.1) (List.ofFn_inj.mp h2.2) (List.ofFn_inj.mp h1.2)

end Key

section Canon

variable {K : Type*} [Field K] [DecidableEq K] {d e n : ℕ}

/-- The packaged canonicalization of gated blocks: sorting the units, each in its normal form
`GUnit.scale`, by the key `GUnit.key`, is a canonicalization of `TEquiv GUnit.act`. -/
theorem canon_Mono_scale_sort [LinearOrder K] :
    Canonicalization
      (TEquiv (GUnit.act (K := K) (d := d) (e := e)) :
        (Fin n → GUnit K d e) → (Fin n → GUnit K d e) → Prop)
      (NF GUnit.key GUnit.scale) := by
  exact canonicalization GUnit.key_injective GUnit.itemNormalizer

/-- The normal form of a block has the same function as the block, for every nonlinearity `s`. -/
theorem GBlock.F_NF [LinearOrder K] (s : K → K) (θ : Fin n → GUnit K d e) :
    GBlock.F s (NF GUnit.key GUnit.scale θ) = GBlock.F s θ := by
  have hc := canon_Mono_scale_sort (K := K) (d := d) (e := e) (n := n)
  have h := hc.representative θ
  exact (GBlock.F_tequiv s h).symm

end Canon

end LmnfNormal
