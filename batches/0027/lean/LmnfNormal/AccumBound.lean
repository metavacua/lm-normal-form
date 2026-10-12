/-
SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
SPDX-License-Identifier: AGPL-3.0-or-later
-/
import Mathlib

/-!
# The accumulator of a ternary matrix times an int8 vector stays in an i32

A BitNet-style linear layer multiplies a matrix with entries in `{-1, 0, 1}` by a vector of signed
bytes and sums the products.  Each product has absolute value at most 128, so the sum of `n` of them
has absolute value at most `128 n`; below `2^31` as long as `n ≤ 2^24 - 1`.  WebAssembly's `i32.add`
wraps modulo `2^32`, so within this bound the wrapped sum is the integer sum, and the sum does not
depend on the order of the terms: a permutation of the columns does not change a bit of the output
(`batches/0027/wasm_host.py` checks this in wasmtime on the module built from `batches/0027/wasm`;
the sum of floats, which is not exact, does change).
-/

namespace LmnfNormal

/-- A ternary row times a vector of signed bytes: the sum has absolute value at most `128 n`. -/
theorem ternary_dot_bound {n : ℕ} (w x : Fin n → ℤ) (hw : ∀ i, |w i| ≤ 1) (hx : ∀ i, |x i| ≤ 128) :
    |∑ i, w i * x i| ≤ 128 * n := by
  have h1 : |∑ i, w i * x i| ≤ ∑ i, |w i * x i| := Finset.abs_sum_le_sum_abs _ _
  have h2 : ∀ i, |w i * x i| ≤ 128 := by
    intro i
    rw [abs_mul]
    have h := mul_le_mul (hw i) (hx i) (abs_nonneg _) (by norm_num : (0 : ℤ) ≤ 1)
    linarith
  have h3 : ∑ i, |w i * x i| ≤ ∑ _i : Fin n, (128 : ℤ) := Finset.sum_le_sum (fun i _ => h2 i)
  have h4 : ∑ _i : Fin n, (128 : ℤ) = 128 * n := by
    simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
    ring
  linarith

/-- The bound fits an `i32` for every length below `2^24`. -/
theorem ternary_dot_fits_i32 (n : ℕ) (hn : n ≤ 16777215) : 128 * (n : ℤ) < 2 ^ 31 := by
  have h : (n : ℤ) ≤ 16777215 := by exact_mod_cast hn
  norm_num
  linarith

/-- The accumulator for a row of width 6912, the `intermediate_size` of BitNet b1.58 2B4T (the width
is taken from the model's configuration, not proved here): a ternary row times int8 activations
sums to less than `2^31` in absolute value. -/
theorem ternary_dot_2b4t_down_fits_i32 (w x : Fin 6912 → ℤ) (hw : ∀ i, |w i| ≤ 1)
    (hx : ∀ i, |x i| ≤ 128) : |∑ i, w i * x i| < 2 ^ 31 := by
  have h := ternary_dot_bound w x hw hx
  have h2 : (128 : ℤ) * (6912 : ℕ) < 2 ^ 31 := by norm_num
  exact lt_of_le_of_lt h h2

end LmnfNormal
