/-
SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
SPDX-License-Identifier: AGPL-3.0-or-later
-/
import Mathlib

/-!
# Kleene's strong logic as balanced ternary

The verdict of a check is one of three: refuted, unknown, supported.  Read as the balanced ternary digits `-1`, `0`, `+1`, negation is the
change of sign, conjunction is the minimum and disjunction the maximum (Wikipedia, "Three-valued logic", the section on Kleene and Priest
logics).  `nf.dl` computes the verdict of a plan as an integer minimum; `val_conj` says that this is the conjunction of this file.

The theorems here are the properties that the use of the verdicts relies on: negation is an involution, the laws of De Morgan hold (so that a
negation normal form exists), the excluded middle and the law of non-contradiction fail at `unknown` (so the verdicts are not Boolean), and the
three connectives are monotone for the order of information in which `unknown` is below both decided verdicts (so that learning more can only
turn an unknown verdict into a decided one, never change a decided one).
-/

namespace LmnfNormal

/-- The three verdicts. -/
inductive Verdict where
  | refuted
  | unknown
  | supported
  deriving DecidableEq

namespace Verdict

/-- The balanced ternary digit of a verdict. -/
def val : Verdict → ℤ
  | refuted => -1
  | unknown => 0
  | supported => 1

/-- Negation: refuted and supported exchange, unknown stays. -/
def neg : Verdict → Verdict
  | refuted => supported
  | unknown => unknown
  | supported => refuted

/-- Conjunction: refuted if either is refuted, else unknown if either is unknown, else supported. -/
def conj : Verdict → Verdict → Verdict
  | refuted, _ => refuted
  | _, refuted => refuted
  | unknown, _ => unknown
  | _, unknown => unknown
  | supported, supported => supported

/-- Disjunction: supported if either is supported, else unknown if either is unknown, else refuted. -/
def disj : Verdict → Verdict → Verdict
  | supported, _ => supported
  | _, supported => supported
  | unknown, _ => unknown
  | _, unknown => unknown
  | refuted, refuted => refuted

/-- `leInfo a b`: `b` is `a` or `a` is unknown (the order of information). -/
def leInfo : Verdict → Verdict → Bool
  | unknown, _ => true
  | a, b => decide (a = b)

theorem val_injective : Function.Injective val := by
  intro a b h
  cases a <;> cases b <;> first | rfl | exact absurd h (by decide)

theorem val_neg (a : Verdict) : (neg a).val = -a.val := by
  cases a <;> first | rfl | decide

theorem val_conj (a b : Verdict) : (conj a b).val = min a.val b.val := by
  cases a <;> cases b <;> first | rfl | decide

theorem val_disj (a b : Verdict) : (disj a b).val = max a.val b.val := by
  cases a <;> cases b <;> first | rfl | decide

theorem neg_neg (a : Verdict) : neg (neg a) = a := by
  cases a <;> rfl

theorem de_morgan_conj (a b : Verdict) : neg (conj a b) = disj (neg a) (neg b) := by
  cases a <;> cases b <;> rfl

theorem de_morgan_disj (a b : Verdict) : neg (disj a b) = conj (neg a) (neg b) := by
  cases a <;> cases b <;> rfl

theorem conj_comm (a b : Verdict) : conj a b = conj b a := by
  cases a <;> cases b <;> rfl

theorem disj_comm (a b : Verdict) : disj a b = disj b a := by
  cases a <;> cases b <;> rfl

/-- The excluded middle fails: `unknown or not unknown` is `unknown`. -/
theorem excluded_middle_fails : disj unknown (neg unknown) = unknown := rfl

/-- The law of non-contradiction fails: `unknown and not unknown` is `unknown`. -/
theorem non_contradiction_fails : conj unknown (neg unknown) = unknown := rfl

theorem neg_mono (a b : Verdict) (h : leInfo a b = true) : leInfo (neg a) (neg b) = true := by
  revert h
  cases a <;> cases b <;> decide

theorem conj_mono (a a' b b' : Verdict) (h₁ : leInfo a a' = true) (h₂ : leInfo b b' = true) :
    leInfo (conj a b) (conj a' b') = true := by
  revert h₁ h₂
  cases a <;> cases a' <;> cases b <;> cases b' <;> decide

theorem disj_mono (a a' b b' : Verdict) (h₁ : leInfo a a' = true) (h₂ : leInfo b b' = true) :
    leInfo (disj a b) (disj a' b') = true := by
  revert h₁ h₂
  cases a <;> cases a' <;> cases b <;> cases b' <;> decide

end Verdict

end LmnfNormal
