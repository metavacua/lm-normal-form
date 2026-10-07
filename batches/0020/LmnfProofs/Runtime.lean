/-
SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
SPDX-License-Identifier: AGPL-3.0-or-later
-/
import Mathlib
import LmnfProofs.Whole
import LmnfProofs.GaugeStreamPerm

/-!
# The definitions of `Model.lean`, run over a finite field

What the theorems of batches 0020 and 0021 are about is the Lean definition of the forward pass; this file runs that definition, over the field `ZMod 1000003` (a field because 1000003 is
prime; small enough that nothing grows, as it does over the rationals, where every application of a nonlinearity doubles the degree). A model in the layout of `model.py` (the matrices of
Hugging Face's Llama, `out × in`, the heads and the rotary planes in the flat rows) is turned into the structured layer of `Model.lean` by `toLayer`, which *is* the correspondence between
the two layouts (the plane `p` of head `a` is the rows `a * hv + p` and `a * hv + p + hv / 2`); `emit` prints arrays exactly; `emitRun` prints, layer by layer, the queries, keys and values
of the cache, the stream and the logits; `emitWeights` prints the weights in the flat layout again. `batches/0022/gen22.py` writes the data and the transformed models, and compares the
output with the exact arithmetic of `model.py` and `gauge.py` over the same field (`ops.FieldOps`).
-/

set_option linter.unusedSectionVars false

namespace Lmnf.Runtime

open Matrix

/-- The field: the integers modulo the prime 1000003. -/
abbrev Fp := ZMod 1000003

instance : Fact (Nat.Prime 1000003) := ⟨by norm_num⟩

def showF (x : Fp) : String := toString x.val

def emit (label : String) (rows : List (List Fp)) : IO Unit := do
  IO.println s!"## {label}"
  for r in rows do
    IO.println (" ".intercalate (r.map showF))
  (← IO.getStdout).flush

def emitNat (label : String) (rows : List (List ℕ)) : IO Unit := do
  IO.println s!"## {label}"
  for r in rows do
    IO.println (" ".intercalate (r.map toString))
  (← IO.getStdout).flush

def tab (rows : List (List Fp)) (i j : ℕ) : Fp := (rows.getD i []).getD j 0

def tab1 (xs : List Fp) (i : ℕ) : Fp := xs.getD i 0

/-- The weights of one layer in the layout of `model.py`: a function of the row and the column. -/
structure Flat where
  wq : ℕ → ℕ → Fp
  wk : ℕ → ℕ → Fp
  wv : ℕ → ℕ → Fp
  wo : ℕ → ℕ → Fp
  wg : ℕ → ℕ → Fp
  wu : ℕ → ℕ → Fp
  wd : ℕ → ℕ → Fp
  an : ℕ → Fp
  mn : ℕ → Fp

/-- The structured layer of `Model.lean` from the flat layout: head `a`, plane `p` of the queries is the rows `a * hv + p` (first component) and `a * hv + p + h2` (second) of `wq`; the
value dimension `hv` is the head dimension, `h2` half of it; the output matrix of head `a` is the columns `a * hv ..` of `wo`. -/
def toLayer (d hv h2 nh nkv dff : ℕ) (F : Flat) : Layer Fp d hv dff (Fin nh) (Fin nkv) (Fin h2) where
  Wq := fun a p => Matrix.of fun (r : Fin 2) (j : Fin d) => F.wq (a.val * hv + p.val + r.val * h2) j.val
  Wk := fun g p => Matrix.of fun (r : Fin 2) (j : Fin d) => F.wk (g.val * hv + p.val + r.val * h2) j.val
  Wv := fun g => Matrix.of fun (r : Fin hv) (j : Fin d) => F.wv (g.val * hv + r.val) j.val
  Wo := fun a => Matrix.of fun (i : Fin d) (r : Fin hv) => F.wo i.val (a.val * hv + r.val)
  Wg := Matrix.of fun (i : Fin dff) (j : Fin d) => F.wg i.val j.val
  Wu := Matrix.of fun (i : Fin dff) (j : Fin d) => F.wu i.val j.val
  Wd := Matrix.of fun (i : Fin d) (j : Fin dff) => F.wd i.val j.val
  ga := fun i => F.an i.val
  gm := fun i => F.mn i.val

def circleC (t : Fp) : Fp := (1 - t ^ 2) / (1 + t ^ 2)

def circleS (t : Fp) : Fp := 2 * t / (1 + t ^ 2)

/-- The `n`-th power of the rotation `(c, s)`. -/
def rotPow (c s : Fp) : ℕ → Fp × Fp
  | 0 => (1, 0)
  | n + 1 =>
    let r := rotPow c s n
    (c * r.1 - s * r.2, s * r.1 + c * r.2)

/-- The environment of `FieldOps` (`ops.py`) over `Fp`: exp s = s² + s + 1, silu s = s² + 2 s, the scale of the norm 1 / (m² + 3), the scale of the scores 7, plane `p` rotating by
the point of the circle of `ts p` at each position. Query head `a` reads group `a / rep`. -/
def mkEnv (T nh nkv h2 rep : ℕ) (hgr : ∀ a : Fin nh, a.val / rep < nkv) (ts : List Fp) :
    Env Fp T (Fin nh) (Fin nkv) (Fin h2) where
  F := { expo := fun s => s ^ 2 + s + 1, silu := fun s => s ^ 2 + 2 * s, rs := fun m => (m ^ 2 + 3)⁻¹ }
  gr := fun a => ⟨a.val / rep, hgr a⟩
  rot := fun t p => cplx (rotPow (circleC (tab1 ts p.val)) (circleS (tab1 ts p.val)) t.val).1
    (rotPow (circleC (tab1 ts p.val)) (circleS (tab1 ts p.val)) t.val).2
  c := 7

def mkModel (d hv h2 nh nkv dff nv : ℕ) (emb head : ℕ → ℕ → Fp) (gf : ℕ → Fp)
    (layers : List (Layer Fp d hv dff (Fin nh) (Fin nkv) (Fin h2))) :
    Model Fp d hv dff (Fin nh) (Fin nkv) (Fin h2) (Fin nv) where
  emb := fun v j => emb v.val j.val
  head := Matrix.of fun (v : Fin nv) (j : Fin d) => head v.val j.val
  gf := fun j => gf j.val
  layers := layers

section Rows

variable {T d hv dff nh nkv h2 nv : ℕ}

def streamRows (h : St Fp T d) : List (List Fp) :=
  (List.finRange T).map fun t => (List.finRange d).map fun j => h t j

/-- The queries, as the array `(heads, positions, head dimension)` of `model.py` flattened to rows `(head, position)`. -/
def qRows (E : Env Fp T (Fin nh) (Fin nkv) (Fin h2)) (L : Layer Fp d hv dff (Fin nh) (Fin nkv) (Fin h2)) (h : St Fp T d) :
    List (List Fp) :=
  (List.finRange nh).flatMap fun a => (List.finRange T).map fun t =>
    (List.finRange 2).flatMap fun r => (List.finRange h2).map fun p => qv E L h t a p r

/-- The keys, rows `(group, position)` of the array `(groups, positions, head dimension)`. -/
def kRows (E : Env Fp T (Fin nh) (Fin nkv) (Fin h2)) (L : Layer Fp d hv dff (Fin nh) (Fin nkv) (Fin h2)) (h : St Fp T d) :
    List (List Fp) :=
  (List.finRange nkv).flatMap fun g => (List.finRange T).map fun s =>
    (List.finRange 2).flatMap fun r => (List.finRange h2).map fun p => kv E L h s g p r

def vRows (E : Env Fp T (Fin nh) (Fin nkv) (Fin h2)) (L : Layer Fp d hv dff (Fin nh) (Fin nkv) (Fin h2)) (h : St Fp T d) :
    List (List Fp) :=
  (List.finRange nkv).flatMap fun g => (List.finRange T).map fun s => (List.finRange hv).map fun r => vv E L h s g r

/-- The weights of a layer in the flat layout of `model.py` (rows of `wq`: head, then component, then plane). -/
def emitLayer (label : String) (L : Layer Fp d hv dff (Fin nh) (Fin nkv) (Fin h2)) : IO Unit := do
  emit s!"{label}.wq" ((List.finRange nh).flatMap fun a => (List.finRange 2).flatMap fun r => (List.finRange h2).map fun p =>
    (List.finRange d).map fun j => L.Wq a p r j)
  emit s!"{label}.wk" ((List.finRange nkv).flatMap fun g => (List.finRange 2).flatMap fun r => (List.finRange h2).map fun p =>
    (List.finRange d).map fun j => L.Wk g p r j)
  emit s!"{label}.wv" ((List.finRange nkv).flatMap fun g => (List.finRange hv).map fun r =>
    (List.finRange d).map fun j => L.Wv g r j)
  emit s!"{label}.wo" ((List.finRange d).map fun i => (List.finRange nh).flatMap fun a => (List.finRange hv).map fun r => L.Wo a i r)
  emit s!"{label}.wg" ((List.finRange dff).map fun i => (List.finRange d).map fun j => L.Wg i j)
  emit s!"{label}.wu" ((List.finRange dff).map fun i => (List.finRange d).map fun j => L.Wu i j)
  emit s!"{label}.wd" ((List.finRange d).map fun i => (List.finRange dff).map fun j => L.Wd i j)
  emit s!"{label}.an" [(List.finRange d).map fun j => L.ga j]
  emit s!"{label}.mn" [(List.finRange d).map fun j => L.gm j]

def emitWeights (label : String) (M : Model Fp d hv dff (Fin nh) (Fin nkv) (Fin h2) (Fin nv)) : IO Unit := do
  emit s!"{label}.emb" ((List.finRange nv).map fun v => (List.finRange d).map fun j => M.emb v j)
  emit s!"{label}.head" ((List.finRange nv).map fun v => (List.finRange d).map fun j => M.head v j)
  emit s!"{label}.gf" [(List.finRange d).map fun j => M.gf j]
  let mut i := 0
  for L in M.layers do
    emitLayer s!"{label}.l{i}" L
    i := i + 1

/-- Run the model layer by layer: for each layer the queries, keys and values of the cache (from the stream that enters it), the stream after it; then the logits. The stream after a layer
is kept as a table, so that the layer function is evaluated once per entry. -/
def emitRun (label : String) (E : Env Fp T (Fin nh) (Fin nkv) (Fin h2))
    (M : Model Fp d hv dff (Fin nh) (Fin nkv) (Fin h2) (Fin nv)) (toks : Fin T → Fin nv) : IO Unit := do
  let mut h : St Fp T d := fun t => M.emb (toks t)
  let mut i := 0
  for L in M.layers do
    emit s!"{label}.q{i}" (qRows E L h)
    emit s!"{label}.k{i}" (kRows E L h)
    emit s!"{label}.v{i}" (vRows E L h)
    let tbl := (List.finRange T).map fun t => (List.finRange d).map fun j => layerFn E L h t j
    h := fun t j => tab tbl t.val j.val
    emit s!"{label}.h{i}" tbl
    i := i + 1
  emit s!"{label}.logits" ((List.finRange T).map fun t => (List.finRange nv).map fun v =>
    (M.head *ᵥ nrm E.F M.gf (h t)) v)

/-- The logits by the definition `logits` itself (no tables): expensive, for one layer. -/
def emitLogitsDef (label : String) (E : Env Fp T (Fin nh) (Fin nkv) (Fin h2))
    (M : Model Fp d hv dff (Fin nh) (Fin nkv) (Fin h2) (Fin nv)) (toks : Fin T → Fin nv) : IO Unit :=
  emit s!"{label}.logits_def" ((List.finRange T).map fun t => (List.finRange nv).map fun v => logits E M toks t v)

end Rows

end Lmnf.Runtime
