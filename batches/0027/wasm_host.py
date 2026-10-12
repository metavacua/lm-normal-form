#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Batch 0027: the host of the module of batches/0027/wasm, in wasmtime (the Python binding).

Three groups of checks, each against something that was fixed before the run:
  closure   the numbers that closure.dl derives from the module's structure (imports, exports, memory, table, stack) against what wasmtime does:
            the memory grows page by page to the least of the specification's, the module's and the host's bound and then fails with -1; a load or store
            at or beyond the end of the memory traps; an index outside the table, or of another signature, traps; a recursion and a loop end in a trap
            (the stack, the fuel); the instantiation with an empty linker succeeds exactly when the module has no imports, and a module without a start
            function uses no fuel when it is instantiated; two instances share nothing.
  kernel    tern_i8 against exact Python integers and under permutations of the columns (bit-exact, a theorem about integer addition), tern_f32 under the
            same permutations (not a theorem: the count of outputs that change is recorded), and the extreme sums against the bound that fits an i32.
  canon     nf_units, the compiled canonical form of a block of gated units, against ternary.py's leading-trit normalizer and sort on every ordered block of
            the configurations, against the orbits found by union-find over the same blocks (the same partition), and its own idempotence.
Each check has an id, the prediction, the observation and a verdict: supported (they agree), refuted (they differ), unknown (not observable here).

Usage: wasm_host.py MODULE.wasm OUT.json --facts DIR --predict DIR [--controls DIR] [--backend wasm|python] [--only closure,kernel,canon]
  --facts DIR    the directory that wasm_facts.py wrote; scenario.facts in it (written by the workflow) names the memory limits of the runs (name, KiB)
  --predict DIR  the output of closure.dl (predict.csv, may_grow.csv, fixed_memory.csv)
  --controls DIR the directory with the compiled control modules bulk.wasm and simd.wasm: adds the checks that the engine with the switchable post-1.0
                 proposals off (LOCKDOWN) refuses what the default engine compiles
  --backend python  run the kernel and canon groups against a Python model of the exports, to test this script without the module
"""

import csv
import importlib.metadata
import json
import os
import random
import re
import struct
import sys
import time

import wasmtime as w

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ternary

PAGE = 65536
FULL_FUEL = 1 << 60  # a store with fuel metering starts with none: every instance is given this much unless a check says otherwise
T = (-1, 0, 1)
CANON_CONFIGS = [(1, 1, 3), (1, 2, 2), (2, 1, 2)]
W_OFF, X_OFF, Y_OFF = 0, 8192, 32768

TRAP_KINDS = [
    ("oob_memory", r"out of bounds memory access|memory fault"),
    ("unreachable", r"unreachable"),
    ("fuel", r"fuel"),
    ("stack_exhausted", r"call stack exhausted|stack overflow"),
    ("oob_table", r"undefined element|out of bounds table|table index is out of bounds"),
    ("null_element", r"uninitialized element"),
    ("sig_mismatch", r"indirect call type mismatch|signature mismatch"),
    ("div_zero", r"divide by zero"),
]


def i32(x):
    x &= 0xFFFFFFFF
    return x - (1 << 32) if x >= (1 << 31) else x


def u32(x):
    return x & 0xFFFFFFFF


def kind_of(e):
    msg = str(e).lower()
    for k, p in TRAP_KINDS:
        if re.search(p, msg):
            return k
    return "other: " + msg.strip().split("\n")[0][:100]


def f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


class Checks:
    def __init__(self):
        self.rows = []

    def add(self, cid, claim, predicted, observed):
        verdict = "unknown" if observed is None or predicted is None else ("supported" if predicted == observed else "refuted")
        self.rows.append({"id": cid, "claim": claim, "predicted": predicted, "observed": observed, "verdict": verdict})
        print(f"{verdict:9s} {cid}: predicted {predicted!r}, observed {observed!r}", flush=True)

    def counts(self):
        c = {"supported": 0, "refuted": 0, "unknown": 0}
        for r in self.rows:
            c[r["verdict"]] += 1
        return c


def read_predict(d):
    pred = {}
    with open(os.path.join(d, "predict.csv")) as f:
        for k, v in csv.reader(f, delimiter="\t"):
            pred.setdefault(k, []).append(int(v))
    sets = {}
    for name in ("may_grow", "fixed_memory"):
        p = os.path.join(d, name + ".csv")
        if os.path.exists(p):
            with open(p) as f:
                sets[name] = {line.rstrip("\n") for line in f}
        else:
            sets[name] = set()
    return pred, sets


def read_scenarios(d):
    out = {}
    with open(os.path.join(d, "scenario.facts")) as f:
        for n, k in csv.reader(f, delimiter="\t"):
            out[n] = int(k)
    return out


# The proposals after WebAssembly 1.0 (and mutable globals, which the engine always has) that the binding can switch off, in the order that
# lets each be switched off after the ones that depend on it (simd needs relaxed simd off first). Reference types are not in the list: the
# engine's component-model-async feature, which the binding does not expose, requires them, and an engine built without them while that
# feature is on aborts the process.
LOCKDOWN = (
    "wasm_gc",
    "wasm_exceptions",
    "wasm_function_references",
    "wasm_stack_switching",
    "wasm_component_model",
    "wasm_component_model_implements",
    "wasm_component_model_map",
    "wasm_relaxed_simd",
    "wasm_simd",
    "wasm_threads",
    "shared_memory",
    "wasm_memory64",
    "wasm_multi_memory",
    "wasm_custom_page_sizes",
    "wasm_tail_call",
    "wasm_wide_arithmetic",
    "wasm_bulk_memory",
    "wasm_multi_value",
)


def lock_down(cfg):
    """Switch off every proposal after WebAssembly 1.0 that the binding can; returns (switched off, not switched off)."""
    applied, skipped = [], []
    for name in LOCKDOWN:
        try:
            setattr(cfg, name, False)
            applied.append(name)
        except Exception as e:
            skipped.append(f"{name}: {e}")
    return applied, skipped


class Runner:
    """Instances of one module, each in a store of its own."""

    def __init__(self, w, path, fuel=False, stack_kib=None, lockdown=False):
        self.w = w
        cfg = w.Config()
        self.lockdown = lock_down(cfg) if lockdown else ([], [])
        if fuel:
            cfg.consume_fuel = True
        if stack_kib is not None:
            try:
                cfg.max_wasm_stack = stack_kib * 1024
            except Exception as e:
                print("max_wasm_stack not set:", e)
        self.engine = w.Engine(cfg)
        self.module = w.Module.from_file(self.engine, path)
        self.fuel = fuel

    def instantiate(self, limit_kib=None, fuel=None):
        w = self.w
        store = w.Store(self.engine)
        if limit_kib is not None:
            store.set_limits(memory_size=limit_kib * 1024)
        if self.fuel:
            store.set_fuel(fuel if fuel is not None else FULL_FUEL)
        inst = w.Linker(self.engine).instantiate(store, self.module)
        return Inst(self, store, inst)


class Inst:
    def __init__(self, runner, store, inst):
        self.r, self.store, self.inst = runner, store, inst
        self.ex = inst.exports(store)
        self.mem = self.ex["memory"]

    def names(self):
        return [e.name for e in self.r.module.exports]

    def has(self, name):
        return name in self.names()

    def call(self, name, *args):
        return self.ex[name](self.store, *[i32(a) for a in args])

    def try_call(self, name, *args):
        """(result, None) or (None, trap kind)."""
        try:
            return self.call(name, *args), None
        except Exception as e:
            return None, kind_of(e)

    def pages(self):
        return self.mem.size(self.store)

    def write(self, off, data):
        self.mem.write(self.store, bytes(data), off)

    def read(self, off, n):
        return bytes(self.mem.read(self.store, off, off + n))

    def fuel_left(self):
        if hasattr(self.store, "get_fuel"):
            return self.store.get_fuel()
        return FULL_FUEL - self.store.fuel_consumed()


# ---------------------------------------------------------------------------------------------------------------------------- closure


def group_closure(run, checks, pred, sets, scen):
    w = run.w
    mod = run.module

    def P(k):
        return pred[k][0] if k in pred else None

    # the interface of the module
    n_imp = len(mod.imports)
    checks.add("imports", "the number of imports", P("imports"), n_imp)
    funcs = [e for e in mod.exports if isinstance(e.type, w.FuncType)]
    mems = [e for e in mod.exports if isinstance(e.type, w.MemoryType)]
    checks.add("exports_func", "the number of exported functions", P("exports_func"), len(funcs))
    checks.add("exports_mem", "the number of exported memories", P("exports_mem"), len(mems))
    checks.add(
        "exports_table",
        "the number of exported tables (reach assumes the host cannot write the table)",
        P("exports_table"),
        sum(isinstance(e.type, w.TableType) for e in mod.exports),
    )
    checks.add("exports_global", "the number of exported globals", P("exports_global"), sum(isinstance(e.type, w.GlobalType) for e in mod.exports))

    # the instantiation with a linker that defines nothing succeeds exactly when there is no import
    try:
        base = run.instantiate()
        instantiates = 1
    except Exception as e:
        instantiates = 0
        print("instantiation with an empty linker failed:", str(e)[:120])
    checks.add(
        "instantiates_with_an_empty_linker",
        "the module instantiates with a linker that defines nothing exactly when it has no imports",
        None if P("imports") is None else int(P("imports") == 0),
        instantiates,
    )
    if not instantiates:
        return {"note": "the module needs imports; the remaining closure checks do not apply"}
    if run.fuel and P("start_function") == 0:
        checks.add(
            "no_start_function_nothing_runs_at_instantiation",
            "the module has no start function (closure.dl) and instantiating it uses no fuel: no instruction of the guest runs before the host calls",
            0,
            FULL_FUEL - base.fuel_left(),
        )
    checks.add(
        "data_segments_fit_the_initial_memory",
        "closure.dl derives that the active data segments end inside the initial memory, and the instantiation, which checks them, succeeded",
        P("static_fits"),
        1,
    )
    mt = base.mem.type(base.store)
    lim = mt.limits
    max_pages = getattr(lim, "max", None)
    checks.add("mem_initial_kib", "the initial memory", P("mem_initial_kib"), base.pages() * 64)
    checks.add(
        "mem_declared_max_kib",
        "the maximum that the module declares (-1: none)",
        P("mem_declared_max_kib"),
        -1 if max_pages is None else max_pages * 64,
    )

    detail = {}

    # the memory: the end, and the growth
    size = base.pages() * PAGE
    _, k_in = base.try_call("probe_load8", size - 1)
    _, k_out = base.try_call("probe_load8", size)
    _, k_neg = base.try_call("probe_load8", -1)
    _, k_st = base.try_call("probe_store8", size, 1)
    checks.add("load_last_byte", "a load of the last byte of the memory succeeds", True, k_in is None)
    checks.add("load_past_end_traps", "a load at the end of the memory traps (out of bounds)", "oob_memory", k_out)
    checks.add("load_at_minus_one_traps", "a load at address 0xFFFFFFFF traps (out of bounds)", "oob_memory", k_neg)
    checks.add("store_past_end_traps", "a store at the end of the memory traps (out of bounds)", "oob_memory", k_st)
    before = base.pages()
    grown = u32(base.call("probe_mem_grow", 1))
    checks.add("grow_returns_old_size", "memory.grow returns the previous size in pages", before, grown)
    checks.add(
        "grow_moves_the_end",
        "after a growth by one page the old end is addressable and the new end is not",
        [None, "oob_memory"],
        [base.try_call("probe_load8", size)[1], base.try_call("probe_load8", size + PAGE)[1]],
    )

    for name, lim_kib in sorted(scen.items()):
        key_f, key_r = f"final_kib_{name}", f"rejected_{name}"
        rejected_pred = P(key_r)
        try:
            inst = run.instantiate(limit_kib=lim_kib)
            rejected_obs = 0
        except Exception as e:
            rejected_obs = 1
            detail[f"scenario_{name}_error"] = str(e).strip().split("\n")[0][:160]
        checks.add(
            key_r,
            f"scenario {name} ({lim_kib} KiB): the instantiation is refused when the initial memory exceeds the limit",
            rejected_pred,
            rejected_obs,
        )
        if rejected_obs:
            continue
        start_pages = inst.pages()
        seq = []
        for _ in range(70000):
            r = u32(inst.call("probe_mem_grow", 1))
            if r == 0xFFFFFFFF:
                break
            seq.append(r)
        end_pages = inst.pages()
        final_kib = end_pages * 64
        r_after = u32(inst.call("probe_mem_grow", 1))
        r_big = u32(inst.call("probe_mem_grow", 100000))
        detail[f"scenario_{name}"] = {"growths": len(seq), "final_pages": inst.pages(), "monotone": seq == sorted(seq)}
        checks.add(
            key_f,
            f"scenario {name} ({lim_kib} KiB): the memory grows page by page to the least of the three bounds",
            P(key_f),
            final_kib,
        )
        checks.add(
            f"grow_fails_with_minus_one_{name}",
            f"scenario {name}: at the bound memory.grow returns -1 (no trap) and so does a growth by 100000 pages",
            [0xFFFFFFFF, 0xFFFFFFFF],
            [r_after, r_big],
        )
        checks.add(
            f"growths_are_consecutive_{name}",
            f"scenario {name}: the successful growths return start, start + 1, ... in order, and the size at the end is start plus their number",
            True,
            seq == list(range(start_pages, start_pages + len(seq))) and end_pages == start_pages + len(seq),
        )

    # the table: the smallest index that is out of bounds is the declared minimum
    if base.has("probe_call"):
        lo, hi = 0, 1 << 20
        _, k_hi = base.try_call("probe_call", hi, 1)
        while lo < hi and k_hi == "oob_table":
            mid = (lo + hi) // 2
            _, k = base.try_call("probe_call", mid, 1)
            if k == "oob_table":
                hi = mid
            else:
                lo = mid + 1
        checks.add(
            "table_size",
            "the smallest table index that is out of bounds is the declared minimum",
            P("table_size"),
            hi if k_hi == "oob_table" else None,
        )
        a0, a1, a2 = (u32(base.call(f"probe_addr{i}")) for i in range(3))
        r0, _ = base.try_call("probe_call", a0, 5)
        r1, _ = base.try_call("probe_call", a1, 5)
        _, k2 = base.try_call("probe_call", a2, 5)
        checks.add(
            "indirect_call_ok",
            "call_indirect on the index of a function of the named signature runs it",
            [6, 10],
            [r0, r1],
        )
        checks.add(
            "indirect_call_wrong_signature",
            "call_indirect on the index of a function of another signature traps",
            "sig_mismatch",
            k2,
        )
        detail["addr"] = [a0, a1, a2]

    # the stack and the static data
    if base.has("buf_ptr"):
        bp, bl = u32(base.call("buf_ptr")), u32(base.call("buf_len"))
        stack = P("stack_kib")
        checks.add(
            "static_above_stack",
            "the scratch buffer lies above the shadow stack (--stack-first)",
            True,
            stack is not None and bp >= stack * 1024,
        )
        checks.add("static_inside_memory", "the scratch buffer lies inside the initial memory", True, bp + bl <= before * PAGE)
        if P("stack_clear") == 1 and stack is not None:
            checks.add(
                "stack_area_is_zero_after_instantiation",
                "closure.dl derives that no data segment lies below the initial stack pointer, and the bytes below it are zero in a new instance",
                True,
                not any(run.instantiate().read(0, stack * 1024)),
            )
        detail["buf"] = [bp, bl]

    # recursion ends in a trap: the shadow stack (below address 0 is out of bounds) or the stack of the engine
    if base.has("probe_recurse"):
        fresh = run.instantiate()
        r10, _ = fresh.try_call("probe_recurse", 10)
        checks.add(
            "recurse_small",
            "a recursion of depth 10 returns the sum of k mod 256",
            sum(k % 256 for k in range(1, 11)),
            r10,
        )
        lo, hi = 10, 1 << 24
        best = lo
        while lo <= hi:
            mid = (lo + hi) // 2
            probe = run.instantiate()
            _, k = probe.try_call("probe_recurse", mid)
            if k is None:
                best, lo = mid, mid + 1
            else:
                hi = mid - 1
        deep = run.instantiate()
        _, kd = deep.try_call("probe_recurse", 1 << 24)
        checks.add(
            "recurse_deep_traps",
            "a recursion of depth 2^24 ends in a trap (the shadow stack or the engine's stack)",
            True,
            kd in ("oob_memory", "stack_exhausted"),
        )
        detail["recursion"] = {"deepest_without_trap": best, "trap_kind_at_2^24": kd}
        bound = P("recursion_frames_bound")
        if bound is not None and kd == "oob_memory":
            # a recursion of depth n has n + 1 frames of `recurse`; the bound is the stack divided by the frame that its prologue takes
            frames = best + 1
            checks.add(
                "recursion_frames_within_derived_bound",
                "when the shadow stack ends the recursion, its frames do not exceed (stack) / (the frame of its prologue), derived from the binary",
                True,
                frames <= bound,
            )
            checks.add(
                "recursion_frames_equal_derived_bound",
                "and they equal it: the recursive function is the only user of the shadow stack (an observation; this check was added after the observation)",
                bound,
                frames,
            )

    # fuel: a loop is bounded by it
    if run.fuel and base.has("probe_loop"):
        used = {}
        for n in (100, 1100, 10100):
            f = run.instantiate(fuel=10_000_000)
            r = f.call("probe_loop", n)
            used[n] = 10_000_000 - f.fuel_left()
            assert r == (n * (n - 1) // 2) & 0xFFFFFFFF or i32(r) == i32(n * (n - 1) // 2)
        slope = (used[1100] - used[100]) / 1000
        affine = used[10100] - used[1100] == 9 * (used[1100] - used[100])
        checks.add(
            "fuel_is_affine_in_the_iterations",
            "the fuel that a loop uses is affine in its iterations (checked at 100, 1100 and 10100; the slope is recorded, not predicted)",
            True,
            affine,
        )
        f = run.instantiate(fuel=1_000_000)
        _, kf = f.try_call("probe_loop", 2_000_000_000)
        checks.add("loop_ends_in_fuel_trap", "a loop of 2*10^9 iterations on 10^6 units of fuel ends in a trap", "fuel", kf)
        detail["fuel"] = {"used": used, "per_iteration": slope}

    # a trap
    if base.has("probe_trap"):
        _, kt = run.instantiate().try_call("probe_trap")
        checks.add("unreachable_traps", "the unreachable instruction traps", "unreachable", kt)

    # nothing is shared between two instances
    if base.has("buf_ptr"):
        a, b = run.instantiate(), run.instantiate()
        p = u32(a.call("buf_ptr"))
        a.call("probe_store8", p, 7)
        checks.add(
            "instances_share_nothing",
            "a byte written in one instance is not seen in another",
            [7, 0],
            [u32(a.call("probe_load8", p)), u32(b.call("probe_load8", p))],
        )

    # which exports change the size of the memory
    sweep = {
        "buf_ptr": (),
        "buf_len": (),
        "probe_mem_size": (),
        "probe_loop": (10,),
        "probe_recurse": (5,),
        "probe_addr0": (),
        "probe_addr1": (),
        "probe_addr2": (),
        "tern_i8": (2, 8, W_OFF, X_OFF, Y_OFF),
        "tern_f32": (2, 8, W_OFF, X_OFF, Y_OFF),
        "nf_units": (2, 1, 1, 0),
        "probe_mem_grow": (1,),
        "probe_load8": (None,),
        "probe_store8": (None, 1),
    }
    inst = run.instantiate()
    changed = set()
    bp = u32(inst.call("buf_ptr")) if inst.has("buf_ptr") else 0
    for name, args in sweep.items():
        if not inst.has(name):
            continue
        call_args = tuple(bp if a is None else a for a in args)
        s0 = inst.pages()
        inst.try_call(name, *call_args)
        if inst.pages() != s0:
            changed.add(name)
    if inst.has("probe_call"):
        s0 = inst.pages()
        inst.try_call("probe_call", u32(inst.call("probe_addr0")), 3)
        if inst.pages() != s0:
            changed.add("probe_call")
    may, fixed = sets["may_grow"], sets["fixed_memory"]
    checks.add(
        "changed_exports_are_predicted",
        "every export that changed the size of the memory is one from which closure.dl reaches memory.grow",
        True,
        changed <= may,
    )
    checks.add(
        "fixed_exports_did_not_change",
        "no export that closure.dl derives to have a fixed memory changed it",
        True,
        not (changed & fixed),
    )
    checks.add(
        "exports_that_may_grow_are_the_ones_that_grew",
        "the exports from which closure.dl reaches memory.grow are exactly the ones that changed the memory here",
        sorted(n for n in may if n != "<start>"),
        sorted(changed),
    )
    checks.add(
        "module_may_grow",
        "closure.dl derives that the memory of the module may grow (some entry reaches memory.grow), and some export changed the size here",
        P("module_may_grow"),
        int(bool(changed)),
    )
    detail["changed_memory"] = sorted(changed)
    return detail


def compiles(w, cfg, path):
    try:
        w.Module.from_file(w.Engine(cfg), path)
        return True
    except Exception:
        return False


def group_features(w, checks, controls):
    """The engine with the switchable post-1.0 proposals off (LOCKDOWN) refuses what the default engine takes: a module with a bulk-memory
    instruction and one with a SIMD value. Only these two proposals are exercised; reference types, sign extension and the non-trapping conversions
    stay accepted by every configuration of the binding."""
    for name, what in (("bulk", "memory.copy (bulk memory)"), ("simd", "i32x4.add (SIMD)")):
        path = os.path.join(controls, f"{name}.wasm")
        if not os.path.exists(path):
            continue
        locked = w.Config()
        lock_down(locked)
        checks.add(f"default_engine_accepts_{name}", f"the default engine compiles a module that uses {what}", True, compiles(w, w.Config(), path))
        checks.add(f"locked_engine_rejects_{name}", "the engine with the switchable post-1.0 proposals off refuses it", False, compiles(w, locked, path))


# ---------------------------------------------------------------------------------------------------------------------------- backends


class WasmBackend:
    def __init__(self, inst):
        self.inst = inst
        self.base = u32(inst.call("buf_ptr"))

    def nf(self, block, n, d, e):
        L = 2 * d + e
        self.inst.write(self.base, block)
        self.inst.call("nf_units", n, d, e, 0)
        return self.inst.read(self.base, n * L)

    def tern_i8(self, W, x):
        rows, cols = len(W), len(W[0])
        self._load(W)
        self.inst.write(self.base + X_OFF, bytes(v & 0xFF for v in x))
        self.inst.call("tern_i8", rows, cols, W_OFF, X_OFF, Y_OFF)
        raw = self.inst.read(self.base + Y_OFF, 4 * rows)
        return list(struct.unpack(f"<{rows}i", raw))

    def tern_f32(self, W, x):
        rows, cols = len(W), len(W[0])
        self._load(W)
        self.inst.write(self.base + X_OFF, struct.pack(f"<{cols}f", *x))
        self.inst.call("tern_f32", rows, cols, W_OFF, X_OFF, Y_OFF)
        raw = self.inst.read(self.base + Y_OFF, 4 * rows)
        return list(struct.unpack(f"<{rows}f", raw))

    def _load(self, W):
        rows, cols = len(W), len(W[0])
        packed = bytearray((rows * cols + 3) // 4)
        for r in range(rows):
            for c in range(cols):
                k = r * cols + c
                packed[k // 4] |= (W[r][c] + 1) << (2 * (k % 4))
        self.inst.write(self.base + W_OFF, bytes(packed))


class PyBackend:
    """A model of the exports in Python, to test this script without the module."""

    def nf(self, block, n, d, e):
        L = 2 * d + e
        units = [tuple(((b + 128) % 256) - 128 for b in block[i * L : (i + 1) * L]) for i in range(n)]
        _, _, _, lead = ternary.unit_space(d, e)
        out = sorted(lead(u) for u in units)
        return bytes(t & 0xFF for u in out for t in u)

    def tern_i8(self, W, x):
        return [sum(t * v for t, v in zip(row, x, strict=True)) for row in W]

    def tern_f32(self, W, x):
        out = []
        for row in W:
            acc = 0.0
            for t, v in zip(row, x, strict=True):
                acc = f32(acc + f32(t * v))
            out.append(acc)
        return out


# ---------------------------------------------------------------------------------------------------------------------------- kernel


def group_kernel(be, checks, *, seed=27, rows=8, cols=48, perms=20):
    rng = random.Random(seed)
    W = [[rng.choice(T) for _ in range(cols)] for _ in range(rows)]
    x8 = [rng.randrange(-128, 128) for _ in range(cols)]
    xf = [f32(rng.uniform(-1.0, 1.0)) for _ in range(cols)]
    ref = [sum(W[r][c] * x8[c] for c in range(cols)) for r in range(rows)]
    base_i8 = be.tern_i8(W, x8)
    checks.add("tern_i8_exact", "tern_i8 equals the exact integer sums", ref, base_i8)
    base_f = be.tern_f32(W, xf)
    bad_i8 = bad_f32 = changed_rows = 0
    for _ in range(perms):
        pi = list(range(cols))
        rng.shuffle(pi)
        Wp = [[row[c] for c in pi] for row in W]
        if be.tern_i8(Wp, [x8[c] for c in pi]) != ref:
            bad_i8 += 1
        got = be.tern_f32(Wp, [xf[c] for c in pi])
        diff = sum(1 for a, b in zip(got, base_f, strict=True) if struct.pack("<f", a) != struct.pack("<f", b))
        changed_rows += diff
        bad_f32 += diff > 0
    checks.add(
        "tern_i8_permutation_invariant",
        f"tern_i8 gives the same sums under {perms} permutations of the columns (exact: integer addition is associative)",
        0,
        bad_i8,
    )
    checks.add(
        "tern_f32_permutation_changes_bits",
        f"tern_f32 changes the bits of some output under some of {perms} permutations of the columns (not a theorem)",
        True,
        changed_rows > 0,
    )
    # large sums: 2^14 columns, a quarter of the bytes of the buffer
    n = 16384
    ones = [[1] * n]
    top = be.tern_i8(ones, [127] * n)[0]
    bot = be.tern_i8(ones, [-128] * n)[0]
    checks.add(
        "tern_i8_extremes",
        f"{n} columns of +1 times 127 and times -128 give the sums 127n and -128n, which fit an i32 (bound 128n = 2^21)",
        [127 * n, -128 * n],
        [top, bot],
    )
    return {"changed_outputs_f32": changed_rows, "permutations_with_a_change_f32": bad_f32}


# ---------------------------------------------------------------------------------------------------------------------------- canonical form


def group_canon(be, checks, configs=CANON_CONFIGS):
    detail = {}
    for d, e, n in configs:
        t0 = time.time()
        units, _, p_sgn, lead = ternary.unit_space(d, e)
        U = len(units)
        N = U**n
        roots, pw = ternary.orbit_roots(U, n, [p_sgn])
        out_of_root = {}
        differs = invariant_bad = idem_bad = 0
        for st in range(N):
            digits = [(st // pw[i]) % U for i in range(n)]
            block = bytes(t & 0xFF for i in digits for t in units[i])
            out = be.nf(block, n, d, e)
            expect = bytes(t & 0xFF for u in sorted(lead(units[i]) for i in digits) for t in u)
            differs += out != expect
            if out_of_root.setdefault(roots[st], out) != out:
                invariant_bad += 1
            idem_bad += be.nf(out, n, d, e) != out
        tag = f"d{d}e{e}n{n}"
        checks.add(
            f"canon_matches_python_{tag}",
            f"nf_units equals leading-trit normalization and sort on all {N} ordered blocks",
            0,
            differs,
        )
        checks.add(
            f"canon_invariant_on_orbits_{tag}",
            "nf_units is constant on every orbit found by union-find over the generators",
            0,
            invariant_bad,
        )
        checks.add(
            f"canon_decisive_{tag}",
            f"nf_units takes {len(out_of_root)} distinct values on {len(out_of_root)} orbits",
            len(out_of_root),
            len(set(out_of_root.values())),
        )
        checks.add(f"canon_idempotent_{tag}", "nf_units applied to its own output changes nothing", 0, idem_bad)
        detail[tag] = {"blocks": N, "orbits": len(out_of_root), "seconds": round(time.time() - t0, 2)}
    return detail


# ---------------------------------------------------------------------------------------------------------------------------- main


def main(argv):
    opt = {}
    it = iter(argv)
    for a in it:
        if a.startswith("--"):
            opt[a[2:]] = next(it)
    wasm_path, out_path = argv[0], argv[1]
    facts, predict = opt["facts"], opt["predict"]
    only = set(opt.get("only", "closure,kernel,canon").split(","))
    backend = opt.get("backend", "wasm")

    pred, sets = read_predict(predict)
    scen = read_scenarios(facts)
    checks = Checks()
    run = Runner(w, wasm_path, fuel=True, stack_kib=256, lockdown=True)
    res = {
        "module": os.path.basename(wasm_path),
        "wasmtime": importlib.metadata.version("wasmtime"),
        "backend": backend,
        "scenarios": scen,
        "engine": {"switched_off": run.lockdown[0], "not_switched_off": run.lockdown[1], "fuel": True, "max_wasm_stack_kib": 256},
    }
    t0 = time.time()
    if "closure" in only:
        res["closure"] = group_closure(run, checks, pred, sets, scen)
        if "controls" in opt:
            group_features(w, checks, opt["controls"])
    be = PyBackend() if backend == "python" else WasmBackend(run.instantiate())
    if "kernel" in only:
        res["kernel"] = group_kernel(be, checks)
    if "canon" in only:
        res["canon"] = group_canon(be, checks)
    res["checks"] = checks.rows
    res["counts"] = checks.counts()
    res["seconds"] = round(time.time() - t0, 2)
    with open(out_path, "w") as f:
        json.dump(res, f, indent=1, sort_keys=True)
    print("counts:", res["counts"])
    return 0 if res["counts"]["refuted"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
