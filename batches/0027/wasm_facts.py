#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
"""WebAssembly 1.0 module -> Datalog facts (the resource-relevant ground of batch 0027).

Reads a binary module (https://www.w3.org/TR/wasm-core-1/ chapter 5) with nothing but the standard library
and writes one tab-separated file per relation, for Souffle (closure.dl):

  wmodule(name)                          the module (the file stem)
  wsection(id, count, bytes)             the sections present, by id, with their item count and size in bytes
  wtype(idx, nparams, nresults)          function types
  wsig(idx, sig)                         the signature of a type as text: parameters, '>', results (i32,i32>i32)
  wimport(module, field, kind, idx)      every import (kind: func table mem global); idx is the index it takes in its space
  wexport(name, kind, idx)               every export
  wfunc(idx, typeidx, imported)          every function (imports first)
  wmemory(idx, min_pages, max_pages)     max_pages is -1 when the module declares none
  wtable(idx, min, max)                  max is -1 when the module declares none
  wglobal(idx, valtype, mutable, imported, init)   init is the i32.const value of a defined global, -1 for any other initializer
  welem(offset, funcidx)                 the functions that the active element segments put into table 0, at offset + position
  wcallind(caller, typeidx)              call_indirect with the type typeidx
  wdata(idx, offset, bytes)              active data segments; offset is -1 when not an i32.const
  wstart(funcidx)                        the start function, if any
  wcall(caller, callee)                  direct calls
  wuses(func, what)                      what in {memory_grow, memory_size, call_indirect, loop, br_table, unreachable, load, store}
  wnote(what, n)                         what the extractor could not read (element segments in another encoding)
  wcode(func, bytes)                     size of each function body in bytes

Usage: wasm_facts.py MODULE.wasm OUTDIR
"""

import os
import sys

VALTYPE = {0x7F: "i32", 0x7E: "i64", 0x7D: "f32", 0x7C: "f64"}
KIND = {0: "func", 1: "table", 2: "mem", 3: "global"}


class Reader:
    def __init__(self, data, pos=0, end=None):
        self.d, self.p, self.end = data, pos, len(data) if end is None else end

    def byte(self):
        b = self.d[self.p]
        self.p += 1
        return b

    def u32(self):
        r = shift = 0
        while True:
            b = self.byte()
            r |= (b & 0x7F) << shift
            if not b & 0x80:
                return r
            shift += 7

    def s32(self):
        return self.sleb()

    def sleb(self):
        r = shift = 0
        while True:
            b = self.byte()
            r |= (b & 0x7F) << shift
            shift += 7
            if not b & 0x80:
                if shift < 64 and b & 0x40:
                    r -= 1 << shift
                return r

    def name(self):
        n = self.u32()
        s = self.d[self.p : self.p + n].decode("utf-8", "replace")
        self.p += n
        return s

    def limits(self):
        flag = self.byte()
        mn = self.u32()
        mx = self.u32() if flag & 1 else -1
        return mn, mx

    def skip_const(self):
        """Step over a constant expression of any type, up to and including its end opcode (0x0B)."""
        op = self.byte()
        if op in {0x41, 0x42}:
            self.sleb()
        elif op == 0x43:
            self.p += 4
        elif op == 0x44:
            self.p += 8
        elif op == 0x23:
            self.u32()
        assert self.byte() == 0x0B

    def const_i32(self):
        """A constant expression `i32.const k; end` -> k, or None for any other expression."""
        save = self.p
        op = self.byte()
        val = None
        if op == 0x41:
            val = self.sleb() & 0xFFFFFFFF
        elif op in (0x23,):
            self.u32()
        else:
            # skip an unknown constant expression up to its end opcode
            while self.d[self.p - 1] != 0x0B and self.p < self.end:
                self.p += 1
            return None
        if self.byte() != 0x0B:
            self.p = save
            return None
        return val


def decode_body(r, end, facts, fidx):
    """Walk the instructions of one function body (MVP opcodes), recording calls and resource-relevant uses."""
    uses = set()
    ncalls = set()
    ngroups = r.u32()
    for _ in range(ngroups):
        r.u32()
        r.byte()
    d = r.d
    while r.p < end:
        op = d[r.p]
        r.p += 1
        if op in (0x02, 0x03, 0x04):  # block loop if: blocktype
            r.p += 1
            if op == 0x03:
                uses.add("loop")
        elif op in (0x0C, 0x0D):  # br br_if
            r.u32()
        elif op == 0x0E:  # br_table
            n = r.u32()
            for _ in range(n + 1):
                r.u32()
            uses.add("br_table")
        elif op == 0x10:  # call
            ncalls.add(r.u32())
        elif op == 0x11:  # call_indirect
            facts["wcallind"].append((fidx, r.u32()))
            r.p += 1
            uses.add("call_indirect")
        elif op == 0x00:
            uses.add("unreachable")
        elif op in (0x20, 0x21, 0x22, 0x23, 0x24):
            r.u32()
        elif 0x28 <= op <= 0x35:  # loads
            r.u32()
            r.u32()
            uses.add("load")
        elif 0x36 <= op <= 0x3E:  # stores
            r.u32()
            r.u32()
            uses.add("store")
        elif op == 0x3F:
            r.p += 1
            uses.add("memory_size")
        elif op == 0x40:
            r.p += 1
            uses.add("memory_grow")
        elif op in {0x41, 0x42}:
            r.sleb()
        elif op == 0x43:
            r.p += 4
        elif op == 0x44:
            r.p += 8
        # every other MVP opcode (drop select nop else end return numeric) has no immediate
    facts["wcall"].extend((fidx, c) for c in sorted(ncalls))
    facts["wuses"].extend((fidx, u) for u in sorted(uses))


def parse(path):
    with open(path, "rb") as fh:
        data = fh.read()
    assert data[:4] == b"\0asm", "not a WebAssembly binary"
    version = int.from_bytes(data[4:8], "little")
    f = {
        k: []
        for k in (
            "wmodule",
            "wsection",
            "wtype",
            "wsig",
            "wimport",
            "wexport",
            "wfunc",
            "wmemory",
            "wtable",
            "wglobal",
            "welem",
            "wdata",
            "wstart",
            "wcall",
            "wcallind",
            "wuses",
            "wcode",
            "wversion",
            "wnote",
        )
    }
    f["wmodule"].append((os.path.splitext(os.path.basename(path))[0],))
    f["wversion"].append((version,))
    r = Reader(data, 8)
    imports = {"func": 0, "table": 0, "mem": 0, "global": 0}
    nfunc_imp = 0
    func_types = []
    bodies = []
    while r.p < len(data):
        sid = r.byte()
        size = r.u32()
        end = r.p + size
        sub = Reader(data, r.p, end)
        count = 0
        if sid == 1:
            count = sub.u32()
            for i in range(count):
                assert sub.byte() == 0x60
                np_ = sub.u32()
                params = [VALTYPE[sub.byte()] for _ in range(np_)]
                nr = sub.u32()
                results = [VALTYPE[sub.byte()] for _ in range(nr)]
                f["wtype"].append((i, np_, nr))
                f["wsig"].append((i, ",".join(params) + ">" + ",".join(results)))
        elif sid == 2:
            count = sub.u32()
            for _ in range(count):
                mod, field, kind = sub.name(), sub.name(), KIND[sub.byte()]
                if kind == "func":
                    t = sub.u32()
                    f["wfunc"].append((imports["func"], t, 1))
                    func_types.append(t)
                elif kind == "table":
                    sub.byte()
                    mn, mx = sub.limits()
                    f["wtable"].append((imports["table"], mn, mx))
                elif kind == "mem":
                    mn, mx = sub.limits()
                    f["wmemory"].append((imports["mem"], mn, mx))
                else:
                    vt, mut = VALTYPE[sub.byte()], sub.byte()
                    f["wglobal"].append((imports["global"], vt, mut, 1, -1))
                f["wimport"].append((mod, field, kind, imports[kind]))
                imports[kind] += 1
            nfunc_imp = imports["func"]
        elif sid == 3:
            count = sub.u32()
            for i in range(count):
                t = sub.u32()
                f["wfunc"].append((nfunc_imp + i, t, 0))
                func_types.append(t)
        elif sid == 4:
            count = sub.u32()
            for i in range(count):
                sub.byte()
                mn, mx = sub.limits()
                f["wtable"].append((imports["table"] + i, mn, mx))
        elif sid == 5:
            count = sub.u32()
            for i in range(count):
                mn, mx = sub.limits()
                f["wmemory"].append((imports["mem"] + i, mn, mx))
        elif sid == 6:
            count = sub.u32()
            for i in range(count):
                vt, mut = VALTYPE[sub.byte()], sub.byte()
                init = sub.const_i32() if vt == "i32" else (sub.skip_const() or None)
                f["wglobal"].append((imports["global"] + i, vt, mut, 0, -1 if init is None else init))
        elif sid == 7:
            count = sub.u32()
            for _ in range(count):
                n = sub.name()
                k = KIND[sub.byte()]
                f["wexport"].append((n, k, sub.u32()))
        elif sid == 8:
            f["wstart"].append((sub.u32(),))
            count = 1
        elif sid == 9:
            count = sub.u32()
            for _ in range(count):
                flag = sub.u32()
                if flag != 0:
                    f["wnote"].append(("element_segment_flag", flag))
                    break
                off = sub.const_i32()
                for k in range(sub.u32()):
                    fi = sub.u32()
                    if off is not None:
                        f["welem"].append((off + k, fi))
        elif sid == 10:
            count = sub.u32()
            for i in range(count):
                sz = sub.u32()
                bodies.append((nfunc_imp + i, sub.p, sub.p + sz))
                f["wcode"].append((nfunc_imp + i, sz))
                sub.p += sz
        elif sid == 11:
            count = sub.u32()
            for i in range(count):
                sub.u32()
                off = sub.const_i32()
                n = sub.u32()
                sub.p += n
                f["wdata"].append((i, -1 if off is None else off, n))
        f["wsection"].append((sid, count, size))
        r.p = end
    for fidx, a, b in bodies:
        decode_body(Reader(data, a, b), b, f, fidx)
    return f


def write(f, out):
    os.makedirs(out, exist_ok=True)
    for k, rows in f.items():
        with open(os.path.join(out, k + ".facts"), "w") as fh:
            for row in rows:
                fh.write("\t".join(str(x) for x in row) + "\n")


if __name__ == "__main__":
    path, out = sys.argv[1:3]
    facts = parse(path)
    write(facts, out)
    for k, v in facts.items():
        print(f"{k:10s} {len(v):8d}")
