# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
"""A WebAssembly module for batch 0028: its census of blank and empty constructs, its graph of functions as ground
RDF, and the search for weights in its bytes.

The structure comes from batch 0027's `wasm_facts.parse` (WebAssembly 1.0, standard library only); the names from
the module's own `name` section, read here; the instruction counts from `wasm-tools print` (the flat text form, one
instruction to a line), which is the tool the workflow already uses to validate. A function's name is a
declared identifier and is a literal, not a part of its IRI: a function's IRI is its index. The crates named in a
mangled name are found by the length-prefixed crate roots of Rust's v0 mangling (`C`, an optional disambiguator
`s<base62>_`, a decimal length, the name) or the segments of the legacy one; this is a heuristic and the census
says so.

Usage: wasm_tools.py census TAG ARTIFACT MODULE.wasm OUTDIR [WASM_TOOLS]
       wasm_tools.py search MODULE.wasm FILE.bpk OUT.json
       wasm_tools.py selftest [WASM_TOOLS]
"""

import collections
import json
import mmap
import os
import re
import subprocess
import sys

import common
import ground as G

import wasm_facts as wf  # noqa: E402  (batches/0027, on the path through common)

V0_CRATE = re.compile(r"C(?:s[0-9a-zA-Z]*_)?(\d+)_?")
V0_HASHED = re.compile(r"Cs[0-9a-zA-Z]+_\d+")
LEGACY_HASH = re.compile(r"17h[0-9a-f]{16}E$")
SEGMENT = re.compile(r"(\d+)([A-Za-z_][A-Za-z0-9_$.]*)")


def sections(data):
    """(id, start, end, custom name or None) for each section of a binary module."""
    r = wf.Reader(data, 8)
    out = []
    while r.p < len(data):
        sid = r.byte()
        size = r.u32()
        start = r.p
        name = None
        if sid == 0:
            name = r.name()
        out.append((sid, start, start + size, name))
        r.p = start + size
    return out


def function_names(data):
    """The function-name subsection of the `name` section: index -> name (several indices may share a name)."""
    names = {}
    for sid, start, end, nm in sections(data):
        if sid == 0 and nm == "name":
            r = wf.Reader(data, start)
            r.name()
            while r.p < end:
                sub = r.byte()
                size = r.u32()
                s_end = r.p + size
                if sub == 1:
                    for _ in range(r.u32()):
                        i = r.u32()
                        names[i] = r.name()
                r.p = s_end
    return names


def data_segments(data):
    out = []
    for sid, start, end, _ in sections(data):
        if sid == 11:
            r = wf.Reader(data, start)
            for _ in range(r.u32()):
                flag = r.u32()
                off = r.const_i32() if flag == 0 else None
                n = r.u32()
                seg = data[r.p:r.p + n]
                r.p += n
                out.append({"offset": off, "bytes": n, "zero_bytes": seg.count(0), "flag": flag})
    return out


def crates_of(name):
    """The crate names that a mangled name mentions, in order; the first is taken as the defining crate."""
    out = []
    if name.startswith("_R"):
        i = 2
        while i < len(name):
            m = V0_CRATE.match(name, i) if name[i] == "C" else None
            if m:
                n = int(m.group(1))
                out.append(name[m.end():m.end() + n])
                i = m.end() + n
            else:
                i += 1
    elif name.startswith("_ZN"):
        segs = [m.group(2)[:int(m.group(1))] for m in SEGMENT.finditer(name[3:])]
        if segs:
            out.append(segs[0])
    return out


def body_hashes(data):
    """The sha256 of the bytes of each function body, in order of the code section."""
    import hashlib
    out = []
    for sid, start, end, _ in sections(data):
        if sid == 10:
            r = wf.Reader(data, start)
            for _ in range(r.u32()):
                size = r.u32()
                out.append(hashlib.sha256(data[r.p:r.p + size]).hexdigest())
                r.p += size
    return out


def tarjan(n_nodes, edges):
    """Strongly connected components of the call graph, iteratively: (components of size > 1, self-calling functions)."""
    adj = collections.defaultdict(list)
    selfs = set()
    for a, b in edges:
        adj[a].append(b)
        if a == b:
            selfs.add(a)
    index, low, on, stack, comps = {}, {}, set(), [], []
    counter = 0
    for root in list(adj):
        if root in index:
            continue
        work = [(root, iter(adj[root]))]
        index[root] = low[root] = counter
        counter += 1
        stack.append(root)
        on.add(root)
        while work:
            v, it = work[-1]
            adv = False
            for w in it:
                if w not in index:
                    index[w] = low[w] = counter
                    counter += 1
                    stack.append(w)
                    on.add(w)
                    work.append((w, iter(adj[w])))
                    adv = True
                    break
                if w in on:
                    low[v] = min(low[v], index[w])
            if adv:
                continue
            work.pop()
            if work:
                low[work[-1][0]] = min(low[work[-1][0]], low[v])
            if low[v] == index[v]:
                comp = []
                while True:
                    w = stack.pop()
                    on.discard(w)
                    comp.append(w)
                    if w == v:
                        break
                if len(comp) > 1:
                    comps.append(comp)
    return comps, selfs


def instruction_census(path, tool):
    p = subprocess.Popen([tool, "print", path], stdout=subprocess.PIPE, text=True, errors="replace", bufsize=1 << 20)
    c = collections.Counter()
    typed = collections.Counter()
    for line in p.stdout:
        s = line.lstrip()
        if not s or s[0] in "();":
            continue
        tok = s.split(None, 1)[0]
        c[tok] += 1
        if tok in ("block", "loop", "if"):
            rest = s[len(tok):].lstrip()
            typed[(tok, "typed" if rest.startswith("(result") or rest.startswith("(type") else "empty")] += 1
    if p.wait() != 0:
        raise RuntimeError("wasm-tools print failed")
    keys = ("nop", "drop", "unreachable", "select", "br_table", "br", "br_if", "call", "call_indirect", "return",
            "memory.grow", "memory.size", "end", "else")
    out = {k.replace(".", "_"): c[k] for k in keys}
    for t in ("block", "loop", "if"):
        out[f"{t}_empty"], out[f"{t}_typed"] = typed[(t, "empty")], typed[(t, "typed")]
    out["instructions"] = sum(c.values())
    out["top_mnemonics"] = dict(c.most_common(30))
    return out


def census(path, tool):
    data = open(path, "rb").read()
    facts = wf.parse(path)
    names = function_names(data)
    n_funcs = len(facts["wfunc"])
    n_imports = sum(1 for r in facts["wfunc"] if r[2])
    cnt = collections.Counter(names.values())
    crates = collections.Counter()
    for n in names.values():
        cs = crates_of(n)
        crates[cs[0] if cs else "(none)"] += 1
    comps, selfs = tarjan(n_funcs, [(a, b) for a, b in facts["wcall"]])
    uses = collections.Counter(w for _, w in facts["wuses"])
    segs = data_segments(data)
    exports = {name: (kind, idx) for name, kind, idx in facts["wexport"]}
    secs = sections(data)
    out = {
        "file_bytes": len(data), "sections": [[sid, nm, end - start] for sid, start, end, nm in secs],
        "functions": n_funcs, "functions_imported": n_imports, "types": len(facts["wtype"]), "imports": len(facts["wimport"]),
        "exports": sorted(exports), "export_kinds": dict(collections.Counter(k for k, _ in exports.values())),
        "memory_pages_declared": [[a, b] for _, a, b in facts["wmemory"]], "tables": [[a, b] for _, a, b in facts["wtable"]],
        "globals": len(facts["wglobal"]), "table_entries_initialized": len(facts["welem"]),
        "data_segments": len(segs), "data_segment_bytes": sum(s["bytes"] for s in segs),
        "data_segment_zero_bytes": sum(s["zero_bytes"] for s in segs), "start": len(facts["wstart"]),
        "direct_calls": len(facts["wcall"]), "indirect_call_sites": len(facts["wcallind"]),
        "functions_with": {k: uses[k] for k in ("loop", "br_table", "call_indirect", "unreachable", "memory_grow", "load", "store")},
        "functions_named": len(names), "functions_unnamed": n_funcs - len(names),
        "names_repeated": sum(v for v in cnt.values() if v > 1), "names_distinct_repeated": sum(1 for v in cnt.values() if v > 1),
        "names_v0": sum(1 for n in names.values() if n.startswith("_R")),
        "names_legacy": sum(1 for n in names.values() if n.startswith("_ZN")),
        "names_other": sum(1 for n in names.values() if not n.startswith(("_R", "_ZN"))),
        "names_with_build_hash": sum(1 for n in names.values() if V0_HASHED.search(n) or LEGACY_HASH.search(n)),
        "first_crate_histogram": dict(crates.most_common(40)),
        "functions_in_call_cycles": sum(len(c) for c in comps) + len(selfs - {x for c in comps for x in c}),
        "call_cycle_components": len(comps), "largest_call_cycle": max((len(c) for c in comps), default=0),
        "self_calling_functions": len(selfs), "custom_section_names": [nm for sid, _, _, nm in secs if sid == 0],
        "instruction_census": instruction_census(path, tool),
    }
    return out, facts, names


def graph(tag, artifact, facts, names):
    gr = G.Graph(tag, artifact)
    art = G.iri("a", tag, artifact)
    gr.typed(art, "Artifact")
    gr.add(art, G.pred("format"), G.lit("wasm"))
    uses = collections.defaultdict(set)
    for f, w in facts["wuses"]:
        uses[f].add(w)
    size = dict(facts["wcode"])
    exports = collections.defaultdict(list)
    for name, kind, idx in facts["wexport"]:
        if kind == "func":
            exports[idx].append(name)
    for idx, _t, imported in facts["wfunc"]:
        f = G.iri("f", tag, artifact, idx)
        gr.typed(f, "Function")
        gr.add(f, G.pred("inArtifact"), art)
        gr.add(f, G.pred("index"), G.num(idx))
        if idx in names:
            gr.add(f, G.pred("label"), G.lit(names[idx]))
            cs = crates_of(names[idx])
            if cs:
                gr.add(f, G.pred("definedInCrate"), G.lit(cs[0]))
        if idx in size:
            gr.add(f, G.pred("codeBytes"), G.num(size[idx]))
        for w in sorted(uses.get(idx, ())):
            gr.add(f, G.pred("uses"), G.lit(w))
        for e in exports.get(idx, ()):
            gr.add(f, G.pred("exportsAs"), G.lit(e))
    for a, b in facts["wcall"]:
        gr.add(G.iri("f", tag, artifact, a), G.pred("calls"), G.iri("f", tag, artifact, b))
    return gr


def run(tag, artifact, module, outdir, tool="wasm-tools"):
    os.makedirs(outdir, exist_ok=True)
    c, facts, names = census(module, tool)
    c["sha256"] = common.identity.sha256_file(module)
    common.dump(os.path.join(outdir, f"{artifact}-census.json"), c)
    gr = graph(tag, artifact, facts, names)
    G.write(os.path.join(outdir, f"{artifact}-graph.nq.gz"), gr.text())
    common.dump(os.path.join(outdir, f"{artifact}-graph.json"), {
        "triples": len(gr.triples), "collapsed": gr.collapsed(), "graph_sha256": gr.sha256(), "functions": len(facts["wfunc"]),
        "deterministic": gr.sha256() == graph(tag, artifact, facts, names).sha256(), "blank_terms": G.blank_terms(gr.text())})
    common.dump(os.path.join(outdir, f"{artifact}-bodies.json"), body_hashes(open(module, "rb").read()))
    print(json.dumps({k: c[k] for k in ("functions", "direct_calls", "file_bytes", "sha256")}))


def search(module, bpk, out):
    """P13 and its controls: the first 256 bytes of each large tensor of a burnpack, looked for in the module and in the
    burnpack itself, and in a copy of the module with one of them planted."""
    import cbor2
    import struct
    with open(bpk, "rb") as f:
        magic, version, msize = struct.unpack("<IHI", f.read(10))
        meta = cbor2.loads(f.read(msize))
    start = -(-(10 + msize) // 256) * 256
    wasm = open(module, "rb").read()
    found_module, found_bpk, planted = [], 0, None
    with open(bpk, "rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        big = [(n, d) for n, d in sorted(meta["tensors"].items()) if d["data_offsets"][1] - d["data_offsets"][0] >= 256]
        for i, (name, d) in enumerate(big):
            pat = bytes(mm[start + d["data_offsets"][0]:start + d["data_offsets"][0] + 256])
            if wasm.find(pat) != -1:
                found_module.append(name)
            if mm.find(pat) != -1:
                found_bpk += 1
            if i == 0:
                planted = (wasm + b"\0" * 7 + pat).find(pat) != -1
    res = {"tensors_searched": len(big), "found_in_module": found_module, "found_in_burnpack": found_bpk,
           "planted_window_found": bool(planted)}
    common.dump(out, res)
    print(json.dumps({k: (len(v) if isinstance(v, list) else v) for k, v in res.items()}))
    return res


def selftest(tool="wasm-tools"):
    import tempfile
    bad = 0

    def check(what, ok):
        nonlocal bad
        print(("ok   " if ok else "FAIL ") + what)
        bad += 0 if ok else 1

    wat = """(module
  (memory (export "memory") 1)
  (data (i32.const 0) "ab\\00\\00")
  (func $leaf (param i32) (result i32) local.get 0)
  (func $rec (export "forward") (param i32) (result i32)
    nop
    nop
    block
      local.get 0
      br_if 0
    end
    loop
      local.get 0
      br_if 0
    end
    local.get 0
    call $rec
    drop
    local.get 0
    call $leaf)
  (func $unnamed_by_export_only (param i32) (result i32) local.get 0 call $leaf))"""
    with tempfile.TemporaryDirectory() as d:
        open(d + "/m.wat", "w").write(wat)
        subprocess.run([tool, "parse", d + "/m.wat", "-o", d + "/m.wasm"], check=True)
        c, facts, names = census(d + "/m.wasm", tool)
        ic = c["instruction_census"]
        check("three functions, no import, one export", c["functions"] == 3 and c["imports"] == 0 and c["exports"] == ["forward", "memory"])
        check("two nops, one drop, one empty block, one empty loop", ic["nop"] == 2 and ic["drop"] == 1 and
              ic["block_empty"] == 1 and ic["loop_empty"] == 1 and ic["block_typed"] == 0)
        check("one self-calling function, one loop, a call cycle of one", c["self_calling_functions"] == 1 and
              c["functions_with"]["loop"] == 1 and c["functions_in_call_cycles"] == 1)
        check("the data segment holds two zero bytes of four", c["data_segments"] == 1 and c["data_segment_bytes"] == 4 and
              c["data_segment_zero_bytes"] == 2)
        check("names are read from the name section", names.get(1) == "rec" and c["functions_unnamed"] == 0)
        gr = graph("t", "wasm", facts, names)
        check("the graph has no blank node and the self-call is an edge", G.blank_terms(gr.text()) == 0 and any(
            "calls" in line and line.split()[0] == line.split()[2] for line in gr.lines()))
        # the search, on a tiny burnpack made by hand
        import cbor2
        import struct
        payload = bytes(range(256)) * 2
        meta = {"tensors": {"w": {"data_offsets": [0, 512], "dtype": "F32", "shape": [128], "param_id": 7}}, "metadata": {}}
        cb = cbor2.dumps(meta)
        head = struct.pack("<IHI", 0x42555254, 1, len(cb)) + cb
        head += b"\0" * (-len(head) % 256)
        open(d + "/m.bpk", "wb").write(head + payload)
        r = search(d + "/m.wasm", d + "/m.bpk", d + "/s.json")
        check("the search: not in the module, found in the burnpack, planted window found",
              r["found_in_module"] == [] and r["found_in_burnpack"] == 1 and r["planted_window_found"])
    check("crate heuristics: v0 and legacy", crates_of("_RNvMs3_NtCslO0ShxCE71i_9hashbrown3mapINtB6_7HashMapE3get") == ["hashbrown"]
          and crates_of("_ZN4core3fmt9Arguments6new_v117h0123456789abcdefE") == ["core"])
    return bad


if __name__ == "__main__":
    if sys.argv[1:2] == ["selftest"]:
        sys.exit(1 if selftest(*sys.argv[2:3]) else 0)
    if sys.argv[1:2] == ["census"] and len(sys.argv) in (6, 7):
        run(*sys.argv[2:])
    elif sys.argv[1:2] == ["search"] and len(sys.argv) == 5:
        search(*sys.argv[2:])
    else:
        sys.exit(__doc__)
