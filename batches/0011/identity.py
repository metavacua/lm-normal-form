# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
# The readers of the identity record of docs/interfaces/identity.md, version 1: the digest of every tensor
# in a safetensors file, an ONNX file or a burnpack file, and the comparison of two records. The digest,
# the roots and the verdicts are in identity_core.py, which this file calls and does not repeat.
# Usage: identity.py st|onnx|bpk FILE OUT.json [CENSUS.txt]   write the record, and the census of the file
#        identity.py match REF.json TARGET.json [TABLE.tsv]   which tensors of TARGET are in REF, and how
#        identity.py names REF.json TARGET.json               the pairing by content against the pairing by names
#        identity.py bpkdiff A.bpk B.bpk                      what differs between two burnpack files
#        identity.py selftest
import collections, hashlib, json, os, platform, re, struct, sys, tempfile
import numpy as np
import ml_dtypes
import identity_core as core

NP = {"float32": "f32", "float64": "f64", "int64": "i64", "int32": "i32", "int16": "i16", "int8": "i8",
      "uint64": "u64", "uint32": "u32", "uint16": "u16", "uint8": "u8", "bool": "bool", "float16": "f16", "bfloat16": "bf16"}
UNNP = {"f32": np.float32, "f64": np.float64, "i64": np.int64, "i32": np.int32, "i16": np.int16, "i8": np.int8,
        "u64": np.uint64, "u32": np.uint32, "u16": np.uint16, "u8": np.uint8, "bool": np.bool_, "f16": np.float16,
        "bf16": ml_dtypes.bfloat16}
ST = {"BF16": "bf16", "F16": "f16", "F32": "f32", "F64": "f64", "I64": "i64", "I32": "i32", "I16": "i16", "I8": "i8",
      "U8": "u8", "U16": "u16", "U32": "u32", "U64": "u64", "BOOL": "bool"}
BPK = {"F32": "f32", "F64": "f64", "F16": "f16", "BF16": "bf16", "I64": "i64", "I32": "i32", "I16": "i16", "I8": "i8",
       "U64": "u64", "U32": "u32", "U16": "u16", "U8": "u8"}


def sha256_file(path, start=0, end=None):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        f.seek(start)
        left = (os.path.getsize(path) if end is None else end) - start
        while left > 0:
            b = f.read(min(1 << 24, left))
            if not b:
                break
            h.update(b)
            left -= len(b)
    return h.hexdigest()


def digest(arr):
    """(canonical dtype, digest) of a numpy array: bf16 and f16 widened to f32, little-endian, C order."""
    stored = NP[arr.dtype.name]
    if stored in ("bf16", "f16"):
        arr, stored = arr.astype(np.float32), "f32"
    a = np.asarray(arr, dtype=arr.dtype.newbyteorder("<"))   # not ascontiguousarray: it makes a scalar a vector
    if not a.flags.c_contiguous:
        a = a.copy(order="C")
    return stored, core.tensor_digest(stored, a.shape, memoryview(a.reshape(-1).view(np.uint8)))


def entry(arr, labels, **extra):
    dt, d = digest(arr)
    e = {"digest": d, "digest_t": digest(arr.T)[1] if arr.ndim == 2 else None, "dtype": dt,
         "stored_dtype": NP[arr.dtype.name], "shape": list(arr.shape), "labels": labels}
    if arr.size <= 4096:   # the same values as a vector, to name a scalar stored as [1] "reshaped" and not "none"
        e["x-flat"] = digest(arr.reshape(-1))[1]
    e.update(extra)
    return e


def environment():
    cpu = ""
    try:
        cpu = next(line.split(":", 1)[1].strip() for line in open("/proc/cpuinfo") if line.startswith("model name"))
    except (OSError, StopIteration):
        pass
    return {"python": platform.python_version(), "numpy": np.__version__, "platform": platform.platform(), "cpu": cpu}


def record(container, entries, source, extra=None):
    here = hashlib.sha256(open(__file__, "rb").read() + open(core.__file__, "rb").read()).hexdigest()[:12]
    sub = {"container": container, "tensors": entries, "bag_root": core.bag_root(entries)}
    if container == "safetensors":
        sub["labelled_root"] = core.labelled_root(entries, container)
    sub.update(extra or {})
    rec = core.envelope(sub, {"name": "lmnf batches/0011/identity.py", "version": here}, environment())
    rec["source"] = source
    return rec


def write(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, sort_keys=True)
        f.write("\n")


# ---- safetensors ----------------------------------------------------------------------------------------------

def read_st(path):
    """The header by hand, as the format description gives it; the tensors as numpy views of the file."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        head = f.read(n)
    hdr = json.loads(head)
    meta = hdr.pop("__metadata__", None)
    mm = np.memmap(path, dtype=np.uint8, mode="r")
    ents, base, order = [], 8 + n, []
    for name in sorted(hdr):
        d = hdr[name]
        s, e = d["data_offsets"]
        dt = ST[d["dtype"]]
        arr = np.frombuffer(mm[base + s:base + e], dtype=UNNP[dt]).reshape(d["shape"])
        ents.append(entry(arr, {"safetensors": name}))
        order.append((s, e, name))
    order.sort()
    census = {"file_bytes": os.path.getsize(path), "header_bytes": n, "metadata": meta, "tensors": len(hdr),
              "dtypes": dict(collections.Counter(d["dtype"] for d in hdr.values())),
              "elements": sum(int(np.prod(d["shape"])) for d in hdr.values()),
              "has_lm_head": "lm_head.weight" in hdr,
              "keys_sorted_bytewise": list(hdr) == sorted(hdr, key=lambda k: k.encode()),
              "offsets_contiguous": all(order[i][1] == order[i + 1][0] for i in range(len(order) - 1)) and order[0][0] == 0,
              "file_sha256": sha256_file(path), "header_sha256": sha256_file(path, 0, base),
              "data_region_sha256": sha256_file(path, base)}
    return ents, census


def crosscheck_st(path, ents, count=6):
    """The same tensors through the safetensors library and torch: the first few, in name order."""
    import torch
    from safetensors import safe_open
    bad = 0
    with safe_open(path, framework="pt") as f:
        for e in ents[:count]:
            t = f.get_tensor(e["labels"]["safetensors"]).to(torch.float32).contiguous().numpy()
            if digest(t)[1] != e["digest"]:
                bad += 1
    return bad


# ---- ONNX -----------------------------------------------------------------------------------------------------

def read_onnx(path):
    import onnx
    from onnx import numpy_helper
    m = onnx.load(path)
    g = m.graph
    cons = collections.defaultdict(list)
    for nd in g.node:
        for i, nm in enumerate(nd.input):
            cons[nm].append([nd.op_type, i, nd.name])
    ents = []
    for t in g.initializer:
        a = numpy_helper.to_array(t)
        c = cons.get(t.name, [])
        role = ("matmul-weight" if any(o == "MatMul" and i == 1 for o, i, _ in c) else
                "gather-table" if any(o == "Gather" and i == 0 for o, i, _ in c) else
                "vector" if a.ndim == 1 else "other")
        ents.append(entry(a, {"onnx": t.name}, **{"x-consumers": c[:3], "x-role": role}))
    ents.sort(key=lambda e: e["labels"]["onnx"])
    consts, skipped = [], 0
    for nd in g.node:
        if nd.op_type != "Constant":
            continue
        a = None
        for at in nd.attribute:
            if at.name == "value":
                a = numpy_helper.to_array(at.t)
            elif at.name in ("value_float", "value_int"):
                a = np.array(at.f if at.name == "value_float" else at.i, dtype=np.float32 if at.name == "value_float" else np.int64)
            elif at.name in ("value_floats", "value_ints"):
                a = np.array(list(at.floats if at.name == "value_floats" else at.ints),
                             dtype=np.float32 if at.name == "value_floats" else np.int64)
        if a is None:
            skipped += 1
        else:
            consts.append(entry(a, {"onnx": nd.output[0]}))
    census = {"file_bytes": os.path.getsize(path), "file_sha256": sha256_file(path), "ir_version": m.ir_version,
              "opset_import": [[o.domain, o.version] for o in m.opset_import], "producer_name": m.producer_name,
              "producer_version": m.producer_version, "domain": m.domain, "model_version": m.model_version,
              "doc_string": m.doc_string, "metadata_props": [[p.key, p.value] for p in m.metadata_props],
              "graph_name": g.name, "nodes": len(g.node), "initializers": len(g.initializer),
              "initializer_elements": sum(int(np.prod(e["shape"])) for e in ents),
              "roles": dict(collections.Counter(e["x-role"] for e in ents)),
              "constant_nodes": sum(1 for nd in g.node if nd.op_type == "Constant"), "constants_skipped": skipped,
              "operators": dict(collections.Counter(nd.op_type for nd in g.node))}
    return ents, consts, census


# ---- burnpack -------------------------------------------------------------------------------------------------

def read_bpk(path):
    """The header, the CBOR metadata and the tensors, by the layout of the burn-pack crate (see the batch document)."""
    import cbor2
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        head = f.read(10)
        magic, version, msize = struct.unpack("<IHI", head)
        cbor = f.read(msize)
    meta = cbor2.loads(cbor)
    data_start = -(-(10 + msize) // 256) * 256
    mm = np.memmap(path, dtype=np.uint8, mode="r")
    ents, last_end, pad_ok = [], 0, bool(not mm[10 + msize:data_start].any())
    keys = collections.Counter()
    for name in sorted(meta["tensors"]):
        d = meta["tensors"][name]
        keys[tuple(sorted(d))] += 1
        s, e = d["data_offsets"]
        last_end = max(last_end, e)
        dt = BPK.get(d["dtype"]) if isinstance(d["dtype"], str) else None
        if dt is None:
            ents.append({"digest": None, "labels": {"burnpack": name}, "shape": d["shape"], "dtype": str(d["dtype"])})
            continue
        arr = np.frombuffer(mm[data_start + s:data_start + e], dtype=UNNP[dt]).reshape(d["shape"])
        ents.append(entry(arr, {"burnpack": name}, **{"x-bytes": e - s, "x-param-id": d.get("param_id")}))
    ents_ok = [e for e in ents if e["digest"]]
    names = [e["labels"]["burnpack"] for e in ents]
    pat = re.compile(r"^submodule\d+\.(constant\d+|linear\d+\.weight)$")
    census = {"file_bytes": size, "file_sha256": sha256_file(path), "head_hex": head.hex(), "magic_ascii": head[:4].decode("latin1"),
              "version": version, "metadata_size": msize, "data_start": data_start, "padding_after_cbor_zero": pad_ok,
              "cbor_head_hex": cbor[:12].hex(), "cbor_tail_hex": cbor[-29:].hex(), "last_tensor_end": last_end,
              "file_bytes_equal_data_start_plus_last_end": size == data_start + last_end,
              "top_level_keys": sorted(meta), "metadata": meta.get("metadata"), "has_scalars": "scalars" in meta,
              "tensors": len(ents), "dtypes": dict(collections.Counter(str(d["dtype"]) for d in meta["tensors"].values())),
              "descriptor_keys": {",".join(k): v for k, v in keys.items()},
              "param_ids": sum(1 for d in meta["tensors"].values() if d.get("param_id") is not None),
              "param_id_min": min((d["param_id"] for d in meta["tensors"].values() if d.get("param_id") is not None), default=None),
              "names_matching_pattern": sum(1 for n in names if pat.match(n)),
              "weights": sum(1 for n in names if n.endswith(".weight")), "biases": sum(1 for n in names if n.endswith(".bias")),
              "tensors_256_or_more": sum(1 for e in ents_ok if e["x-bytes"] >= 256),
              "elements_in_those": sum(int(np.prod(e["shape"])) for e in ents_ok if e["x-bytes"] >= 256),
              "first_tensor": min(meta["tensors"].items(), key=lambda kv: kv[1]["data_offsets"][0])[0],
              "scalar_f32_1e-5": sum(1 for e in ents_ok if e["dtype"] == "f32" and e["shape"] == [1] and e["digest"] == digest(np.array([np.float32(1e-5)]))[1])}
    return ents_ok, census, meta


def bpkdiff(a, b):
    ca, cb = read_bpk(a), read_bpk(b)
    ma, mb = ca[2], cb[2]
    da, db = ca[1]["data_start"], cb[1]["data_start"]
    out = {"sizes_equal": ca[1]["file_bytes"] == cb[1]["file_bytes"], "sha256_equal": ca[1]["file_sha256"] == cb[1]["file_sha256"],
           "data_start": [da, db]}
    with open(a, "rb") as fa, open(b, "rb") as fb:
        xa, xb = np.frombuffer(fa.read(da), np.uint8), np.frombuffer(fb.read(db), np.uint8)
    n = min(len(xa), len(xb))
    out["differing_bytes_before_data"] = int((xa[:n] != xb[:n]).sum())
    out["differing_bytes_span"] = [int(i) for i in np.flatnonzero(xa[:n] != xb[:n])[[0, -1]]] if out["differing_bytes_before_data"] else None
    out["data_region_sha256_equal"] = sha256_file(a, da) == sha256_file(b, db)

    def strip(m):
        return {k: {kk: vv for kk, vv in v.items() if kk != "param_id"} for k, v in m["tensors"].items()}
    out["metadata_equal_without_param_id"] = strip(ma) == strip(mb) and ma.get("metadata") == mb.get("metadata")
    out["param_ids_differing"] = sum(1 for k in ma["tensors"] if ma["tensors"][k].get("param_id") != mb["tensors"].get(k, {}).get("param_id"))
    out["bag_roots_equal"] = core.bag_root(ca[0]) == core.bag_root(cb[0])
    return out


# ---- comparison -----------------------------------------------------------------------------------------------

def match(ref, tgt, constants=False):
    by, byt = collections.defaultdict(list), collections.defaultdict(list)
    pool = ref["subject"]["tensors"] + (ref["subject"].get("constants", []) if constants else [])
    for e in pool:
        by[e["digest"]].append(e)
        if e.get("digest_t"):
            byt[e["digest_t"]].append(e)
    flat = collections.defaultdict(list)
    for e in pool:
        if e.get("x-flat"):
            flat[e["x-flat"]].append(e)
    rows = []
    for e in tgt["subject"]["tensors"]:
        same, tr = by.get(e["digest"], []), byt.get(e["digest"], [])
        rel = "none" if not same and not tr else "identical" if same and not tr else "transposed" if tr and not same else "both"
        if rel == "none" and flat.get(e.get("x-flat")):
            rel, same = "reshaped", flat[e["x-flat"]]
        refs = [list(x["labels"].values())[0] for x in same + tr]
        rows.append({"label": list(e["labels"].values())[0], "shape": e["shape"], "role": e.get("x-role", "-"), "relation": rel,
                     "ref": refs, "key": core.key(e)})
    kr = {core.key(e) for e in pool}
    kt = {core.key(e) for e in tgt["subject"]["tensors"]}
    used = collections.Counter(r for row in rows for r in row["ref"])
    summary = {"target_tensors": len(rows), "target_distinct_keys": len(kt), "ref_tensors": len(pool),
               "ref_distinct_keys": len(kr), "keys_in_both": len(kr & kt), "keys_only_target": len(kt - kr), "keys_only_ref": len(kr - kt),
               "relations": dict(collections.Counter(r["relation"] for r in rows)),
               "relations_by_role": {f"{k[0]}/{k[1]}": v for k, v in collections.Counter((r["role"], r["relation"]) for r in rows).items()},
               "ref_tensors_matched_by_more_than_one": {k: v for k, v in used.items() if v > 1},
               "bag_roots_equal": ref["subject"]["bag_root"] == tgt["subject"]["bag_root"]}
    return rows, summary


HF_ALIAS = {"lm_head.weight": "model.embed_tokens.weight"}


def hf_name(e):
    """The Hugging Face name an ONNX initializer would have by its own name or its consumer's node name."""
    n = e["labels"]["onnx"]
    if n.startswith("model.") or n.startswith("lm_head"):
        return HF_ALIAS.get(n, n)
    for op, i, node in e.get("x-consumers", []):
        if op == "MatMul" and i == 1 and node:
            nm = node.strip("/")
            nm = nm[:-len("/MatMul")] if nm.endswith("/MatMul") else nm
            return HF_ALIAS.get(nm.replace("/", ".") + ".weight", nm.replace("/", ".") + ".weight")
    return None


def names(ref, tgt):
    rows, _ = match(ref, tgt)
    agree = disagree = unnamed = 0
    bad = []
    for e, r in zip(tgt["subject"]["tensors"], rows):
        want = hf_name(e)
        if want is None:
            unnamed += 1
        elif want in r["ref"]:
            agree += 1
        else:
            disagree += 1
            bad.append([r["label"], want, r["ref"]])
    return {"agree": agree, "disagree": disagree, "unnamed": unnamed, "disagreements": bad[:20]}


# ---- command line ---------------------------------------------------------------------------------------------

def selftest():
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    import cbor2
    bad = 0

    def check(what, ok):
        nonlocal bad
        print(("ok     " if ok else "FAILED ") + what)
        bad += 0 if ok else 1
    rng = np.random.default_rng(0)
    for name, dt in (("bf16", ml_dtypes.bfloat16), ("f16", np.float16)):
        a = rng.standard_normal((4, 6)).astype(dt)
        check(f"{name} widened by numpy and by the core's own widening give one digest",
              digest(a)[1] == core.tensor_digest(name, a.shape, a.tobytes()))
    a = rng.standard_normal((5, 3)).astype(np.float32)
    b = a.copy()
    b.view(np.uint32)[2, 1] ^= 1
    check("one flipped bit in one element changes the digest", digest(a)[1] != digest(b)[1])
    e, et = entry(a, {"x": "a"}), entry(a.T.copy(), {"x": "b"})
    check("the digest of a transpose is the transposed digest", e["digest_t"] == et["digest"] and e["digest"] == et["digest_t"])
    check("the key of a tensor and of its transpose are one", core.key(e) == core.key(et))
    s0 = np.array(1.0, dtype=np.float32)
    check("a scalar digests as rank 0 (as the core does), and differently from the same value as a vector",
          digest(s0)[1] == core.tensor_digest("f32", (), s0.tobytes()) and digest(s0)[1] != digest(s0.reshape(1))[1])
    check("a matrix that is not contiguous digests as the same matrix that is",
          digest(a.T)[1] == digest(np.ascontiguousarray(a.T))[1] == core.tensor_digest("f32", (3, 5), np.ascontiguousarray(a.T).tobytes()))
    check("the same bytes as i32 and as f32 digest differently",
          digest(np.array([1.0], np.float32))[1] != digest(np.array([1.0], np.float32).view(np.int32))[1])
    with tempfile.TemporaryDirectory() as d:
        w = rng.standard_normal((3, 2)).astype(np.float32)
        g = helper.make_graph([helper.make_node("MatMul", ["x", "w"], ["y"], name="/m/MatMul")], "g",
                              [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3])],
                              [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2])], [numpy_helper.from_array(w, "w")])
        p = f"{d}/m.onnx"
        onnx.save(helper.make_model(g, opset_imports=[helper.make_opsetid("", 18)]), p)
        ents, consts, census = read_onnx(p)
        check("an ONNX initializer is read with its digest and its role", ents[0]["digest"] == digest(w)[1] and ents[0]["x-role"] == "matmul-weight")
        # a burnpack by the documented layout: the header, the CBOR, zeros to 256, tensors each on a 256-byte boundary
        t1, t2 = rng.standard_normal((3, 2)).astype(np.float32), np.array([-1], np.int64)
        desc = {"submodule1.linear1.weight": {"dtype": "F32", "shape": [3, 2], "data_offsets": [0, 24], "param_id": 2 ** 40 + 1},
                "submodule1.constant1": {"dtype": "I64", "shape": [1], "data_offsets": [256, 264], "param_id": 2 ** 40 + 2}}
        cbor = cbor2.dumps({"tensors": desc, "metadata": {"producer": "burn-onnx"}})
        ds = -(-(10 + len(cbor)) // 256) * 256
        blob = struct.pack("<IHI", 0x4255524E, 1, len(cbor)) + cbor
        blob += bytes(ds - len(blob)) + t1.tobytes() + bytes(256 - 24) + t2.tobytes()
        open(f"{d}/m.bpk", "wb").write(blob)
        bents, bc, _ = read_bpk(f"{d}/m.bpk")
        check("a burnpack is read: header, names, digests, padding",
              bc["head_hex"][:8] == "4e525542" and bc["padding_after_cbor_zero"] and bc["file_bytes_equal_data_start_plus_last_end"]
              and bents[1]["digest"] == digest(t1)[1] and bents[0]["digest"] == digest(t2)[1])
        rows, s = match({"subject": {"tensors": ents, "bag_root": "x"}}, {"subject": {"tensors": [entry(w.T.copy(), {"b": "w"})], "bag_root": "y"}})
        check("a transposed tensor is found, and said to be transposed", rows[0]["relation"] == "transposed")
        try:
            from safetensors.numpy import save_file
            save_file({"a": a, "b": np.array([3], np.int64)}, f"{d}/m.safetensors", metadata={"format": "np"})
            sents, sc = read_st(f"{d}/m.safetensors")
            check("a safetensors file is read: names, digests, census",
                  [x["labels"]["safetensors"] for x in sents] == ["a", "b"] and sents[0]["digest"] == digest(a)[1] and sc["offsets_contiguous"])
        except ImportError:
            print("skipped the safetensors read (the library is not installed)")
    print("selftest:", "all ok" if not bad else f"{bad} failed")
    return not bad


def main(argv):
    if len(argv) >= 2 and argv[1] == "selftest":
        return 0 if selftest() else 1
    if len(argv) >= 4 and argv[1] in ("st", "onnx", "bpk"):
        kind, path, out = argv[1], argv[2], argv[3]
        extra, census = {}, None
        if kind == "st":
            ents, census = read_st(path)
        elif kind == "onnx":
            ents, consts, census = read_onnx(path)
            extra = {"constants": consts}
        else:
            ents, census, _ = read_bpk(path)
        rec = record({"st": "safetensors", "onnx": "onnx", "bpk": "burnpack"}[kind], ents, os.path.basename(path), extra)
        rec["census"] = census
        write(out, rec)
        print(json.dumps(census, indent=1, sort_keys=True))
        print("verdict:", core.verify(rec), " tensors:", len(ents), " bag_root:", rec["subject"]["bag_root"])
        if len(argv) >= 5:
            write(argv[4], census)
        if kind == "st" and "--crosscheck" in argv:
            print("tensors read differently by the safetensors library:", crosscheck_st(path, ents))
        return 0
    if len(argv) >= 4 and argv[1] == "match":
        ref, tgt = json.load(open(argv[2])), json.load(open(argv[3]))
        rows, summary = match(ref, tgt, "--constants" in argv)
        print(json.dumps(summary, indent=1, sort_keys=True))
        if len(argv) >= 5 and argv[4] != "--constants":
            with open(argv[4], "w", encoding="utf-8") as f:
                f.write("label\tshape\trole\trelation\tref\n")
                for r in rows:
                    f.write(f"{r['label']}\t{r['shape']}\t{r['role']}\t{r['relation']}\t{'|'.join(r['ref'])}\n")
        return 0
    if len(argv) >= 4 and argv[1] == "names":
        print(json.dumps(names(json.load(open(argv[2])), json.load(open(argv[3]))), indent=1))
        return 0
    if len(argv) >= 4 and argv[1] == "bpkdiff":
        print(json.dumps(bpkdiff(argv[2], argv[3]), indent=1, sort_keys=True))
        return 0
    sys.exit(__doc__)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
