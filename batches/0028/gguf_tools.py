# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
"""GGUF for batch 0028: two independent readers, the identity record of each file, the relations between a
GGUF file's tensors and a safetensors file's, decided by content, the key-value table against config.json, and
the census of blank and empty constructs.

The relations are searched by content. Every tensor of the safetensors file is entered in an index under the
digest of itself, of its transpose and, for a matrix whose rows divide, of llama.cpp's permutation of its rows
for the query head count and for the key-value head count (conversion/llama.py of llama.cpp b11450:
reshape(n, 2, rows // n // 2, ...).swapaxes(1, 2).reshape(shape)); a GGUF tensor is looked up by its own digest.
The pairing by names (gguf-py's own tensor name map) is made apart and compared.

Usage: gguf_tools.py analyze TAG GGUF_F32 GGUF_BF16 SAFETENSORS CONFIG.json OUTDIR
       gguf_tools.py selftest
"""

import collections
import hashlib
import json
import os
import struct
import sys

import numpy as np

import common
import ground as G
from common import identity

GGML_F32, GGML_F16, GGML_BF16 = 0, 1, 30


def perm(arr, n):
    """llama.cpp's permutation of the rows of a query or key matrix into heads of interleaved pairs."""
    return arr.reshape(n, 2, arr.shape[0] // n // 2, *arr.shape[1:]).swapaxes(1, 2).reshape(arr.shape)


def type_name(t):
    import gguf
    try:
        return gguf.GGMLQuantizationType(int(t)).name
    except ValueError:
        return f"TYPE{int(t)}"


def gguf_array(t):
    """A reader tensor as a numpy array in the order Hugging Face uses (ggml lists the fastest dimension first)."""
    import ml_dtypes
    dims = tuple(reversed([int(x) for x in t.shape]))
    tt = int(t.tensor_type)
    d = np.asarray(t.data)
    if tt == GGML_F32:
        return d.reshape(dims)
    if tt == GGML_F16:
        return d.reshape(dims)
    if tt == GGML_BF16:
        return d.reshape(-1).view(np.uint16).view(ml_dtypes.bfloat16).reshape(dims)
    return None


# ---- the second reader: struct alone ------------------------------------------------------------------------------

def parse_stdlib(path):
    """Header, key-value pairs and tensor table of a GGUF file, from the format description alone."""
    sc = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
    with open(path, "rb") as f:
        def rd(fmt):
            return struct.unpack(fmt, f.read(struct.calcsize(fmt)))

        def rstr():
            (n,) = rd("<Q")
            return f.read(n)

        def rval(t):
            if t in sc:
                return rd(sc[t])[0]
            if t == 8:
                return rstr()
            if t == 9:
                (et,) = rd("<I")
                (n,) = rd("<Q")
                return [rval(et) for _ in range(n)]
            raise ValueError(f"value type {t}")

        if f.read(4) != b"GGUF":
            raise ValueError("not a GGUF file")
        (version,) = rd("<I")
        n_tensors, n_kv = rd("<QQ")
        kv = {}
        for _ in range(n_kv):
            k = rstr().decode("utf-8")
            (t,) = rd("<I")
            kv[k] = rval(t)
        infos = []
        for _ in range(n_tensors):
            name = rstr().decode("utf-8")
            (nd,) = rd("<I")
            dims = list(rd(f"<{nd}Q"))
            (typ,) = rd("<I")
            (off,) = rd("<Q")
            infos.append({"name": name, "dims": dims, "type": typ, "offset": off})
        end = f.tell()
    align = int(kv.get("general.alignment", 32))
    start = -(-end // align) * align
    for i in infos:
        i["abs"] = start + i["offset"]
    return {"version": version, "n_tensors": n_tensors, "n_kv": n_kv, "kv": kv, "tensors": infos,
            "alignment": align, "infos_end": end, "data_start": start, "file_bytes": os.path.getsize(path)}


# ---- the first reader: gguf-py ------------------------------------------------------------------------------------

def open_gguf(path):
    import gguf
    return gguf.GGUFReader(path)


def field_value(field):
    return field.contents()


def kv_table(reader):
    """Every key: its type, and its value if a scalar; for an array its length and the sha256 of its contents."""
    out = {}
    for name, field in reader.fields.items():
        if name.startswith("GGUF."):    # the three header fields that gguf-py lists beside the pairs
            continue
        types = [t.name for t in field.types]
        v = field_value(field)
        e = {"type": "<".join(types)}
        if isinstance(v, (list, tuple, np.ndarray)):
            seq = [x.item() if hasattr(x, "item") else x for x in (v.tolist() if isinstance(v, np.ndarray) else v)]
            e["len"] = len(seq)
            e["sha256"] = hashlib.sha256(json.dumps(seq, ensure_ascii=True).encode()).hexdigest()
        else:
            v = v.item() if hasattr(v, "item") else v
            if isinstance(v, bytes):
                v = v.decode("utf-8", "replace")
            e["value"] = v
            if isinstance(v, str) and len(v) > 200:
                e["len"] = len(v)
                e["sha256"] = hashlib.sha256(v.encode()).hexdigest()
                e["value"] = v[:200]
        out[name] = e
    return out


def crosscheck(reader, std):
    """C5: the two readers on every tensor name, dimension list, type and absolute offset, and on the key set."""
    bad = []
    a = {t.name: (list(map(int, t.shape)), int(t.tensor_type), int(t.data_offset)) for t in reader.tensors}
    b = {i["name"]: (i["dims"], i["type"], i["abs"]) for i in std["tensors"]}
    if len(reader.tensors) != len(std["tensors"]):
        bad.append(["tensor count", len(reader.tensors), len(std["tensors"])])
    for k in sorted(set(a) | set(b)):
        if a.get(k) != b.get(k):
            bad.append([k, a.get(k), b.get(k)])
    keys = {k for k in reader.fields if not k.startswith("GGUF.")}
    if keys != set(std["kv"]):
        bad.append(["key set", sorted(keys ^ set(std["kv"]))[:10]])
    for k in sorted(keys & set(std["kv"])):
        if isinstance(std["kv"][k], list):
            n = len(field_value(reader.fields[k]))
            if n != len(std["kv"][k]):
                bad.append([k, "array length", n, len(std["kv"][k])])
    return {"tensors_compared": len(b), "keys_compared": len(keys), "mismatches": bad[:20], "mismatch_count": len(bad)}


# ---- records, census ----------------------------------------------------------------------------------------------

def record(path, reader):
    ents, skipped = [], []
    for t in sorted(reader.tensors, key=lambda t: t.name):
        a = gguf_array(t)
        if a is None:
            skipped.append([t.name, type_name(t.tensor_type)])
            continue
        ents.append(identity.entry(a, {"gguf": t.name}, **{"x-gguf-type": type_name(t.tensor_type)}))
    return identity.record("gguf", ents, os.path.basename(path)), skipped


def census(path, reader, std):
    infos = sorted(std["tensors"], key=lambda i: i["abs"])
    sizes = {t.name: int(t.n_bytes) for t in reader.tensors}
    gaps = sum(max(0, b["abs"] - (a["abs"] + sizes[a["name"]])) for a, b in zip(infos, infos[1:]))
    end = max((i["abs"] + sizes[i["name"]] for i in infos), default=std["data_start"])
    kv = kv_table(reader)
    return {
        "file_bytes": std["file_bytes"], "file_sha256": identity.sha256_file(path), "version": std["version"],
        "alignment": std["alignment"], "tensors": len(infos), "kv": len(std["kv"]),
        "dtypes": dict(collections.Counter(type_name(t.tensor_type) for t in reader.tensors)),
        "padding_before_data_bytes": std["data_start"] - std["infos_end"],
        "padding_between_tensors_bytes": gaps, "trailing_bytes": std["file_bytes"] - end,
        "kv_empty_strings": sum(1 for v in kv.values() if v.get("value") == "" and "STRING" in v["type"]),
        "kv_empty_arrays": sum(1 for v in kv.values() if v.get("len") == 0),
        "kv_arrays": sum(1 for v in kv.values() if "ARRAY" in v["type"]),
        "tensors_one_dimensional": sum(1 for i in infos if len(i["dims"]) <= 1),
        "has_output_weight": any(i["name"] == "output.weight" for i in infos),
        "has_rope_freqs": any(i["name"] == "rope_freqs.weight" for i in infos),
    }, kv


# ---- the relations to the safetensors file ------------------------------------------------------------------------

def st_index(arrays, n_head, n_kv):
    """digest -> [(name, relation, n)]: the tensor itself, its transpose, llama.cpp's permutations of its rows."""
    idx = collections.defaultdict(list)
    for name, a in arrays.items():
        idx[identity.digest(a)[1]].append((name, "identical", None))
        if a.ndim == 2:
            idx[identity.digest(a.T)[1]].append((name, "transposed", None))
            for n in sorted({n_head, n_kv}):
                if a.shape[0] % (2 * n) == 0:
                    idx[identity.digest(perm(a, n))[1]].append((name, "permuted", n))
    return idx


PRIORITY = {"identical": 0, "permuted": 1, "transposed": 2}


def relate(arr, idx):
    hits = sorted(idx.get(identity.digest(arr)[1], []), key=lambda h: PRIORITY[h[1]])
    if not hits:
        return "none", [], None
    return hits[0][1], [h[0] for h in hits if h[1] == hits[0][1]], hits[0][2]


def name_map(hf_names, n_blocks):
    """gguf-py's own map from Hugging Face tensor names to GGUF names, as the converter uses it."""
    import gguf
    tm = gguf.get_tensor_name_map(gguf.MODEL_ARCH.LLAMA, n_blocks)
    return {n: tm.get_name(n, try_suffixes=(".weight", ".bias")) for n in hf_names}


def maps(reader, arrays, n_head, n_kv, n_blocks):
    idx = st_index(arrays, n_head, n_kv)
    to_gguf = name_map(list(arrays), n_blocks)
    to_hf = {v: k for k, v in to_gguf.items() if v}
    rows = []
    for t in sorted(reader.tensors, key=lambda t: t.name):
        a = gguf_array(t)
        rel, refs, n = relate(a, idx) if a is not None else ("unread", [], None)
        by_name = to_hf.get(t.name)
        rows.append({"gguf": t.name, "relation": rel, "refs": refs, "n": n, "by_name": by_name,
                     "agree": by_name in refs if refs else False, "shape": list(a.shape) if a is not None else None,
                     "type": type_name(t.tensor_type)})
    return rows, idx


def controls(reader, arrays, idx, n_head, n_kv):
    """C1: one unit in the last place moved in one tensor is `none`; C2: the wrong head count is `none`, the right one not."""
    out = {}
    norm = next(t for t in reader.tensors if len(t.shape) == 1)
    a = gguf_array(norm).copy()
    flat = a.reshape(-1).view(np.uint16 if a.dtype.itemsize == 2 else np.uint32)
    flat[-1] += 1
    out["C1_moved_tensor_is_none"] = relate(a, idx)[0] == "none"
    out["C1_unmoved_tensor_is_found"] = relate(gguf_array(norm), idx)[0] != "none"
    q = arrays["model.layers.0.self_attn.q_proj.weight"]
    gq = next(t for t in reader.tensors if t.name == "blk.0.attn_q.weight")
    want = identity.digest(gguf_array(gq))[1]
    wrong = next(n for n in (n_head - 1, n_head + 1, 2 * n_head, n_head // 2, 2, 4, 8, 16)
                 if n >= 1 and n != n_head and q.shape[0] % (2 * n) == 0)
    out["C2_wrong_head_count_is_none"] = identity.digest(perm(q, wrong))[1] != want
    out["C2_right_head_count_is_found"] = identity.digest(perm(q, n_head))[1] == want
    return out


# ---- the key-value table against config.json ----------------------------------------------------------------------

def f32(x):
    return float(np.float32(x))


def config_checks(kv, cfg):
    def val(k):
        return kv.get(k, {}).get("value")

    heads = cfg["num_attention_heads"]
    want = [
        ("general.architecture", "llama", False),
        ("llama.block_count", cfg["num_hidden_layers"], False),
        ("llama.context_length", cfg["max_position_embeddings"], False),
        ("llama.embedding_length", cfg["hidden_size"], False),
        ("llama.feed_forward_length", cfg["intermediate_size"], False),
        ("llama.attention.head_count", heads, False),
        ("llama.attention.head_count_kv", cfg["num_key_value_heads"], False),
        ("llama.rope.freq_base", f32(cfg["rope_theta"]), True),
        ("llama.attention.layer_norm_rms_epsilon", f32(cfg["rms_norm_eps"]), True),
        ("llama.vocab_size", cfg["vocab_size"], False),
        ("llama.rope.dimension_count", cfg["hidden_size"] // heads, False),
    ]
    rows = []
    for k, w, is_float in want:
        got = val(k)
        ok = got is not None and (f32(got) == w if is_float else got == w)
        rows.append({"key": k, "want": w, "got": got, "ok": ok})
    ntok = kv.get("tokenizer.ggml.tokens", {}).get("len")
    rows.append({"key": "len(tokenizer.ggml.tokens)", "want": cfg["vocab_size"], "got": ntok, "ok": ntok == cfg["vocab_size"]})
    return rows


# ---- the graph of the key-value pairs -----------------------------------------------------------------------------

def kv_graph(tag, artifact, kv):
    gr = G.Graph(tag, artifact + "-kv")
    art = G.iri("a", tag, artifact)
    gr.typed(art, "Artifact")
    gr.add(art, G.pred("format"), G.lit("gguf"))
    for k in sorted(kv):
        e = kv[k]
        n = G.iri("k", tag, artifact, k)
        gr.typed(n, "KeyValue")
        gr.add(n, G.pred("inArtifact"), art)
        gr.add(n, G.pred("label"), G.lit(k))
        gr.add(n, G.pred("valueType"), G.lit(e["type"]))
        if "value" in e:
            gr.add(n, G.pred("value"), G.lit(e["value"]))
        if "len" in e:
            gr.add(n, G.pred("length"), G.num(e["len"]))
        if "sha256" in e:
            gr.add(n, G.pred("sha256"), G.lit(e["sha256"]))
    return gr


# ---- the command --------------------------------------------------------------------------------------------------

def analyze(tag, f32_path, bf16_path, st_path, cfg_path, outdir):
    os.makedirs(outdir, exist_ok=True)
    cfg = common.load(cfg_path)
    arrays, st_info = common.st_arrays(st_path)
    n_head, n_kv, n_blocks = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["num_hidden_layers"]
    result = {"tag": tag, "safetensors": st_info, "safetensors_sha256": identity.sha256_file(st_path)}
    recs = {}
    for label, path in (("f32", f32_path), ("bf16", bf16_path)):
        reader, std = open_gguf(path), parse_stdlib(path)
        rec, skipped = record(path, reader)
        c, kv = census(path, reader, std)
        c["tensors_not_read"] = skipped
        recs[label] = rec
        common.dump(os.path.join(outdir, f"gguf-{label}-record.json"), rec)
        common.dump(os.path.join(outdir, f"gguf-{label}-census.json"), c)
        common.dump(os.path.join(outdir, f"gguf-{label}-independent.json"), crosscheck(reader, std))
        if label == "f32":
            common.dump(os.path.join(outdir, "gguf-kv.json"), kv)
            common.dump(os.path.join(outdir, "gguf-config.json"), config_checks(kv, cfg))
            G.write(os.path.join(outdir, "gguf-f32-kv.nq.gz"), kv_graph(tag, "gguf-f32", kv).text())
            rows, idx = maps(reader, arrays, n_head, n_kv, n_blocks)
            summary = {
                "relations": dict(collections.Counter(r["relation"] for r in rows)),
                "permuted_by_n": dict(collections.Counter(str(r["n"]) for r in rows if r["relation"] == "permuted")),
                "permuted_roles": dict(collections.Counter(
                    (r["by_name"] or "").rsplit(".", 2)[-2] for r in rows if r["relation"] == "permuted")),
                "name_pairing_agrees": sum(1 for r in rows if r["agree"]), "name_pairing_disagrees": [
                    [r["gguf"], r["by_name"], r["refs"]] for r in rows if not r["agree"]][:20],
                "tensors": len(rows), "st_tensors": len(arrays)}
            common.dump(os.path.join(outdir, "gguf-maps.json"), {"summary": summary, "rows": rows})
            common.dump(os.path.join(outdir, "gguf-controls.json"), controls(reader, arrays, idx, n_head, n_kv))
            result["f32_summary"] = summary
    rows, summ = identity.match(recs["f32"], recs["bf16"])
    summ["rows_not_identical"] = [r["label"] for r in rows if r["relation"] != "identical"][:20]
    common.dump(os.path.join(outdir, "gguf-bf16-vs-f32.json"), summ)
    result["bf16_vs_f32"] = {k: summ[k] for k in ("bag_roots_equal", "keys_in_both", "relations")}
    common.dump(os.path.join(outdir, "gguf-summary.json"), result)
    print(json.dumps(result, sort_keys=True)[:2000])


def synthetic_model(seed, hid=64, heads=4, kvh=2, ff=128, vocab=20, layers=2):
    """A small model with the tensor names of a Hugging Face Llama: BF16 weights, and its config. For the tests of the batch."""
    import ml_dtypes
    rng = np.random.default_rng(seed)
    hd = hid // heads

    def bf(shape):
        return rng.standard_normal(shape).astype(ml_dtypes.bfloat16)

    st = {"model.embed_tokens.weight": bf((vocab, hid)), "model.norm.weight": bf((hid,))}
    for i in range(layers):
        p = f"model.layers.{i}."
        st.update({p + "input_layernorm.weight": bf((hid,)), p + "post_attention_layernorm.weight": bf((hid,)),
                   p + "self_attn.q_proj.weight": bf((hid, hid)), p + "self_attn.k_proj.weight": bf((kvh * hd, hid)),
                   p + "self_attn.v_proj.weight": bf((kvh * hd, hid)), p + "self_attn.o_proj.weight": bf((hid, hid)),
                   p + "mlp.gate_proj.weight": bf((ff, hid)), p + "mlp.up_proj.weight": bf((ff, hid)),
                   p + "mlp.down_proj.weight": bf((hid, ff))})
    cfg = {"num_hidden_layers": layers, "max_position_embeddings": 64, "hidden_size": hid, "intermediate_size": ff,
           "num_attention_heads": heads, "num_key_value_heads": kvh, "rope_theta": 100000.0, "rms_norm_eps": 1e-5,
           "vocab_size": vocab}
    return st, cfg


def write_safetensors(path, st):
    hdr, data, off = {"__metadata__": {"format": "pt"}}, b"", 0
    for k in sorted(st):
        raw = st[k].tobytes()
        hdr[k] = {"dtype": "BF16", "shape": list(st[k].shape), "data_offsets": [off, off + len(raw)]}
        data, off = data + raw, off + len(raw)
    h = json.dumps(hdr, separators=(",", ":")).encode()
    h += b" " * (-len(h) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)) + h + data)


def write_gguf(path, st, cfg, outtype, permute=True, extra_kv=None):
    """A GGUF file as the converter would write it: the query and key weights permuted, norms and 1-D tensors in F32."""
    import gguf
    layers, heads, kvh = cfg["num_hidden_layers"], cfg["num_attention_heads"], cfg["num_key_value_heads"]
    tmap = gguf.get_tensor_name_map(gguf.MODEL_ARCH.LLAMA, layers)
    w = gguf.GGUFWriter(path, "llama")
    w.add_block_count(layers)
    w.add_context_length(cfg["max_position_embeddings"])
    w.add_embedding_length(cfg["hidden_size"])
    w.add_feed_forward_length(cfg["intermediate_size"])
    w.add_head_count(heads)
    w.add_head_count_kv(kvh)
    w.add_rope_freq_base(cfg["rope_theta"])
    w.add_layer_norm_rms_eps(cfg["rms_norm_eps"])
    w.add_vocab_size(cfg["vocab_size"])
    w.add_rope_dimension_count(cfg["hidden_size"] // heads)
    w.add_token_list([f"t{i}" for i in range(cfg["vocab_size"])])
    for k, v in (extra_kv or {}).items():
        w.add_string(k, v)
    for name, a in st.items():
        g = tmap.get_name(name, try_suffixes=(".weight", ".bias"))
        x = a.astype(np.float32)
        if permute and name.endswith("q_proj.weight"):
            x = perm(x, heads)
        if permute and name.endswith("k_proj.weight"):
            x = perm(x, kvh)
        if outtype == "bf16" and x.ndim > 1 and not g.endswith("_norm.weight"):
            w.add_tensor(g, gguf.quants.quantize(x, gguf.GGMLQuantizationType.BF16), raw_dtype=gguf.GGMLQuantizationType.BF16)
        else:
            w.add_tensor(g, x)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def selftest():
    import tempfile
    bad = 0

    def check(what, ok):
        nonlocal bad
        print(("ok   " if ok else "FAIL ") + what)
        bad += 0 if ok else 1

    st, cfg = synthetic_model(1, hid=8, heads=2, kvh=1, ff=16, vocab=6, layers=2)
    heads, kvh, layers = 2, 1, 2
    with tempfile.TemporaryDirectory() as d:
        write_gguf(d + "/f32.gguf", st, cfg, "f32")
        write_gguf(d + "/bf16.gguf", st, cfg, "bf16")
        write_gguf(d + "/np.gguf", st, cfg, "f32", permute=False)
        write_safetensors(d + "/m.safetensors", st)
        json.dump(cfg, open(d + "/config.json", "w"))
        analyze("t", d + "/f32.gguf", d + "/bf16.gguf", d + "/m.safetensors", d + "/config.json", d + "/out")
        s = common.load(d + "/out/gguf-summary.json")
        rel = s["f32_summary"]["relations"]
        check("212-style counts: identical = everything but the 4 query and key matrices",
              rel == {"identical": len(st) - 2 * layers, "permuted": 2 * layers})
        check("the permutation is by the head count of the role",
              s["f32_summary"]["permuted_by_n"] == {str(heads): layers, str(kvh): layers} or heads == kvh)
        check("the pairing by names agrees with the pairing by content", s["f32_summary"]["name_pairing_agrees"] == len(st))
        check("bf16 and f32 files hold the same keys", s["bf16_vs_f32"]["bag_roots_equal"])
        c16 = common.load(d + "/out/gguf-bf16-census.json")
        check("bf16 file: BF16 for matrices, F32 for the norm vectors",
              c16["dtypes"] == {"BF16": 1 + 7 * layers, "F32": 1 + 2 * layers})
        ctl = common.load(d + "/out/gguf-controls.json")
        check("C1 and C2 hold", all(ctl.values()))
        ind = common.load(d + "/out/gguf-f32-independent.json")
        check("C5: the two readers agree", ind["mismatch_count"] == 0 and ind["tensors_compared"] == len(st))
        cc = common.load(d + "/out/gguf-config.json")
        check("the key-value table matches config.json", all(r["ok"] for r in cc))
        analyze("u", d + "/np.gguf", d + "/bf16.gguf", d + "/m.safetensors", d + "/config.json", d + "/out2")
        rel2 = common.load(d + "/out2/gguf-summary.json")["f32_summary"]["relations"]
        check("without the permutation every tensor is identical, none permuted", rel2 == {"identical": len(st)})
    return bad


if __name__ == "__main__":
    if sys.argv[1:] == ["selftest"]:
        sys.exit(1 if selftest() else 0)
    if len(sys.argv) == 8 and sys.argv[1] == "analyze":
        analyze(*sys.argv[2:])
    else:
        sys.exit(__doc__)
