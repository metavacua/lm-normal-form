# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Cell 15: synthetic output rows, verified by the reference evaluators (docs/batches/0033.md, amendment 16).

usage: synth.py MODEL_DIR TAG PATCHES.json ONNX_MODEL GGUF_F32 LMNF_LOGITS OUT_DIR
The patches (built locally by synth_rows.py, each a list of {row, hex} writes to the tied embedding matrix) are applied by address in each of the three
encodings: at the byte offsets of the rows in the safetensors file, in the ONNX file (in both copies of the tied matrix the export keeps, the lookup's
initializer and the head's) and in the GGUF file; the three evaluators (PyTorch, ONNX Runtime, llama.cpp through lmnf_logits.cpp) produce the
last-position logits before and after. Nothing else is changed."""
import json
import mmap
import os
import shutil
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from synth_common import bf16_to_f32, sha256_file, st_header, st_tensor  # noqa: E402


def torch_logits(model_dir, ids):
    import torch
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32).eval()
    with torch.no_grad():
        return m(torch.tensor([ids])).logits[0, -1].float().numpy().copy()


def ort_logits(onnx_path, ids):
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    n = len(ids)
    feeds = {"input_ids": np.array([ids], dtype=np.int64), "attention_mask": np.ones((1, n), dtype=np.int64), "position_ids": np.arange(n, dtype=np.int64)[None, :]}
    names = [i.name for i in sess.get_inputs()]
    if set(names) - set(feeds):
        raise SystemExit(f"unexpected ONNX inputs {names}")
    return np.asarray(sess.run(["logits"], {k: feeds[k] for k in names})[0], dtype=np.float32)[0, -1].copy()


def llama_logits(binary, gguf, ids, out):
    r = subprocess.run([binary, gguf, ",".join(map(str, ids)), out], capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"lmnf-logits failed: {r.stderr[-2000:]}")
    return np.fromfile(out, dtype=np.float32)


def evaluate(st_dir, onnx_path, gguf_path, ids, lmnf, f32_out):
    """The last-position logits of the three evaluators, each from its own encoding of the same weights."""
    return {"torch": torch_logits(st_dir, ids), "onnxruntime": ort_logits(onnx_path, ids), "llamacpp": llama_logits(lmnf, gguf_path, ids, f32_out)}


def onnx_head(m, inits, by_out):
    """The initializer the logits multiplication reads: walk back from the logits output to the first MatMul or Gemm, through a Transpose if there is one."""
    start = next((o.name for o in m.graph.output if "logits" in o.name), m.graph.output[0].name)
    seen, stack = set(), [start]
    while stack:
        nm = stack.pop()
        if nm in seen or nm not in by_out:
            continue
        seen.add(nm)
        n = by_out[nm]
        if n.op_type in ("MatMul", "Gemm"):
            for i in n.input:
                if i in inits:
                    return n, inits[i], None
                if i in by_out and by_out[i].op_type == "Transpose" and by_out[i].input[0] in inits:
                    return n, inits[by_out[i].input[0]], by_out[i]
        stack.extend(n.input)
    raise SystemExit("no MatMul or Gemm feeds the logits")


def onnx_layout(path, cols):
    """Where the tied matrix is in the ONNX file, found from the graph and not by name: the initializer the token lookup (the Gather on input_ids) reads
    and the one the logits multiplication reads, each with the byte offset of its data in the file. Run 39095581261 showed why both are needed: the
    Optimum export keeps its own transposed copy of the tied matrix for the head (`onnx::MatMul_9693`), and a write to `model.embed_tokens.weight`
    reaches the lookup only."""
    import onnx
    m = onnx.load(path)
    inits = {t.name: t for t in m.graph.initializer}
    by_out = {o: n for n in m.graph.node for o in n.output}

    def from_input_ids(name):
        return name == "input_ids" or (name in by_out and "input_ids" in by_out[name].input)
    emb = next(inits[n.input[0]] for n in m.graph.node if n.op_type == "Gather" and n.input[0] in inits and from_input_ids(n.input[1]))
    node, head, transp = onnx_head(m, inits, by_out)
    lay = {"lookup_initializer": emb.name, "lookup_dims": list(emb.dims), "head_node": node.name, "head_initializer": head.name, "head_dims": list(head.dims),
           "head_through_transpose_node": transp.name if transp is not None else None, "one_copy": head.name == emb.name, "data_byte_offset": {}}
    with open(path, "rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        for t in (emb, head):
            assert t.data_type == onnx.TensorProto.FLOAT and t.HasField("raw_data") and cols in list(t.dims), (t.name, t.data_type, list(t.dims))
            raw = t.raw_data
            o = mm.find(raw[:4096])
            assert o >= 0 and mm[o:o + len(raw)] == raw, t.name
            lay["data_byte_offset"][t.name] = o
    return lay


def onnx_views(path, lay, cols, mode):
    """The copies of the tied matrix in the ONNX file at path, as row-addressable arrays (the head's through a transposed view when the export stores
    it transposed), and the memmaps behind them; with mode "r+" a write to a view reaches the file."""
    views, maps = {}, []
    for name, dims in ((lay["lookup_initializer"], lay["lookup_dims"]), (lay["head_initializer"], lay["head_dims"])):
        a = np.memmap(path, dtype=np.float32, mode=mode, offset=lay["data_byte_offset"][name], shape=tuple(dims))
        maps.append(a)
        views[name] = a if dims[1] == cols else a.T
    return views, maps


def onnx_patch(src, dst, writes, lay, cols):
    """A copy of the ONNX file with the rows written into every copy of the tied matrix. Returns, per written row, how many entries of the head's copy
    equalled the lookup's before the write."""
    shutil.copyfile(src, dst)
    views, maps = onnx_views(dst, lay, cols, "r+")
    tie = {w["row"]: int((views[lay["head_initializer"]][w["row"]] == views[lay["lookup_initializer"]][w["row"]]).sum()) for w in writes}
    for w in writes:
        v = bf16_to_f32(bytes.fromhex(w["hex"]))
        for a in views.values():
            a[w["row"]] = v
    for a in maps:
        a.flush()
    return tie


def gguf_embed(path, mode):
    """The GGUF reader of the file and its token_embd.weight tensor (float32, row-addressable through the reader's memmap)."""
    from gguf import GGUFReader
    rd = GGUFReader(path, mode)
    t = next(x for x in rd.tensors if x.name == "token_embd.weight")
    assert t.tensor_type.name == "F32" and t.data.shape[1:] == (int(t.shape[0]),), (t.tensor_type.name, t.data.shape)
    return rd, t


def gguf_patch(src, dst, writes):
    """A copy of the GGUF file with the rows written into token_embd.weight, in place through the reader's memmap."""
    shutil.copyfile(src, dst)
    rd, t = gguf_embed(dst, "r+")
    for w in writes:
        t.data[w["row"]] = bf16_to_f32(bytes.fromhex(w["hex"]))
    rd.data.flush()


def st_patch(model_dir, d, writes, st0, cols):
    """A directory of symlinks to the model's files around a copy of the safetensors file with the rows written at their byte offsets."""
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    for fn in os.listdir(model_dir):
        if fn != "model.safetensors" and not fn.startswith("."):
            os.symlink(os.path.abspath(os.path.join(model_dir, fn)), os.path.join(d, fn))
    shutil.copyfile(os.path.join(model_dir, "model.safetensors"), os.path.join(d, "model.safetensors"))
    with open(os.path.join(d, "model.safetensors"), "r+b") as f:
        for w in writes:
            f.seek(st0 + w["row"] * cols * 2)
            f.write(bytes.fromhex(w["hex"]))


def summary(z, pair):
    out_row, comp_row = pair
    return {"argmax": int(np.argmax(z)), f"z{out_row}": float(z[out_row]), f"z{comp_row}": float(z[comp_row]), "top6": [(int(i), float(z[i])) for i in np.argsort(-z)[:6]]}


def cross(z):
    """The largest absolute difference of the other two evaluators' logits from PyTorch's."""
    return {"ort_vs_torch_max_abs": float(np.max(np.abs(z["onnxruntime"] - z["torch"]))), "llama_vs_torch_max_abs": float(np.max(np.abs(z["llamacpp"] - z["torch"])))}


def compare(name, z0, z1, changed, pair, pred):
    mask = np.ones(z0.shape[0], dtype=bool)
    mask[changed] = False
    return {"evaluator": name, **summary(z1, pair), "argmax_predicted": pred["argmax"], "argmax_as_predicted": int(np.argmax(z1)) == pred["argmax"],
            "unchanged_rows_differing": int((z1[mask] != z0[mask]).sum()), "unchanged_rows_max_abs_diff": float(np.max(np.abs(z1[mask] - z0[mask])))}


def main():
    model_dir, tag, pj, onnx_model, gguf_f32, lmnf, out = sys.argv[1:8]
    work = os.path.join(out, "work")
    os.makedirs(work, exist_ok=True)
    with open(pj) as f:
        P = json.load(f)
    ids, pair = P["pair"]["prompt_ids"], (P["pair"]["output_row"], P["pair"]["competitor_row"])
    out_row, comp_row = pair
    st = os.path.join(model_dir, "model.safetensors")
    hdr, data_start = st_header(st)
    st0, (_, cols) = st_tensor(hdr, data_start, "model.embed_tokens.weight")
    res = {"model": tag, "prompt_ids": ids, "model_safetensors_sha256": sha256_file(st), "data_start": st0, "patches_file": os.path.basename(pj), "patches_sha256": sha256_file(pj)}
    # the construction was made from this file: the published rows must equal the swap patch's writes crosswise
    swap = {w["row"]: w["hex"] for w in next(p for p in P["patches"] if p["name"] == "A_swap_rows")["writes"]}
    with open(st, "rb") as f:
        pub = {}
        for r in pair:
            f.seek(st0 + r * cols * 2)
            pub[r] = f.read(cols * 2).hex()
    res["identity_check"] = {"row_out_equals_swap_write_for_comp": pub[out_row] == swap[comp_row], "row_comp_equals_swap_write_for_out": pub[comp_row] == swap[out_row]}
    print("identity", res["identity_check"])
    # where the tied matrix is in the other two encodings
    res["onnx"] = lay = onnx_layout(onnx_model, cols)
    res["gguf"] = {"tensor": "token_embd.weight", "data_byte_offset": int(gguf_embed(gguf_f32, "r")[1].data_offset)}
    print("onnx", json.dumps(lay), "gguf", json.dumps(res["gguf"]))
    # baselines
    z0 = evaluate(model_dir, onnx_model, gguf_f32, ids, lmnf, os.path.join(work, "baseline.f32"))
    res["baseline"] = {ev: summary(z, pair) for ev, z in z0.items()}
    res["baseline"]["agreement"] = {**cross(z0), "numpy_local_vs_torch": {f"z{r}": float(P["baseline"][f"z{r}"]) - float(z0["torch"][r]) for r in pair}}
    print("baseline", json.dumps(res["baseline"]))
    # patches: each applied to fresh copies of the three files, evaluated, and removed
    res["patches"] = []
    for p in P["patches"]:
        changed = [w["row"] for w in p["writes"]]
        d = os.path.join(work, f"st_{p['name']}")
        op = os.path.join(work, f"onnx_{p['name']}.onnx")
        gp = os.path.join(work, f"gguf_{p['name']}.gguf")
        st_patch(model_dir, d, p["writes"], st0, cols)
        tie = onnx_patch(onnx_model, op, p["writes"], lay, cols)
        gguf_patch(gguf_f32, gp, p["writes"])
        z1 = evaluate(d, op, gp, ids, lmnf, os.path.join(work, f"{p['name']}.f32"))
        r = {"name": p["name"],
             "writes": [{"row": w["row"], "safetensors_byte_offset": st0 + w["row"] * cols * 2, "gguf_byte_offset": res["gguf"]["data_byte_offset"] + w["row"] * cols * 4} for w in p["writes"]],
             "onnx_tie_before_patch_equal_entries": tie, "prediction": p["prediction"], "construction": p.get("construction"),
             "results": [compare(ev, z0[ev], z1[ev], changed, pair, p["prediction"]) for ev in z0], "cross_evaluator": cross(z1)}
        res["patches"].append(r)
        for q in r["results"]:
            print(f"{tag} {p['name']} {q['evaluator']}: argmax {q['argmax']} (predicted {q['argmax_predicted']}) z{out_row} {q[f'z{out_row}']:.4f} z{comp_row} {q[f'z{comp_row}']:.4f} "
                  f"unchanged rows differing {q['unchanged_rows_differing']} max {q['unchanged_rows_max_abs_diff']:.2e}")
        print(f"{tag} {p['name']} onnx tie before the write {tie} | cross: ort-torch {r['cross_evaluator']['ort_vs_torch_max_abs']:.2e} llama-torch {r['cross_evaluator']['llama_vs_torch_max_abs']:.2e}")
        os.remove(op)
        os.remove(gp)
        shutil.rmtree(d)
    with open(os.path.join(out, f"synth_{tag}.json"), "w") as f:
        json.dump(res, f, indent=1)
        f.write("\n")


if __name__ == "__main__":
    main()
