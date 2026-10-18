# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Cell 15: synthetic output rows, verified by the reference evaluators (docs/batches/0033.md, amendment 16).

usage: synth.py MODEL_DIR TAG PATCHES.json ONNX_MODEL GGUF_F32 LMNF_LOGITS OUT_DIR
The patches (built locally by synth_rows.py, each a list of {row, hex} writes to the tied embedding matrix) are applied by address in each of the three
encodings: at the byte offsets of the safetensors file, in the ONNX initializer, in the GGUF tensor; the three evaluators (PyTorch, ONNX Runtime, llama.cpp
through lmnf_logits.cpp) produce the last-position logits before and after. Nothing else is changed."""
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys

import numpy as np

IDS = [504, 3575, 282, 4649, 314]
ROW = 576


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for ch in iter(lambda: f.read(1 << 24), b""):
            h.update(ch)
    return h.hexdigest()


def bf16_to_f32(b):
    return (np.frombuffer(b, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def st_layout(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    e = hdr["model.embed_tokens.weight"]
    assert e["dtype"] == "BF16" and e["shape"][1] == ROW
    return 8 + n + e["data_offsets"][0], e["shape"][0]


def torch_logits(model_dir):
    import torch
    from transformers import AutoModelForCausalLM
    try:
        m = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32)
    except TypeError:
        m = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float32)
    m.eval()
    with torch.no_grad():
        z = m(torch.tensor([IDS])).logits[0, -1].float().numpy().copy()
    del m
    return z


def ort_feeds(sess):
    feeds = {}
    for i in sess.get_inputs():
        name, shape = i.name, i.shape
        if name == "input_ids":
            feeds[name] = np.array([IDS], dtype=np.int64)
        elif name == "attention_mask":
            feeds[name] = np.ones((1, len(IDS)), dtype=np.int64)
        elif name == "position_ids":
            feeds[name] = np.arange(len(IDS), dtype=np.int64)[None, :]
        elif name.startswith("past_key_values"):
            dims = [d if isinstance(d, int) else (1 if k == 0 else 0) for k, d in enumerate(shape)]
            feeds[name] = np.zeros(dims, dtype=np.float32)
        else:
            raise SystemExit(f"unexpected ONNX input {name} {shape}")
    return feeds


def ort_logits(onnx_path):
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    out = [o.name for o in sess.get_outputs()]
    z = sess.run([out[0] if "logits" not in out else "logits"], ort_feeds(sess))[0]
    return np.asarray(z, dtype=np.float32)[0, -1].copy()


def onnx_patch(src, dst, writes):
    import onnx
    from onnx import numpy_helper
    m = onnx.load(src)
    inits = [t for t in m.graph.initializer if t.name == "model.embed_tokens.weight"]
    if not inits:
        inits = [t for t in m.graph.initializer if list(t.dims) == [49152, ROW]]
    t = inits[0]
    arr = numpy_helper.to_array(t).copy()
    assert arr.dtype == np.float32 and arr.shape[1] == ROW, (arr.dtype, arr.shape)
    for w in writes:
        arr[w["row"]] = bf16_to_f32(bytes.fromhex(w["hex"]))
    t.CopyFrom(numpy_helper.from_array(arr, t.name))
    onnx.save(m, dst)
    return t.name


def gguf_patch(src, dst, writes):
    from gguf import GGUFReader
    shutil.copyfile(src, dst)
    rd = GGUFReader(dst, "r+")
    t = next(x for x in rd.tensors if x.name == "token_embd.weight")
    assert t.tensor_type.name == "F32", t.tensor_type.name
    flat = t.data.reshape(-1)
    for w in writes:
        flat[w["row"] * ROW:(w["row"] + 1) * ROW] = bf16_to_f32(bytes.fromhex(w["hex"]))
    rd.data.flush()
    del flat, t, rd
    return int(0)


def llama_logits(binary, gguf, out):
    r = subprocess.run([binary, gguf, ",".join(map(str, IDS)), out, "4"], capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"lmnf-logits failed: {r.stderr[-2000:]}")
    return np.fromfile(out, dtype=np.float32)


def compare(name, z0, z1, changed, out_row, comp_row, pred):
    mask = np.ones(z0.shape[0], dtype=bool)
    mask[changed] = False
    return {"evaluator": name, "argmax": int(np.argmax(z1)), "argmax_predicted": pred["argmax"], "argmax_as_predicted": int(np.argmax(z1)) == pred["argmax"],
            f"z{out_row}": float(z1[out_row]), f"z{comp_row}": float(z1[comp_row]),
            "unchanged_rows_differing": int((z1[mask] != z0[mask]).sum()), "unchanged_rows_max_abs_diff": float(np.max(np.abs(z1[mask] - z0[mask]))),
            "top6": [(int(i), float(z1[i])) for i in np.argsort(-z1)[:6]]}


def main():
    model_dir, tag, pj, onnx_model, gguf_f32, lmnf, out = sys.argv[1:8]
    os.makedirs(out, exist_ok=True)
    P = json.load(open(pj))
    out_row, comp_row = P["pair"]["output_row"], P["pair"]["competitor_row"]
    st = os.path.join(model_dir, "model.safetensors")
    data_start, vocab = st_layout(st)
    res = {"model": tag, "model_safetensors_sha256": sha256(st), "data_start": data_start, "patches_file": os.path.basename(pj), "patches_sha256": sha256(pj)}
    # the construction was made from this file: the published rows must equal the swap patch's writes crosswise
    swap = next(p for p in P["patches"] if p["name"] == "A_swap_rows")
    with open(st, "rb") as f:
        pub = {}
        for r in (out_row, comp_row):
            f.seek(data_start + r * ROW * 2)
            pub[r] = f.read(ROW * 2).hex()
    res["identity_check"] = {"row_out_equals_swap_write_for_comp": pub[out_row] == next(w["hex"] for w in swap["writes"] if w["row"] == comp_row),
                             "row_comp_equals_swap_write_for_out": pub[comp_row] == next(w["hex"] for w in swap["writes"] if w["row"] == out_row)}
    print("identity", res["identity_check"])
    # baselines
    z_t0 = torch_logits(model_dir)
    z_o0 = ort_logits(onnx_model)
    z_l0 = llama_logits(lmnf, gguf_f32, os.path.join(out, "l0.f32"))
    np.save(os.path.join(out, f"logits_torch_{tag}.npy"), z_t0)
    res["baseline"] = {ev: {"argmax": int(np.argmax(z)), f"z{out_row}": float(z[out_row]), f"z{comp_row}": float(z[comp_row]), "top6": [(int(i), float(z[i])) for i in np.argsort(-z)[:6]]}
                       for ev, z in (("torch", z_t0), ("onnxruntime", z_o0), ("llamacpp", z_l0))}
    res["baseline"]["agreement"] = {"ort_vs_torch_max_abs": float(np.max(np.abs(z_o0 - z_t0))), "llama_vs_torch_max_abs": float(np.max(np.abs(z_l0 - z_t0))),
                                    "numpy_local_vs_torch": {k: float(P["baseline"][k]) - float(z_t0[int(k[1:])]) for k in (f"z{out_row}", f"z{comp_row}")}}
    print("baseline", json.dumps(res["baseline"]))
    # patches
    work = os.path.join(out, "work")
    os.makedirs(work, exist_ok=True)
    res["patches"] = []
    for p in P["patches"]:
        changed = [w["row"] for w in p["writes"]]
        # safetensors by byte offset, a directory of symlinks around the patched copy
        d = os.path.join(work, f"st_{p['name']}")
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d)
        for fn in os.listdir(model_dir):
            if fn != "model.safetensors" and not fn.startswith("."):
                os.symlink(os.path.abspath(os.path.join(model_dir, fn)), os.path.join(d, fn))
        shutil.copyfile(st, os.path.join(d, "model.safetensors"))
        with open(os.path.join(d, "model.safetensors"), "r+b") as f:
            for w in p["writes"]:
                f.seek(data_start + w["row"] * ROW * 2)
                f.write(bytes.fromhex(w["hex"]))
        z_t = torch_logits(d)
        # ONNX initializer
        op = os.path.join(work, f"onnx_{p['name']}.onnx")
        init_name = onnx_patch(onnx_model, op, p["writes"])
        z_o = ort_logits(op)
        # GGUF tensor
        gp = os.path.join(work, f"gguf_{p['name']}.gguf")
        gguf_patch(gguf_f32, gp, p["writes"])
        z_l = llama_logits(lmnf, gp, os.path.join(out, f"l_{p['name']}.f32"))
        r = {"name": p["name"], "writes": [{"row": w["row"], "safetensors_byte_offset": data_start + w["row"] * ROW * 2} for w in p["writes"]], "onnx_initializer": init_name,
             "prediction": p["prediction"], "construction": p.get("construction"),
             "results": [compare("torch", z_t0, z_t, changed, out_row, comp_row, p["prediction"]), compare("onnxruntime", z_o0, z_o, changed, out_row, comp_row, p["prediction"]),
                         compare("llamacpp", z_l0, z_l, changed, out_row, comp_row, p["prediction"])],
             "cross_evaluator": {"ort_vs_torch_max_abs": float(np.max(np.abs(z_o - z_t))), "llama_vs_torch_max_abs": float(np.max(np.abs(z_l - z_t)))}}
        res["patches"].append(r)
        for q in r["results"]:
            print(f"{tag} {p['name']} {q['evaluator']}: argmax {q['argmax']} (predicted {q['argmax_predicted']}) z{out_row} {q[f'z{out_row}']:.4f} z{comp_row} {q[f'z{comp_row}']:.4f} "
                  f"unchanged rows differing {q['unchanged_rows_differing']} max {q['unchanged_rows_max_abs_diff']:.2e}")
        print(f"{tag} {p['name']} cross: ort-torch {r['cross_evaluator']['ort_vs_torch_max_abs']:.2e} llama-torch {r['cross_evaluator']['llama_vs_torch_max_abs']:.2e}")
        for fn in (op, gp):
            os.remove(fn)
        shutil.rmtree(d, ignore_errors=True)
    json.dump(res, open(os.path.join(out, f"synth_{tag}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
