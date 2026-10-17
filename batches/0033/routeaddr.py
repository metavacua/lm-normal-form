# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Cell 13 (docs/batches/0033.md, amendment 12). Read-only.
usage: routeaddr.py MODEL_DIR ONNX_DIR GGUF_PATH OUT_DIR
The INFER route of LARQL (batches/0033/larql/05_explain_infer.txt, 06_explain_attn.txt) as discrete weight addresses in the three encodings."""
import json
import os
import re
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core  # noqa: E402
from path import capture_all  # noqa: E402
from record import locate_encodings  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def vec(arr, transform, axis_in_st, index):
    """The vector of a tensor of a safetensors layout ((out,in) weight): axis_in_st 0 = row `index`, 1 = column `index`, read from an encoding that stores it as identity or transposed."""
    if transform == "identity":
        return arr[index, :] if axis_in_st == 0 else arr[:, index]
    if transform == "transpose":
        return arr[:, index] if axis_in_st == 0 else arr[index, :]
    raise ValueError(transform)


def main():
    model_dir, onnx_dir, gguf_path, out = sys.argv[1:5]; os.makedirs(out, exist_ok=True)
    model, tok = core.load(model_dir)
    for p in model.parameters():
        p.requires_grad_(False)
    L = model.config.num_hidden_layers; ffn = model.config.intermediate_size; d = model.config.hidden_size
    feats = []
    for ln in open(os.path.join(HERE, "larql", "05_explain_infer.txt"), encoding="utf-8"):
        m = re.match(r"\s+L\s*(\d+):\s+F(\d+)\s+gate=([+-][\d.]+)\s+→\s+(.*?)\s+\[(.*)\]\s*$", ln)
        if m:
            feats.append({"layer": int(m.group(1)), "feature": int(m.group(2)), "gate_printed": float(m.group(3)), "top_token": m.group(4), "down_tokens": m.group(5)})
    att = {}
    for ln in open(os.path.join(HERE, "larql", "06_explain_attn.txt"), encoding="utf-8"):
        m = re.match(r"\s+L\s*(\d+)\s+(.*?)\s+([+-][\d.]+)\s+(\S+?)\((\d+)%\)\s+→\s+(.*?)\s*$", ln)
        if m:
            att[int(m.group(1))] = {"attended_token": m.group(4), "attended_percent": int(m.group(5)), "top_feature_token": m.group(2), "lens_next": m.group(6)}
    assert len(feats) == 150 and len(att) == L, (len(feats), len(att))
    assert len({(f["layer"], f["feature"]) for f in feats}) == 150
    enc, arrays, info, st = locate_encodings(model, onnx_dir, gguf_path)
    ids = core.prompt_ids(tok, "France"); T = ids.shape[1]
    with torch.no_grad():
        c, z = capture_all(model, ids)
    E = model.model.embed_tokens.weight.data; g = model.model.norm.weight.data; xL = c["xL"][-1]; r = torch.sqrt((xL ** 2).mean() + model.config.rms_norm_eps)
    the = int(z.argmax()); paris = core.target_ids(tok, [("France", dict(core.atoms())["France"])])[0]
    ok = {"onnx": 0, "gguf": 0}; total = {"onnx": 0, "gguf": 0}; missing = []
    for f in feats:
        k, i = f["layer"], f["feature"]; f["addresses"] = {}
        for role, tname, axis in (("gate_row", "gate_proj", 0), ("up_row", "up_proj", 0), ("down_col", "down_proj", 1)):
            n = f"model.layers.{k}.mlp.{tname}.weight"; ref = st[n]; refv = ref[i, :] if axis == 0 else ref[:, i]
            a = {"safetensors": {"tensor": n, "index": i, "axis": "row" if axis == 0 else "column", "length": int(refv.shape[0])}}
            for e in ("onnx", "gguf"):
                spec = enc[n][e]
                if spec is None:
                    missing.append((n, e)); a[e] = None; total[e] += 1; continue
                arr = arrays[e][spec["name"]]; v = vec(arr, spec["transform"], axis, i)
                a[e] = {"tensor": spec["name"], "transform": spec["transform"], "index": i, "axis": ("row" if (axis == 0) == (spec["transform"] == "identity") else "column"), "equal": bool(np.array_equal(v, refv))}
                total[e] += 1; ok[e] += int(a[e]["equal"])
            f["addresses"][role] = a
        gate_pre = float(c["gate"][k][-1][i]); act = float(c["act"][k][-1][i]); Wd = model.model.layers[k].mlp.down_proj.weight.data[:, i]
        f["gate_pre_here"] = gate_pre; f["gate_equal_printed"] = bool(abs(gate_pre - f["gate_printed"]) <= max(0.05, 0.01 * abs(f["gate_printed"]))); f["activation"] = act
        f["contribution_to_logit_of_output"] = float(((E[the] * g / r) @ Wd) * act); f["contribution_to_logit_of_paris"] = float(((E[paris] * g / r) @ Wd) * act)
    layers = []
    for k in range(L):
        nm = lambda s: f"model.layers.{k}.{s}.weight"
        tens = {s: {"safetensors": nm(s), "shape": enc[nm(s)]["shape"], "onnx": enc[nm(s)]["onnx"], "gguf": enc[nm(s)]["gguf"]} for s in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "input_layernorm", "post_attention_layernorm")}
        layers.append({"layer": k, "attention": att[k], "attention_and_norm_tensors": tens, "features": [f["feature"] for f in feats if f["layer"] == k]})
    toks = [int(t) for t in ids[0]]
    io = {"prompt_tokens": [{"position": p, "token_id": toks[p], "token": tok.convert_ids_to_tokens(toks[p]), "embedding_row": {"safetensors": "model.embed_tokens.weight", "row": toks[p], "onnx": enc["model.embed_tokens.weight"]["onnx"], "gguf": enc["model.embed_tokens.weight"]["gguf"]}} for p in range(T)],
          "output": {"token_id": the, "token": tok.convert_ids_to_tokens(the), "head_row": {"safetensors": "model.embed_tokens.weight (tied)", "row": the, "onnx": enc["model.embed_tokens.weight"]["onnx"], "gguf": enc["model.embed_tokens.weight"]["gguf"]}},
          "final_norm": {"safetensors": "model.norm.weight", "onnx": enc["model.norm.weight"]["onnx"], "gguf": enc["model.norm.weight"]["gguf"]}}
    n_feat_w = 150 * 3 * d; att_w = sum(int(np.prod(enc[f"model.layers.{k}.self_attn.{s}_proj.weight"]["shape"])) for k in range(L) for s in ("q", "k", "v", "o")); allw = sum(int(np.prod(v["shape"])) for v in enc.values())
    npos = sum(1 for f in feats if f["activation"] > 0 and f["gate_pre_here"] > 0)
    res = {"model_total_weights": allw, "features": feats, "layers": layers, "io": io, "counts": {"feature_vectors": 450, "feature_weights": n_feat_w, "feature_weights_share": n_feat_w / allw, "attention_weights": att_w, "attention_share": att_w / allw,
           "vector_comparisons": total, "vector_equal": ok, "missing_encodings": missing, "features_gate_equal_printed": sum(f["gate_equal_printed"] for f in feats), "features_with_gate_positive": npos,
           "sum_contribution_to_output": sum(f["contribution_to_logit_of_output"] for f in feats), "sum_contribution_to_paris": sum(f["contribution_to_logit_of_paris"] for f in feats)}}
    core.dump(os.path.join(out, "route_addresses.json"), res)
    md = ["| layer | attends (INFER) | feature | gate | down tokens | gate row / up row / down col in safetensors | ONNX | GGUF | contrib to the | contrib to Paris |", "|---|---|---|---|---|---|---|---|---|---|"]
    for f in feats:
        a = f["addresses"]; k = f["layer"]
        md.append(f"| {k} | {att[k]['attended_token']} {att[k]['attended_percent']}% | {f['feature']} | {f['gate_printed']:+.1f} | {f['down_tokens'][:40]} | mlp.gate_proj[{f['feature']},:] / up_proj[{f['feature']},:] / down_proj[:,{f['feature']}] | "
                  f"{a['gate_row']['onnx']['tensor'] if a['gate_row']['onnx'] else None} ({a['gate_row']['onnx']['axis'] if a['gate_row']['onnx'] else '-'}) | {a['gate_row']['gguf']['tensor'] if a['gate_row']['gguf'] else None} | {f['contribution_to_logit_of_output']:+.3f} | {f['contribution_to_logit_of_paris']:+.3f} |")
    open(os.path.join(out, "route_addresses.md"), "w").write("\n".join(md) + "\n")
    cn = res["counts"]
    print(f"V8a vector comparisons equal: onnx {ok['onnx']}/{total['onnx']} gguf {ok['gguf']}/{total['gguf']} missing {len(missing)} | gates equal printed {cn['features_gate_equal_printed']}/150", flush=True)
    print(f"V8b feature weights {n_feat_w} ({n_feat_w / allw:.4%}) attention weights {att_w} ({att_w / allw:.2%}) of {allw}", flush=True)
    print(f"V8c features with gate>0: {npos}/150 | sum contribution to output {cn['sum_contribution_to_output']:+.3f} to Paris {cn['sum_contribution_to_paris']:+.3f}", flush=True)
    print("route layer 0..2:", [(f["layer"], f["feature"], f["gate_printed"], f["down_tokens"][:20]) for f in feats[:10]], flush=True)


if __name__ == "__main__":
    main()
