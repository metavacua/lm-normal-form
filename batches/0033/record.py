# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Cell 05: the IO characterization record (version 1) of docs/batches/0033.md, amendment 2.

usage: record.py MODEL_DIR NAME ONNX_DIR GGUF_FILE OUT_DIR [PROMPT_COUNTRY ...]
Writes OUT_DIR/record_{NAME}_{country}.json and .nt (ground N-Triples), OUT_DIR/encodings_{NAME}.json, and
OUT_DIR/record_{NAME}_check.json (the checks of R1 to R3). Numbers come from hooks and autograd on the loaded
checkpoint; the other encodings are located by content digest of the tensors, never by name alone.
"""
import hashlib
import json
import math
import os
import struct
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core  # noqa: E402

IO = "urn:lmnf:io:v1:"
P = "urn:lmnf:io:p:"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def bf16_hex(v):
    return "0x%04x" % (struct.unpack("<I", struct.pack("<f", float(v)))[0] >> 16)


def digest(a):
    return hashlib.sha1(np.ascontiguousarray(a, dtype=np.float32).tobytes()).hexdigest()


def llama_permute(w, n_head, n_head_kv):
    """llama.cpp convert_hf_to_gguf.py, LlamaModel.permute at b11450: the rotary reordering of q and k rows."""
    if n_head_kv is not None and n_head != n_head_kv:
        n_head = n_head_kv
    return w.reshape(n_head, 2, w.shape[0] // n_head // 2, *w.shape[1:]).swapaxes(1, 2).reshape(w.shape)


# ---------------------------------------------------------------------------------------------- encodings
def locate_encodings(model, onnx_dir, gguf_path):
    nh, nkv = model.config.num_attention_heads, model.config.num_key_value_heads
    st = {n: p.detach().cpu().numpy().astype(np.float32) for n, p in model.named_parameters()}
    out = {n: {"shape": list(a.shape), "onnx": None, "gguf": None} for n, a in st.items()}
    arrays = {"onnx": {}, "gguf": {}}
    info = {"files": {}}
    if os.path.exists(os.path.join(onnx_dir, "model.onnx")):
        import onnx
        from onnx import numpy_helper
        m = onnx.load(os.path.join(onnx_dir, "model.onnx"), load_external_data=True)
        dig = {}
        for t in m.graph.initializer:
            a = numpy_helper.to_array(t)
            if a.dtype != np.float32 or a.ndim == 0:
                continue
            arrays["onnx"][t.name] = a
            dig.setdefault(digest(a), t.name)
        for n, a in st.items():
            if digest(a) in dig:
                out[n]["onnx"] = {"name": dig[digest(a)], "transform": "identity"}
            elif a.ndim == 2 and digest(a.T) in dig:
                out[n]["onnx"] = {"name": dig[digest(a.T)], "transform": "transpose"}
        info["files"]["onnx_sha256"] = sha256_file(os.path.join(onnx_dir, "model.onnx"))
        info["onnx_initializers"] = len(arrays["onnx"]); info["onnx_opsets"] = {o.domain or "ai.onnx": o.version for o in m.opset_import}
        del m
    if os.path.exists(gguf_path):
        from gguf import GGUFReader
        r = GGUFReader(gguf_path); dig = {}
        for tt in r.tensors:
            a = np.asarray(tt.data, dtype=np.float32) if tt.tensor_type.name in ("F32", "F16") else None
            if a is None:
                continue
            arrays["gguf"][tt.name] = a
            dig.setdefault(digest(a), tt.name)
        for n, a in st.items():
            cands = [("identity", a)]
            if n.endswith("q_proj.weight"):
                cands.append(("rotary_permute_q", llama_permute(a, nh, nh)))
            if n.endswith("k_proj.weight"):
                cands.append(("rotary_permute_k", llama_permute(a, nh, nkv)))
            for tr, arr in cands:
                if digest(arr) in dig:
                    out[n]["gguf"] = {"name": dig[digest(arr)], "transform": tr}; break
        info["files"]["gguf_sha256"] = sha256_file(gguf_path); info["gguf_tensors"] = len(arrays["gguf"])
    info["burn"] = "not produced in this batch"
    return out, arrays, info, st


def coord_in_encoding(enc, flat, shape, nh, nkv):
    """The coordinate of the element with flat index `flat` of a safetensors tensor, inside the other encoding."""
    if enc is None:
        return None
    r, c = (flat // shape[1], flat % shape[1]) if len(shape) == 2 else (None, flat)
    tr = enc["transform"]
    if tr == "identity":
        return (r, c) if r is not None else (c,)
    if tr == "transpose":
        return (c, r)
    idx = np.arange(int(np.prod(shape))).reshape(shape)
    perm = (llama_permute(idx, nh, nh) if tr.endswith("_q") else llama_permute(idx, nh, nkv)).reshape(-1)
    pos = int(np.nonzero(perm == flat)[0][0])
    return (pos // shape[1], pos % shape[1])


# ---------------------------------------------------------------------------------------------- the record
def make_record(model, tok, name, X, C, auth, enc, arrays, st, info):
    L = model.config.num_hidden_layers; nh, nkv = model.config.num_attention_heads, model.config.num_key_value_heads
    d_model = model.config.hidden_size; hd = d_model // nh; eps_n = model.config.rms_norm_eps
    t = core.target_ids(tok, [(X, C)])[0]
    ids = core.prompt_ids(tok, X)
    tokens = tok.convert_ids_to_tokens(ids[0].tolist())
    # hooks
    c = {"attn": [None] * L, "mlp": [None] * L, "o_in": [None] * L, "gate": [None] * L, "up": [None] * L, "act": [None] * L}
    hs = [model.model.embed_tokens.register_forward_hook(lambda m, i, o: c.__setitem__("emb", o[0, -1].detach().clone()))]
    for k, Ly in enumerate(model.model.layers):
        hs.append(Ly.self_attn.register_forward_hook(lambda m, i, o, k=k: c["attn"].__setitem__(k, (o[0] if isinstance(o, tuple) else o)[0, -1].detach().clone())))
        hs.append(Ly.self_attn.o_proj.register_forward_pre_hook(lambda m, i, k=k: c["o_in"].__setitem__(k, i[0][0, -1].detach().clone())))
        hs.append(Ly.mlp.register_forward_hook(lambda m, i, o, k=k: c["mlp"].__setitem__(k, o[0, -1].detach().clone())))
        hs.append(Ly.mlp.gate_proj.register_forward_hook(lambda m, i, o, k=k: c["gate"].__setitem__(k, o[0, -1].detach().clone())))
        hs.append(Ly.mlp.up_proj.register_forward_hook(lambda m, i, o, k=k: c["up"].__setitem__(k, o[0, -1].detach().clone())))
        hs.append(Ly.mlp.down_proj.register_forward_pre_hook(lambda m, i, k=k: c["act"].__setitem__(k, i[0][0, -1].detach().clone())))
    hs.append(model.model.norm.register_forward_pre_hook(lambda m, i: c.__setitem__("xL", i[0][0, -1].detach().clone())))
    with torch.no_grad():
        z = model(input_ids=ids, use_cache=False).logits[0, -1].detach().clone()
    for h in hs:
        h.remove()
    E = model.model.embed_tokens.weight.data; g = model.model.norm.weight.data; xL = c["xL"]
    r = torch.sqrt((xL * xL).mean() + eps_n)
    o = z.clone(); o[t] = float("-inf"); j = int(o.argmax())
    dvec = (E[t] - E[j]) * g / r
    margin = float(z[t] - z[j])
    prob = torch.softmax(z.double(), -1); logp = torch.log_softmax(z.double(), -1)
    ent = float(-(prob * logp).sum() / math.log(2))
    rank = int((z > z[t]).sum()) + 1
    top = torch.topk(z, 20)
    # the object string under teacher forcing (compares across tokenizers)
    obj_ids = tok(" " + C, add_special_tokens=False)["input_ids"]
    with torch.no_grad():
        full = torch.cat([ids, torch.tensor([obj_ids])], 1)
        lp = torch.log_softmax(model(input_ids=full, use_cache=False).logits[0].double(), -1)
        n0 = ids.shape[1]
        obj_logp = float(sum(lp[n0 - 1 + k, obj_ids[k]] for k in range(len(obj_ids))))
        gen = model.generate(ids, max_new_tokens=8, do_sample=False, use_cache=False, pad_token_id=tok.eos_token_id)
    # layers
    comps = {"embedding": float(dvec @ c["emb"])}
    for k in range(L):
        comps[f"attn.{k}"] = float(dvec @ c["attn"][k]); comps[f"ffn.{k}"] = float(dvec @ c["mlp"][k])
    # heads
    heads = []
    for k in range(L):
        Wo = model.model.layers[k].self_attn.o_proj.weight.data
        for h in range(nh):
            sl = slice(h * hd, (h + 1) * hd)
            heads.append({"layer": k, "head": h, "kv_group": h // (nh // nkv), "contribution": float(dvec @ (Wo[:, sl] @ c["o_in"][k][sl]))})
    # units
    units_all = []
    for k in range(L):
        Wd = model.model.layers[k].mlp.down_proj.weight.data
        contrib = (dvec @ Wd) * c["act"][k]
        units_all.append(contrib)
    U = torch.stack(units_all)                                   # (L, ffn)
    flat_top = torch.topk(U.abs().view(-1), 50).indices
    units = []
    for f in flat_top.tolist():
        k, i = divmod(f, U.shape[1]); Wd = model.model.layers[k].mlp.down_proj.weight.data
        rd = torch.topk(E @ (g * Wd[:, i]), 5)
        units.append({"layer": k, "index": i, "contribution": float(U[k, i]), "gate_pre": float(c["gate"][k][i]), "up_pre": float(c["up"][k][i]), "activation": float(c["act"][k][i]),
                      "gate_row": [f"model.layers.{k}.mlp.gate_proj.weight", i], "up_row": [f"model.layers.{k}.mlp.up_proj.weight", i], "down_column": [f"model.layers.{k}.mlp.down_proj.weight", i],
                      "value_readout": [{"id": int(a), "token": tok.convert_ids_to_tokens(int(a)), "score": float(b)} for b, a in zip(rd.values, rd.indices)]})
    # lens
    lens = []
    x = c["emb"].clone()
    for k in range(L + 1):
        if k > 0:
            x = x + c["attn"][k - 1] + c["mlp"][k - 1]
        zl = E @ (g * x / torch.sqrt((x * x).mean() + eps_n))
        lens.append({"after_layer": k, "relative_depth": k / L, "target_rank": int((zl > zl[t]).sum()) + 1, "target_logit": float(zl[t]), "target_prob": float(torch.softmax(zl.double(), -1)[t]),
                     "competitor_rank": int((zl > zl[j]).sum()) + 1})
    first_stay = None
    for k in range(L, -1, -1):
        if lens[k]["target_rank"] == 1:
            first_stay = k
        else:
            break
    # nodes and edges (the residual chain)
    chain = ["embedding"] + [n for k in range(L) for n in (f"attn.{k}", f"ffn.{k}")] + ["final_norm", "head"]
    nodes = []
    for n in chain:
        nodes.append({"node": n, "contribution": comps.get(n)})
    # weights: gradient, first order
    core.margin_and_grad(model, tok, X, t)
    ps = core.params(model)
    cand_gw, cand_gh = [], []
    for ti, (pn, p) in enumerate(ps):
        gt = p.grad.view(-1); w = p.data.view(-1)
        gw = (gt * w).abs(); gh = gt.abs() * torch.clamp(0.05 * w.abs(), min=0.002)
        for lst, s in ((cand_gw, gw), (cand_gh, gh)):
            v, i = torch.topk(s, min(50, s.numel())); lst.append((v, torch.full_like(i, ti), i))
    def pick(cand):
        V = torch.cat([a[0] for a in cand]); T = torch.cat([a[1] for a in cand]); I = torch.cat([a[2] for a in cand])
        o_ = torch.topk(V, 50).indices
        return [(int(T[q]), int(I[q])) for q in o_]
    sel = {"by_abs_g_times_w": pick(cand_gw), "by_abs_g_times_h": pick(cand_gh)}
    kind_of = lambda pn, f: ("final_norm_gain" if pn == "model.norm.weight" else ("head_row_target_or_competitor" if (pn == "model.embed_tokens.weight" and f // d_model in (t, j)) else
                             ("embedding_row" if pn == "model.embed_tokens.weight" else ("last_layer_down_proj" if pn == f"model.layers.{L - 1}.mlp.down_proj.weight" else "other"))))
    weights = []; value_checks = []
    for which, lst in sel.items():
        for ti, f in lst:
            pn, p = ps[ti]; shp = list(p.shape); w = float(p.data.view(-1)[f]); gi = float(p.grad.view(-1)[f]); h = max(0.05 * abs(w), 0.002)
            row = {"selection": which, "tensor": pn, "shape": shp, "flat": f, "row": (f // shp[1] if len(shp) == 2 else None), "col": (f % shp[1] if len(shp) == 2 else f),
                   "value": w, "bf16_hex": bf16_hex(w), "gradient": gi, "g_times_w": gi * w, "g_times_h": gi * h, "kind": kind_of(pn, f), "encodings": {}}
            for e in ("onnx", "gguf"):
                en = enc[pn][e]
                if en is None:
                    row["encodings"][e] = None; continue
                pos = coord_in_encoding(en, f, shp, nh, nkv)
                arr = arrays[e].get(en["name"]); val = float(arr[pos]) if (arr is not None and pos is not None) else None
                row["encodings"][e] = {"name": en["name"], "transform": en["transform"], "coordinate": list(pos) if pos else None, "value": val, "equal": (val == w) if val is not None else None}
                value_checks.append(val == w if val is not None else False)
            weights.append(row)
    involved = sorted({w_["tensor"] for w_ in weights} | {u[k_][0] for u in units for k_ in ("gate_row", "up_row", "down_column")})
    tensors = [{"name": n, "shape": enc[n]["shape"], "sha1_float32": digest(st[n]), "onnx": enc[n]["onnx"], "gguf": enc[n]["gguf"], "burn": None} for n in involved]
    # concentration of the first-order attribution (over the whole gradient)
    allgw = torch.cat([(p.grad.view(-1) * p.data.view(-1)).abs() for _, p in ps])
    tot = float(allgw.sum()); conc = {}
    for kq in (10, 100, 1000, 10000):
        conc[f"top_{kq}_share"] = float(torch.topk(allgw, kq).values.sum() / tot)
    del allgw
    a_sum = sum(v for k_, v in comps.items() if k_.startswith("attn")); f_sum = sum(v for k_, v in comps.items() if k_.startswith("ffn"))
    rec = {
        "version": 1, "record": f"{name}/{X}",
        "model": {"name": name, "files": info["files"], "config": {k_: getattr(model.config, k_, None) for k_ in ("hidden_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "intermediate_size", "vocab_size", "rms_norm_eps", "tie_word_embeddings")},
                  "rope_theta": (getattr(model.config, "rope_parameters", None) or {}).get("rope_theta", getattr(model.config, "rope_theta", None)),
                  "parameters": int(sum(p.numel() for p in model.parameters())), "tensors": len(st), "burn": info["burn"]},
        "entity_relation": {"subject": {"label": X, "iri": auth[X]["country"], "token_positions": [i for i, tk in enumerate(tokens) if X.lower() in tk.lower().replace("▁", "").replace("Ġ", "")]},
                            "relation": {"label": "capital", "iri": "http://www.wikidata.org/prop/direct/P36"}, "object": {"label": C, "iri": auth[X]["capital"], "token_ids": obj_ids, "tokens": tok.convert_ids_to_tokens(obj_ids)},
                            "expected_triple": [auth[X]["country"], "http://www.wikidata.org/prop/direct/P36", auth[X]["capital"]]},
        "prompt": {"text": f"The capital of {X} is", "token_ids": ids[0].tolist(), "tokens": tokens},
        "output": {"logits_sha256": hashlib.sha256(z.numpy().tobytes()).hexdigest(), "target_id": t, "target_token": tok.convert_ids_to_tokens(t), "target_logit": float(z[t]), "target_rank": rank,
                   "target_prob": float(prob[t]), "target_logprob": float(logp[t]), "competitor_id": j, "competitor_token": tok.convert_ids_to_tokens(j), "competitor_logit": float(z[j]),
                   "margin_logit": margin, "margin_bits": float((logp[t] - logp[j]) / math.log(2)), "entropy_bits": ent, "object_string_logprob": obj_logp, "greedy_continuation": tok.decode(gen[0][n0:]),
                   "top20": [{"id": int(a), "token": tok.convert_ids_to_tokens(int(a)), "logit": float(b), "prob": float(prob[int(a)])} for b, a in zip(top.values, top.indices)]},
        "layers": {"components": comps, "sum_of_components": sum(comps.values()), "residual_reconstruction_max_diff": float((c["emb"] + sum(c["attn"]) + sum(c["mlp"]) - xL).abs().max()),
                   "lens": lens, "lens_last_equals_logits": float(abs(lens[-1]["target_logit"] - float(z[t]))), "first_layer_target_stays_rank1": first_stay},
        "heads": heads, "heads_sum_per_layer_max_diff": max(abs(sum(h_["contribution"] for h_ in heads if h_["layer"] == k) - comps[f"attn.{k}"]) for k in range(L)),
        "units": units, "units_sum_per_layer_max_diff": max(abs(float(U[k].sum()) - comps[f"ffn.{k}"]) for k in range(L)),
        "nodes": nodes, "weights": weights, "tensors": tensors,
        "comparison": {"margin_bits": float((logp[t] - logp[j]) / math.log(2)), "object_string_logprob": obj_logp, "rank": rank, "entropy_bits": ent, "attention_share_of_signed_sum": a_sum, "ffn_share_of_signed_sum": f_sum,
                       "attribution_concentration": conc, "depth_normalised_trajectory": [(l_["relative_depth"], l_["target_rank"]) for l_ in lens]},
        "checks": {"value_checks": len(value_checks), "values_equal": int(sum(bool(v) for v in value_checks))},
    }
    return rec, comps, margin


def write_nt(rec, path):
    n = 0
    def lit_s(s): return '"' + str(s).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n") + '"'
    def lit_d(x): return f'"{float(x)!r}"^^<http://www.w3.org/2001/XMLSchema#double>'
    def lit_i(i): return f'"{int(i)}"^^<http://www.w3.org/2001/XMLSchema#integer>'
    with open(path, "w", encoding="utf-8") as f:
        def tr(s, p_, o):
            nonlocal n
            f.write(f"<{s}> <{P}{p_}> {o} .\n"); n += 1
        rid = IO + rec["record"]
        tr(rid, "modelName", lit_s(rec["model"]["name"]))
        for k_, v in rec["model"]["files"].items():
            tr(rid, k_, lit_s(v))
        for k_, v in rec["model"]["config"].items():
            tr(rid, "config_" + k_, lit_s(v))
        er = rec["entity_relation"]
        f.write(f"<{er['expected_triple'][0]}> <{er['expected_triple'][1]}> <{er['expected_triple'][2]}> .\n"); n += 1
        tr(rid, "subject", f"<{er['subject']['iri']}>"); tr(rid, "relation", f"<{er['relation']['iri']}>"); tr(rid, "object", f"<{er['object']['iri']}>")
        for i, (tid, ts) in enumerate(zip(rec["prompt"]["token_ids"], rec["prompt"]["tokens"])):
            tk = f"{rid}/tok/{i}"; tr(rid, "hasToken", f"<{tk}>"); tr(tk, "position", lit_i(i)); tr(tk, "id", lit_i(tid)); tr(tk, "string", lit_s(ts))
        o = rec["output"]
        for k_ in ("logits_sha256", "target_token", "competitor_token", "greedy_continuation"):
            tr(rid, k_, lit_s(o[k_]))
        for k_ in ("target_id", "target_rank", "competitor_id"):
            tr(rid, k_, lit_i(o[k_]))
        for k_ in ("target_logit", "target_prob", "target_logprob", "competitor_logit", "margin_logit", "margin_bits", "entropy_bits", "object_string_logprob"):
            tr(rid, k_, lit_d(o[k_]))
        for i, t_ in enumerate(o["top20"]):
            tk = f"{rid}/top/{i}"; tr(rid, "hasTop", f"<{tk}>"); tr(tk, "id", lit_i(t_["id"])); tr(tk, "string", lit_s(t_["token"])); tr(tk, "logit", lit_d(t_["logit"])); tr(tk, "prob", lit_d(t_["prob"]))
        prev = None
        for i, nd in enumerate(rec["nodes"]):
            nu = f"{rid}/node/{nd['node']}"; tr(rid, "hasNode", f"<{nu}>"); tr(nu, "order", lit_i(i))
            if nd["contribution"] is not None:
                tr(nu, "marginContribution", lit_d(nd["contribution"]))
            if prev:
                tr(prev, "next", f"<{nu}>")
            prev = nu
        for h_ in rec["heads"]:
            hu = f"{rid}/head/{h_['layer']}/{h_['head']}"; tr(hu, "partOf", f"<{rid}/node/attn.{h_['layer']}>"); tr(hu, "kvGroup", lit_i(h_["kv_group"])); tr(hu, "marginContribution", lit_d(h_["contribution"]))
        for u in rec["units"]:
            uu = f"{rid}/unit/{u['layer']}/{u['index']}"; tr(uu, "partOf", f"<{rid}/node/ffn.{u['layer']}>"); tr(uu, "marginContribution", lit_d(u["contribution"]))
            tr(uu, "gatePre", lit_d(u["gate_pre"])); tr(uu, "upPre", lit_d(u["up_pre"])); tr(uu, "activation", lit_d(u["activation"]))
            for k_ in ("gate_row", "up_row", "down_column"):
                tr(uu, k_, lit_s(f"{u[k_][0]}#{u[k_][1]}"))
            for q, rd in enumerate(u["value_readout"]):
                tr(uu, "raises", f"<{uu}/raises/{q}>"); tr(f"{uu}/raises/{q}", "id", lit_i(rd["id"])); tr(f"{uu}/raises/{q}", "string", lit_s(rd["token"])); tr(f"{uu}/raises/{q}", "score", lit_d(rd["score"]))
        for i, w in enumerate(rec["weights"]):
            wu = f"{rid}/weight/{i}"; tr(rid, "hasWeight", f"<{wu}>"); tr(wu, "selection", lit_s(w["selection"])); tr(wu, "tensor", lit_s(w["tensor"])); tr(wu, "flat", lit_i(w["flat"]))
            tr(wu, "value", lit_d(w["value"])); tr(wu, "gradient", lit_d(w["gradient"])); tr(wu, "gTimesW", lit_d(w["g_times_w"])); tr(wu, "kind", lit_s(w["kind"]))
            for e, ev in w["encodings"].items():
                if ev:
                    tr(wu, e + "Name", lit_s(ev["name"])); tr(wu, e + "Transform", lit_s(ev["transform"])); tr(wu, e + "Coordinate", lit_s(ev["coordinate"])); tr(wu, e + "Equal", lit_s(ev["equal"]))
        for t_ in rec["tensors"]:
            tu = f"{rid}/tensor/{t_['name']}"; tr(rid, "hasTensor", f"<{tu}>"); tr(tu, "shape", lit_s(t_["shape"])); tr(tu, "sha1Float32", lit_s(t_["sha1_float32"]))
            for e in ("onnx", "gguf"):
                if t_[e]:
                    tr(tu, e + "Name", lit_s(t_[e]["name"])); tr(tu, e + "Transform", lit_s(t_[e]["transform"]))
    return n


def main():
    model_dir, name, onnx_dir, gguf_path, out = sys.argv[1:6]
    prompts = sys.argv[6:] or ["France", "Spain"]
    os.makedirs(out, exist_ok=True)
    clock = core.Clock()
    model, tok = core.load(model_dir)
    auth = {x["countryLabel"]: {"country": x["country"], "capital": x["canonical"][0]["capital"]} for x in json.load(open(os.path.join(core.HERE, "..", "0032", "countries.json"))) if x["determined"]}
    caps = dict(core.atoms())
    enc, arrays, info, st = locate_encodings(model, onnx_dir, gguf_path)
    info["files"]["safetensors_sha256"] = sha256_file(os.path.join(model_dir, "model.safetensors"))
    core.dump(os.path.join(out, f"encodings_{name}.json"), {"info": info, "tensors": enc})
    checks = {"model": name, "records": {}}
    import pyoxigraph as ox
    for X in prompts:
        rec, comps, margin = make_record(model, tok, name, X, caps[X], auth, enc, arrays, st, info)
        core.dump(os.path.join(out, f"record_{name}_{X}.json"), rec)
        ntp = os.path.join(out, f"record_{name}_{X}.nt"); n = write_nt(rec, ntp)
        store = ox.Store(); store.bulk_load(path=ntp, format=ox.RdfFormat.N_TRIPLES)
        blank = bool(store.query("ASK { ?s ?p ?o FILTER(isBlank(?s) || isBlank(?o)) }")); loaded = sum(1 for _ in store)
        L = rec["model"]["config"]["num_hidden_layers"]
        checks["records"][X] = {"margin": margin, "sum_of_components_diff": abs(rec["layers"]["sum_of_components"] - margin), "residual_reconstruction_max_diff": rec["layers"]["residual_reconstruction_max_diff"],
                                "heads_sum_max_diff": rec["heads_sum_per_layer_max_diff"], "units_sum_max_diff": rec["units_sum_per_layer_max_diff"], "lens_last_diff": rec["layers"]["lens_last_equals_logits"],
                                "triples": n, "triples_loaded": loaded, "blank_nodes": blank, "value_checks": rec["checks"],
                                "tensors_located": {"onnx": sum(1 for t_ in rec["tensors"] if t_["onnx"]), "gguf": sum(1 for t_ in rec["tensors"] if t_["gguf"]), "of": len(rec["tensors"])},
                                "target_rank": rec["output"]["target_rank"], "greedy": rec["output"]["greedy_continuation"], "first_layer_target_stays_rank1": rec["layers"]["first_layer_target_stays_rank1"],
                                "final_lens_rank": rec["layers"]["lens"][-1]["target_rank"]}
        print(X, checks["records"][X], f"{clock.s()} s", flush=True)
    core.dump(os.path.join(out, f"record_{name}_check.json"), checks)


if __name__ == "__main__":
    main()
