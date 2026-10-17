# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Cell 07: the read-only isolation of the IO path (docs/batches/0033.md, amendment 4). Nothing is zeroed, masked or patched.

usage: trace.py MODEL_DIR NAME OUT_DIR [COUNTRY ...]
1. The executed operations (TorchDispatchMode) with the module that issued each (ModuleTracker), the parameters each read and
   how many elements, the producer-consumer edges; checked acyclic and complete against the parameter list.
2. The structural cone: gradients of the margin with respect to every intermediate tensor and attention probability.
3. The exact flow: node contributions, head contributions by source position, with conservation checks.
Writes OUT_DIR/trace_{NAME}_{X}.json and .nt (ground N-Triples)."""
import os
import sys
import time

import numpy as np
import torch
import torch.utils._pytree as pytree
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.module_tracker import ModuleTracker

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core  # noqa: E402

P = "urn:lmnf:trace:p:"
VIEW_OPS = {"aten.t.default", "aten.transpose.int", "aten.view.default", "aten._unsafe_view.default", "aten.expand.default", "aten.reshape.default", "aten.permute.default",
            "aten.unsqueeze.default", "aten.squeeze.dim", "aten.squeeze.dims", "aten.slice.Tensor", "aten.select.int", "aten.alias.default", "aten.detach.default", "aten.lift_fresh.default",
            "aten.split.Tensor", "aten.split_with_sizes.default", "aten.chunk.default", "aten.unbind.int", "aten.as_strided.default", "aten.contiguous.default"}


class Tracer(TorchDispatchMode):
    def __init__(self, pmap, mt):
        super().__init__()
        self.pmap, self.mt = pmap, mt
        self.ops = []; self.keep = []; self.prod = {}; self.edges = []; self.reads = {}

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        fname = str(func); i = len(self.ops)
        mods = [m for m in self.mt.parents if m != "Global"]
        module = max(mods, key=len) if mods else "Global"
        tens = [a for a in pytree.tree_leaves((args, kwargs)) if isinstance(a, torch.Tensor)]
        preads = []; srcs = []
        for a in tens:
            try:
                ptr = a.untyped_storage().data_ptr()
            except Exception:
                ptr = None
            if ptr in self.pmap:
                if fname in VIEW_OPS:
                    continue
                name = self.pmap[ptr]
                if fname == "aten.embedding.default":
                    rows = torch.unique(args[1]).tolist(); n = len(rows) * a.shape[-1]
                    self.reads.setdefault(name, {"ops": [], "elements": 0, "rows": set()})["rows"].update(rows)
                else:
                    n = a.numel()
                rec = self.reads.setdefault(name, {"ops": [], "elements": 0, "rows": set()}); rec["ops"].append(i); rec["elements"] = max(rec["elements"], n) if fname != "aten.embedding.default" else len(rec["rows"]) * a.shape[-1]
                preads.append(name)
            elif id(a) in self.prod:
                srcs.append(self.prod[id(a)])
        for o in pytree.tree_leaves(out):
            if isinstance(o, torch.Tensor):
                self.prod[id(o)] = i; self.keep.append(o)
        for a in tens:
            self.keep.append(a)
        self.ops.append({"i": i, "op": fname, "module": module, "reads": preads, "from": sorted(set(srcs)), "out_shape": [list(o.shape) for o in pytree.tree_leaves(out) if isinstance(o, torch.Tensor)][:2]})
        return out


def trace_execution(model, ids):
    pmap = {}
    for n, p in model.named_parameters():
        pmap[p.untyped_storage().data_ptr()] = n
    mt = ModuleTracker()
    with mt:
        tr = Tracer(pmap, mt)
        with tr, torch.inference_mode():
            model(input_ids=ids, use_cache=False)
    return tr


def cone_and_flow(model, tok, X, C):
    L = model.config.num_hidden_layers; nh = model.config.num_attention_heads; nkv = model.config.num_key_value_heads; hd = model.config.hidden_size // nh; eps_n = model.config.rms_norm_eps
    ids = core.prompt_ids(tok, X); T = ids.shape[1]; t = core.target_ids(tok, [(X, C)])[0]
    store = {"layer": [None] * L, "attn": [None] * L, "mlp": [None] * L, "act": [None] * L, "gate": [None] * L, "up": [None] * L, "probs": [None] * L, "v": [None] * L, "o_in": [None] * L}
    hs = []
    def keep(slot, k):
        def f(m, i, o):
            x = o[0] if isinstance(o, tuple) else o
            if x.requires_grad:
                x.retain_grad()
            store[slot][k] = x
            if slot == "attn" and isinstance(o, tuple) and o[1] is not None:
                w = o[1]
                if w.requires_grad:
                    w.retain_grad()
                store["probs"][k] = w
        return f
    for k, Ly in enumerate(model.model.layers):
        hs.append(Ly.register_forward_hook(keep("layer", k)))
        hs.append(Ly.self_attn.register_forward_hook(keep("attn", k)))
        hs.append(Ly.mlp.register_forward_hook(keep("mlp", k)))
        hs.append(Ly.mlp.gate_proj.register_forward_hook(keep("gate", k)))
        hs.append(Ly.mlp.up_proj.register_forward_hook(keep("up", k)))
        hs.append(Ly.self_attn.v_proj.register_forward_hook(keep("v", k)))
        def pre_act(m, i, k=k):
            x = i[0]
            if x.requires_grad:
                x.retain_grad()
            store["act"][k] = x
        hs.append(Ly.mlp.down_proj.register_forward_pre_hook(pre_act))
        def pre_o(m, i, k=k):
            store["o_in"][k] = i[0].detach()
        hs.append(Ly.self_attn.o_proj.register_forward_pre_hook(pre_o))
    for p in model.parameters():
        p.grad = None
    with torch.enable_grad():
        z = model(input_ids=ids, use_cache=False, output_attentions=True).logits[0, -1]
        o = z.detach().clone(); o[t] = float("-inf"); j = int(o.argmax())
        margin = z[t] - z[j]
        margin.backward()
    for h in hs:
        h.remove()
    torch.set_grad_enabled(False)
    # cone
    cone = []
    for k in range(L):
        row = {"layer": k}
        for slot in ("layer", "attn", "mlp"):
            g = store[slot][k].grad
            row[slot + "_positions_nonzero"] = [] if g is None else [int(q) for q in range(T) if bool((g[0, q] != 0).any())]
        ga = store["act"][k].grad
        row["act_nonzero_by_position"] = [int((ga[0, q] != 0).sum()) for q in range(T)] if ga is not None else None
        gp = store["probs"][k].grad if store["probs"][k] is not None else None
        if gp is not None:
            g0 = gp[0]; above = bool((torch.triu(g0, diagonal=1) != 0).any())
            row["attention_prob_nonzero_pairs"] = int((g0 != 0).sum()); row["attention_prob_nonzero_above_diagonal"] = above
            row["attention_prob_query_positions_nonzero"] = sorted({int(q) for q in torch.nonzero((g0 != 0).any(-1))[:, 1].tolist()})
        cone.append(row)
    # flow, from the same forward (values detached)
    E = model.model.embed_tokens.weight.data; g_ = model.model.norm.weight.data
    xL = (model.model.embed_tokens(ids)[0, -1].detach() + sum(store["attn"][k][0, -1].detach() + store["mlp"][k][0, -1].detach() for k in range(L)))
    r = torch.sqrt((xL ** 2).mean() + eps_n); dvec = (E[t] - E[j]) * g_ / r
    comps = {"embedding": float(dvec @ model.model.embed_tokens(ids)[0, -1].detach())}
    heads = []; bysrc = {}; max_split_err = 0.0; max_head_err = 0.0
    for k in range(L):
        a_k = store["attn"][k][0, -1].detach(); comps[f"attn.{k}"] = float(dvec @ a_k); comps[f"ffn.{k}"] = float(dvec @ store["mlp"][k][0, -1].detach())
        Wo = model.model.layers[k].self_attn.o_proj.weight.data; w = store["probs"][k][0].detach(); v = store["v"][k][0].detach()
        head_sum = 0.0
        for h in range(nh):
            grp = h // (nh // nkv); vh = v[:, grp * hd:(grp + 1) * hd]; Woh = Wo[:, h * hd:(h + 1) * hd]
            per_src = [float(dvec @ (Woh @ (w[h, -1, p] * vh[p]))) for p in range(T)]
            direct = float(dvec @ (Woh @ store["o_in"][k][0, -1][h * hd:(h + 1) * hd]))
            max_split_err = max(max_split_err, abs(sum(per_src) - direct)); head_sum += direct
            for p in range(T):
                heads.append({"layer": k, "head": h, "kv_group": grp, "source_position": p, "attention": float(w[h, -1, p]), "contribution": per_src[p]})
                bysrc[p] = bysrc.get(p, 0.0) + per_src[p]
        max_head_err = max(max_head_err, abs(head_sum - comps[f"attn.{k}"]))
    top = sorted(heads, key=lambda d_: -abs(d_["contribution"]))[:25]
    tokens = tok.convert_ids_to_tokens(ids[0].tolist())
    for d_ in top:
        d_["source_token"] = tokens[d_["source_position"]]
    absmass = {p: sum(abs(d_["contribution"]) for d_ in heads if d_["source_position"] == p) for p in range(T)}
    return {"prompt_tokens": tokens, "target": t, "competitor": j, "margin": float(margin.detach()), "cone": cone, "components": comps, "sum_components": sum(comps.values()),
            "head_split_max_err": max_split_err, "head_to_layer_max_err": max_head_err, "top_head_source_contributions": top, "signed_contribution_by_source_position": bysrc,
            "absolute_contribution_by_source_position": absmass, "prompt_ids": ids[0].tolist()}


def write_nt(res, path, name, X):
    n = 0
    def s_(x): return '"' + str(x).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n") + '"'
    def i_(x): return f'"{int(x)}"^^<http://www.w3.org/2001/XMLSchema#integer>'
    base = f"urn:lmnf:trace:v1:{name}/{X}"
    with open(path, "w", encoding="utf-8") as f:
        def tr(s, p, o):
            nonlocal n
            f.write(f"<{s}> <{P}{p}> {o} .\n"); n += 1
        for op in res["ops"]:
            u = f"{base}/op/{op['i']}"
            tr(u, "aten", s_(op["op"])); tr(u, "module", s_(op["module"])); tr(u, "order", i_(op["i"]))
            for pn in op["reads"]:
                tr(u, "readsParameter", f"<urn:lmnf:trace:v1:{name}/param/{pn}>")
            for s in op["from"]:
                tr(f"{base}/op/{s}", "feeds", f"<{u}>")
        for pn, rd in res["parameters_read"].items():
            pu = f"urn:lmnf:trace:v1:{name}/param/{pn}"; tr(pu, "elementsRead", i_(rd["elements_read"])); tr(pu, "elementsTotal", i_(rd["elements_total"]))
    return n


def main():
    model_dir, name, out = sys.argv[1:4]; prompts = sys.argv[4:] or ["France", "Spain"]
    os.makedirs(out, exist_ok=True)
    model, tok = core.load(model_dir)
    caps = dict(core.atoms())
    import pyoxigraph as ox
    for X in prompts:
        t0 = time.time()
        ids = core.prompt_ids(tok, X)
        tr = trace_execution(model, ids)
        total = {n: p.numel() for n, p in model.named_parameters()}
        parameters_read = {n: {"elements_read": tr.reads.get(n, {"elements": 0})["elements"], "elements_total": total[n], "ops": len(tr.reads.get(n, {"ops": []})["ops"]),
                               "rows_read": (sorted(tr.reads[n]["rows"]) if n in tr.reads and tr.reads[n]["rows"] else None)} for n in total}
        acyclic = all(all(s < op["i"] for s in op["from"]) for op in tr.ops)
        by_module = {}
        for op in tr.ops:
            k = op["module"].split(".")[0:3]; by_module[".".join(k)] = by_module.get(".".join(k), 0) + 1
        opcount = {}
        for op in tr.ops:
            opcount[op["op"]] = opcount.get(op["op"], 0) + 1
        t1 = time.time()
        cf = cone_and_flow(model, tok, X, caps[X])
        res = {"model": name, "prompt": X, "ops": tr.ops, "n_ops": len(tr.ops), "acyclic": acyclic, "ops_by_aten": dict(sorted(opcount.items(), key=lambda kv: -kv[1])[:25]),
               "parameters_read": parameters_read, "parameters_total_elements": sum(total.values()),
               "parameters_read_elements": sum(v["elements_read"] for v in parameters_read.values()), "parameters_unread": [n for n, v in parameters_read.items() if v["elements_read"] == 0],
               "parameters_partially_read": {n: (v["elements_read"], v["elements_total"]) for n, v in parameters_read.items() if 0 < v["elements_read"] < v["elements_total"]},
               "seconds_trace": round(t1 - t0, 2), **cf, "seconds_total": round(time.time() - t0, 2)}
        core.dump(os.path.join(out, f"trace_{name}_{X}.json"), {k: v for k, v in res.items() if k != "ops"} | {"ops_head": tr.ops[:40]})
        ntp = os.path.join(out, f"trace_{name}_{X}.nt"); n = write_nt(res, ntp, name, X)
        store = ox.Store(); store.bulk_load(path=ntp, format=ox.RdfFormat.N_TRIPLES)
        blank = bool(store.query("ASK { ?s ?p ?o FILTER(isBlank(?s) || isBlank(?o)) }"))
        L = model.config.num_hidden_layers
        last = res["cone"][L - 1]
        print(f"{name} {X}: ops {res['n_ops']} acyclic {acyclic} | params read {res['parameters_read_elements']:,}/{res['parameters_total_elements']:,} unread tensors {len(res['parameters_unread'])} partial {res['parameters_partially_read']} | "
              f"cone last layer positions nonzero layer {last['layer_positions_nonzero']} attn {last['attn_positions_nonzero']} mlp {last['mlp_positions_nonzero']} | above-diagonal grad any {any(c.get('attention_prob_nonzero_above_diagonal') for c in res['cone'])} | "
              f"conservation sum {res['sum_components'] - res['margin']:.1e} head split {res['head_split_max_err']:.1e} head-to-layer {res['head_to_layer_max_err']:.1e} | triples {n} blank {blank} | {res['seconds_total']} s", flush=True)


if __name__ == "__main__":
    main()
