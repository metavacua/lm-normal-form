# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Cell 02, element-wise: one weight, one fact (France). For the three weight kinds whose effect on the margin is closed
form, the margin is evaluated at every finite bf16 value of the weight; for the others on a 513-point grid by full
forward passes. Per weight: the feasible set for margin >= EPS, whether the current value lies in it, the nearest
feasible value, and the effect of that single substitution on all 189 facts.
usage: element.py MODEL_DIR OUT_JSON"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core  # noqa: E402
from verify import capture  # noqa: E402

BF = (np.arange(65536, dtype=np.uint32) << 16).view(np.float32)
FIN = np.unique(BF[np.isfinite(BF)])                 # all finite bf16 values as float32, ascending (about 65,000)


def rows_margin(zp, t):
    zt = zp[:, t].clone(); zp[:, t] = float("-inf")
    return zt - zp.max(1).values


def intervals(vals, ok):
    out, start = [], None
    for k in range(len(vals)):
        if ok[k] and start is None:
            start = k
        if (not ok[k]) and start is not None:
            out.append([float(vals[start]), float(vals[k - 1])]); start = None
    if start is not None:
        out.append([float(vals[start]), float(vals[-1])])
    return out


class Exact:
    """The closed-form margin as a function of one weight, given the cached last-position quantities."""

    def __init__(self, model, c, z, t, j, prompt_tokens):
        self.E = model.model.embed_tokens.weight.data; self.g = model.model.norm.weight.data; self.xL = c["xL"]; self.a = c["a"]
        self.eps_n = model.config.rms_norm_eps; self.r = torch.sqrt((self.xL ** 2).mean() + self.eps_n); self.n = self.xL / self.r
        self.z, self.t, self.j, self.prompt = z, t, j, prompt_tokens
        self.Wd = model.model.layers[-1].mlp.down_proj.weight.data

    def final_norm(self, i, vals):
        cvec = self.E[:, i] * self.n[i]
        out = []
        for ch in torch.split(vals, 512):
            out.append(rows_margin(self.z[None, :] + (ch - self.g[i])[:, None] * cvec[None, :], self.t))
        return torch.cat(out)

    def head_row(self, v, i, vals):
        coef = self.g[i] * self.n[i]; zv = self.z[v] + (vals - self.E[v, i]) * coef
        if v == self.t:
            o = self.z.clone(); o[self.t] = float("-inf"); return zv - o.max()
        o = self.z.clone(); o[self.t] = float("-inf"); o[v] = float("-inf")
        return self.z[self.t] - torch.maximum(zv, o.max().expand_as(zv))

    def last_down(self, i, jj, vals):
        out = []
        for ch in torch.split(vals, 128):
            xp = self.xL[None, :].repeat(len(ch), 1); xp[:, i] += (ch - self.Wd[i, jj]) * self.a[jj]
            npr = xp / torch.sqrt((xp ** 2).mean(1, keepdim=True) + self.eps_n)
            out.append(rows_margin((npr * self.g[None, :]) @ self.E.T, self.t))
        return torch.cat(out)


def main():
    model_dir, out = sys.argv[1:3]
    clock = core.Clock()
    model, tok = core.load(model_dir)
    atoms_ = core.atoms(); targs = core.target_ids(tok, atoms_)
    X = "France"; fi = [x for x, _ in atoms_].index(X); t = targs[fi]
    ms0, _ = core.all_margins(model, tok, atoms_, targs); m0 = float(ms0[fi])
    c, z = capture(model, tok, X, t)
    o = z.clone(); o[t] = float("-inf"); j = int(o.argmax())
    prompt_tokens = set(core.prompt_ids(tok, X)[0].tolist())
    ex = Exact(model, c, z, t, j, prompt_tokens)
    core.margin_and_grad(model, tok, X, t)
    ps = dict(core.params(model)); L = model.config.num_hidden_layers
    gnorm = ps["model.norm.weight"].grad.clone(); gemb = ps["model.embed_tokens.weight"].grad.clone(); gdown = ps[f"model.layers.{L-1}.mlp.down_proj.weight"].grad.clone()
    vals = torch.tensor(FIN)
    picks = []
    for f in torch.topk(gnorm.abs(), 10).indices.tolist():
        picks.append(("final_norm_gain", "model.norm.weight", f))
    emb_rows = torch.stack([gemb[t], gemb[j]]).abs().view(-1); d_ = gemb.shape[1]
    for f in torch.topk(emb_rows, 10).indices.tolist():
        v = t if f < d_ else j; picks.append(("head_row", "model.embed_tokens.weight", v * d_ + (f % d_)))
    for f in torch.topk(gdown.abs().view(-1), 10).indices.tolist():
        picks.append(("last_layer_down_proj", f"model.layers.{L-1}.mlp.down_proj.weight", f))
    res = {"france_margin_start": m0, "competitor": j, "target": t, "eps": core.EPS, "exact": [], "grid": [], "finite_bf16_values": int(len(FIN))}
    rng = np.random.default_rng(33)

    def collateral(p, f, val):
        old = float(p.data.view(-1)[f]); p.data.view(-1)[f] = val
        ms, _ = core.all_margins(model, tok, atoms_, targs); fm = float(ms[fi])
        p.data.view(-1)[f] = old
        return {"count": int((ms > 0).sum()), "france_margin_full_forward": fm, "T12_correct": [x for (x, _), m in zip(atoms_, ms) if x in core.T12 and m > 0]}

    for kind, pn, f in picks:
        p = ps[pn]; shp = list(p.shape); w = float(p.data.view(-1)[f])
        if kind == "final_norm_gain":
            i = f; mv = ex.final_norm(i, vals)
        elif kind == "head_row":
            v, i = divmod(f, d_)
            if v in prompt_tokens:
                continue
            mv = ex.head_row(v, i, vals)
        else:
            i, jj = divmod(f, shp[1]); mv = ex.last_down(i, jj, vals)
        mv = mv.numpy(); ok = mv >= core.EPS; ok0 = mv >= 0.0
        # instrument checks (T2): the enumeration at the current value, and at 3 other values against full forwards
        k_cur = int(np.argmin(np.abs(FIN - w))); chk = {"at_current_abs_diff": abs(float(mv[k_cur]) - m0), "other_values": []}
        for k in rng.choice(len(FIN), 3, replace=False):
            old = float(p.data.view(-1)[f]); p.data.view(-1)[f] = float(FIN[k])
            ms_, _ = core.all_margins(model, tok, [atoms_[fi]], [t]); p.data.view(-1)[f] = old
            chk["other_values"].append({"value": float(FIN[k]), "exact": float(mv[k]), "full_forward": float(ms_[0]), "abs_diff": abs(float(mv[k]) - float(ms_[0]))})
        feas_idx = np.nonzero(ok)[0]
        rec = {"kind": kind, "tensor": pn, "shape": shp, "flat": f, "row": (f // shp[1] if len(shp) == 2 else None), "col": (f % shp[1] if len(shp) == 2 else f), "value": w,
               "gradient": float(p.grad.view(-1)[f]), "feasible_count_eps": int(ok.sum()), "feasible_count_zero": int(ok0.sum()), "feasible_intervals_eps": intervals(FIN, ok)[:6],
               "current_feasible": bool(ok[k_cur]), "checks": chk}
        if len(feas_idx):
            kn = int(feas_idx[np.argmin(np.abs(FIN[feas_idx] - w))]); rec["nearest_feasible"] = float(FIN[kn]); rec["delta"] = float(FIN[kn]) - w; rec["predicted_margin"] = float(mv[kn])
            rec["collateral"] = collateral(p, f, float(FIN[kn]))
        res["exact"].append(rec)
        print(f"exact {kind:22s} {pn.split('.')[-2] if 'layers' in pn else pn} f={f} w={w:+.4f} feasible(eps) {rec['feasible_count_eps']:6d} current_ok={rec['current_feasible']} "
              f"nearest={rec.get('nearest_feasible')} count_after={rec.get('collateral', {}).get('count')} chk={chk['at_current_abs_diff']:.1e}", flush=True)
    # the grid group: the 20 largest-|g| coordinates of any other kind
    cands = []
    for ti, (pn, p) in enumerate(core.params(model)):
        if pn in ("model.norm.weight", "model.embed_tokens.weight", f"model.layers.{L-1}.mlp.down_proj.weight"):
            continue
        v, i = torch.topk(p.grad.abs().view(-1), 20); cands.append((v, torch.full_like(i, ti), i))
    V = torch.cat([a[0] for a in cands]); T = torch.cat([a[1] for a in cands]); I = torch.cat([a[2] for a in cands])
    psl = core.params(model)
    for q in torch.topk(V, 20).indices.tolist():
        pn, p = psl[int(T[q])]; f = int(I[q]); shp = list(p.shape); w = float(p.data.view(-1)[f]); s = max(abs(w), 0.02)
        grid = np.unique(torch.tensor(w + s * np.linspace(-4, 4, 513), dtype=torch.float32).to(torch.bfloat16).to(torch.float32).numpy())
        mv = []
        with torch.no_grad():
            for val in grid:
                p.data.view(-1)[f] = float(val); zz = model(input_ids=core.prompt_ids(tok, X), use_cache=False).logits[0, -1]; mv.append(float(core.margin_of(zz, t)[0]))
            p.data.view(-1)[f] = w
        mv = np.array(mv); ok = mv >= core.EPS; k_cur = int(np.argmin(np.abs(grid - w)))
        rec = {"kind": "other_grid", "tensor": pn, "shape": shp, "flat": f, "row": (f // shp[1] if len(shp) == 2 else None), "col": (f % shp[1] if len(shp) == 2 else f), "value": w,
               "gradient": float(p.grad.view(-1)[f]), "grid_points": int(len(grid)), "feasible_count_eps": int(ok.sum()), "feasible_intervals_eps": intervals(grid, ok)[:6], "current_feasible": bool(ok[k_cur])}
        if ok.any():
            fe = np.nonzero(ok)[0]; kn = int(fe[np.argmin(np.abs(grid[fe] - w))]); rec["nearest_feasible"] = float(grid[kn]); rec["delta"] = float(grid[kn]) - w; rec["predicted_margin"] = float(mv[kn])
            rec["collateral"] = collateral(p, f, float(grid[kn]))
        res["grid"].append(rec)
        print(f"grid  {pn} f={f} w={w:+.4f} feasible(eps) {rec['feasible_count_eps']:4d}/{len(grid)} nearest={rec.get('nearest_feasible')} count_after={rec.get('collateral', {}).get('count')}", flush=True)
    res["seconds"] = clock.s()
    core.dump(out, res)


if __name__ == "__main__":
    main()
