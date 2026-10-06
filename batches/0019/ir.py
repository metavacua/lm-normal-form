# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
# The architecture of batch 0015's model.py, reified: from an Arch (and three flags) the dataflow of one forward pass as a set of relations, which states.dl, a Datalog program,
# reads. Nothing is said here about which transformations of the weights are symmetries: only what exists (the index spaces, the tensors and their axes) and what each operation
# requires of the index spaces it touches (the roles). The correspondence with the numbers is checked, not assumed: measure.py runs the same architecture and compares the shape of
# every state with the axes declared here.
#   spaces      an index space is a set of coordinates that a change of basis acts on: the hidden coordinates (res), the head dimension of the queries and keys of a layer (qk), of
#               the values (ov), the units of the feed-forward block (unit for the up branch and the product, ugate for the gate branch), the key/value groups (grp) and the query
#               heads inside a group (hin) of a layer, the output of a norm weight (na, nm, nf; the hidden coordinates again, after the weight), the tokens (tok, tok2) and the
#               vocabulary (vocab), on which no basis change acts. Each has a size and a number of independent copies (the groups).
#   roles       what an operation requires of a space it touches: lin (it enters a linear map, so any invertible change is compensated in the map), bilinear (a pairing of two
#               spaces by q . k), rmsnorm / rmsnorm0 (the root mean square: orthogonal changes, and with epsilon = 0 a scale as well), rope (the rotary embedding: the commutant of
#               the rotations of the planes), nl (an elementwise nonlinearity: permutations), mono (an elementwise product with a linear branch: scalings and permutations), replica (the
#               copies of a computation indexed by this space: permutations), triv (positions and vocabulary: nothing), and the diagonal link of a norm weight.
# folded: the norm weights are multiplied into the matrices that read them (the final one only if the head is not the embedding), so that the spaces na, nm, nf do not exist.
import os
from dataclasses import dataclass, field


@dataclass
class Prog:
    arch: object
    spaces: list = field(default_factory=list)      # (name, size, copies, kind)
    tensors: list = field(default_factory=list)     # (name, role, stype, layer)
    axes: list = field(default_factory=list)        # (tensor, position, space)
    cons: list = field(default_factory=list)        # (op, space, role)
    diag: list = field(default_factory=list)        # (op, space in, space out)
    ops: list = field(default_factory=list)         # (op, kind)
    flags: dict = field(default_factory=dict)


def build(a, eps_zero=False, logprobs=False, fold=False):
    p = Prog(a)
    p.flags = {"eps_zero": int(eps_zero), "tied": int(a.tied), "logprobs": int(logprobs), "rope": int(a.pos == "rope"), "fold": int(fold)}
    rms = "rmsnorm0" if eps_zero else "rmsnorm"

    def S(name, size, copies, kind):
        p.spaces.append((name, size, copies, kind))
        return name

    def T(name, role, stype, layer, axes):
        p.tensors.append((name, role, stype, layer))
        for i, s in enumerate(axes):
            p.axes.append((name, i, s))
        return name

    def O(op, kind, cons=(), diag=None):
        p.ops.append((op, kind))
        for s, r in cons:
            p.cons.append((op, s, r))
        if diag:
            p.diag.append((op,) + diag)

    tok, tok2, vocab = S("tok", 0, 1, "tok"), S("tok2", 0, 1, "tok"), S("vocab", a.vocab, 1, "vocab")
    res = S("res", a.d, 1, "res")
    nf = res
    final_foldable = not a.tied
    if not fold or not final_foldable:
        nf = res if a.tied else S("nf", a.d, 1, "normed")
    T("E", "param", "E", -1, [vocab, res])
    O("embed", "lookup", [(vocab, "triv"), (res, "lin")])
    T("h_emb", "state", "emb", -1, [tok, res])
    for l in range(a.n_layers):
        qk = S(f"qk{l}", a.hd, a.n_kv, "qk")
        ov = S(f"ov{l}", a.hd, a.n_kv, "ov")
        unit, ugate = S(f"unit{l}", a.d_ff, 1, "unit"), S(f"ugate{l}", a.d_ff, 1, "ugate")
        grp, hin = S(f"grp{l}", a.n_kv, 1, "grp"), S(f"hin{l}", a.rep, a.n_kv, "hin")
        rep_cons = [(grp, "replica"), (hin, "replica")]
        # attention
        T(f"xhat_a{l}", "state", "xhat_a", l, [tok, res])
        O(f"norm_a{l}", "rms_norm", [(res, rms)])
        X = res
        if not fold:
            X = S(f"na{l}", a.d, 1, "normed")
            T(f"gam_a{l}", "param", "gam", l, [X])
            T(f"xt_a{l}", "state", "xt_a", l, [tok, X])
            O(f"scale_a{l}", "diag_mul", diag=(res, X))
        T(f"Wq{l}", "param", "Wq", l, [grp, hin, qk, X])
        T(f"Wk{l}", "param", "Wk", l, [grp, qk, X])
        T(f"Wv{l}", "param", "Wv", l, [grp, ov, X])
        T(f"Wo{l}", "param", "Wo", l, [res, grp, hin, ov])
        O(f"proj_q{l}", "matmul", [(X, "lin"), (qk, "lin")] + rep_cons)
        O(f"proj_k{l}", "matmul", [(X, "lin"), (qk, "lin"), (grp, "replica")])
        O(f"proj_v{l}", "matmul", [(X, "lin"), (ov, "lin"), (grp, "replica")])
        if a.pos == "rope":
            O(f"rope_q{l}", "rope", [(qk, "rope")])
            O(f"rope_k{l}", "rope", [(qk, "rope")])
        T(f"q{l}", "state", "q", l, [tok, grp, hin, qk])
        T(f"k{l}", "state", "k", l, [tok, grp, qk])
        T(f"v{l}", "state", "v", l, [tok, grp, ov])
        O(f"score{l}", "qk_contract", [(qk, "bilinear"), (tok, "triv"), (tok2, "triv")] + rep_cons)
        T(f"score{l}", "state", "score", l, [grp, hin, tok, tok2])
        O(f"softmax{l}", "softmax", [(tok2, "triv")] + rep_cons)
        T(f"pat{l}", "state", "pat", l, [grp, hin, tok, tok2])
        O(f"attend{l}", "attend", [(ov, "lin"), (tok2, "triv")] + rep_cons)
        T(f"oh{l}", "state", "oh", l, [tok, grp, hin, ov])
        O(f"proj_o{l}", "matmul", [(ov, "lin"), (res, "lin")] + rep_cons)
        T(f"ao{l}", "state", "ao", l, [tok, res])
        O(f"add_a{l}", "add")
        T(f"hm{l}", "state", "hm", l, [tok, res])
        # feed-forward block
        T(f"xhat_m{l}", "state", "xhat_m", l, [tok, res])
        O(f"norm_m{l}", "rms_norm", [(res, rms)])
        Xm = res
        if not fold:
            Xm = S(f"nm{l}", a.d, 1, "normed")
            T(f"gam_m{l}", "param", "gam", l, [Xm])
            T(f"xt_m{l}", "state", "xt_m", l, [tok, Xm])
            O(f"scale_m{l}", "diag_mul", diag=(res, Xm))
        T(f"Wg{l}", "param", "Wg", l, [ugate, Xm])
        T(f"Wu{l}", "param", "Wu", l, [unit, Xm])
        T(f"Wd{l}", "param", "Wd", l, [res, unit])
        O(f"proj_g{l}", "matmul", [(Xm, "lin"), (ugate, "lin")])
        O(f"proj_u{l}", "matmul", [(Xm, "lin"), (unit, "lin")])
        T(f"gate{l}", "state", "gate", l, [tok, ugate])
        O(f"silu{l}", "elementwise_nl", [(ugate, "nl")])
        T(f"gact{l}", "state", "gact", l, [tok, ugate])
        T(f"up{l}", "state", "up", l, [tok, unit])
        O(f"gprod{l}", "gated_product", [(unit, "mono"), (ugate, "nl")])
        T(f"m{l}", "state", "m", l, [tok, unit])
        O(f"proj_d{l}", "matmul", [(unit, "lin"), (res, "lin")])
        T(f"mo{l}", "state", "mo", l, [tok, res])
        O(f"add_m{l}", "add")
        T(f"h{l}", "state", "h", l, [tok, res])
    # the end
    T("xhat_f", "state", "xhat_f", a.n_layers, [tok, res])
    O("norm_f", "rms_norm", [(res, rms)])
    if not fold or not final_foldable:
        T("gam_f", "param", "gam", a.n_layers, [nf])
        T("xt_f", "state", "xt_f", a.n_layers, [tok, nf])
        O("scale_f", "diag_mul", diag=(res, nf))
    if not a.tied:
        T("U", "param", "U", -1, [vocab, nf])
    O("head", "matmul", [(nf, "lin"), (vocab, "triv")])
    T("logits", "state", "logits", a.n_layers, [tok, vocab])
    return p


FACT_FILES = ("space", "tensor", "axis", "cons", "diag", "op", "flag")


def write_facts(p, d):
    """Souffle input files, one tab-separated file per relation."""
    os.makedirs(d, exist_ok=True)
    rows = {"space": p.spaces, "tensor": p.tensors, "axis": p.axes, "cons": p.cons, "diag": p.diag, "op": p.ops, "flag": list(p.flags.items())}
    for name, rs in rows.items():
        with open(os.path.join(d, name + ".facts"), "w") as f:
            for r in rs:
                f.write("\t".join(str(x) for x in r) + "\n")


def state_shapes(p):
    """state type -> list of (layer, axes as (space, size, copies)) for the check against the numbers."""
    sizes = {s: (n, c) for s, n, c, _ in p.spaces}
    ax = {}
    for t, i, s in p.axes:
        ax.setdefault(t, []).append((i, s))
    out = {}
    for t, role, stype, layer in p.tensors:
        if role == "state":
            out.setdefault(stype, []).append((layer, [(s, *sizes[s]) for _, s in sorted(ax[t])]))
    return out


def hf_name(t):
    """The name that Hugging Face's Llama gives to the parameter tensor `t` of the program, or None for a tensor that the program does not have a file name for."""
    fixed = {"E": "model.embed_tokens.weight", "U": "lm_head.weight", "gam_f": "model.norm.weight"}
    if t in fixed:
        return fixed[t]
    for prefix, tail in (("Wq", "self_attn.q_proj.weight"), ("Wk", "self_attn.k_proj.weight"), ("Wv", "self_attn.v_proj.weight"), ("Wo", "self_attn.o_proj.weight"),
                         ("Wg", "mlp.gate_proj.weight"), ("Wu", "mlp.up_proj.weight"), ("Wd", "mlp.down_proj.weight"), ("gam_a", "input_layernorm.weight"), ("gam_m", "post_attention_layernorm.weight")):
        if t.startswith(prefix) and t[len(prefix):].isdigit():
            return f"model.layers.{t[len(prefix):]}.{tail}"
    return None


def param_counts(p):
    """file name -> number of elements, for the parameter tensors of the (unfolded) program: the product of the sizes of the axes."""
    sizes = {s: n for s, n, c, _ in p.spaces}
    ax = {}
    for t, i, s in p.axes:
        ax.setdefault(t, []).append(s)
    out = {}
    for t, role, stype, layer in p.tensors:
        if role == "param":
            n = 1
            for s in ax[t]:
                n *= sizes[s]
            out[hf_name(t)] = n
    return out
