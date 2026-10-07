# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
# Batch 0023, helpers for the experiments on real language-model checkpoints in Hugging Face's Transformers (PyTorch, float64 on the CPU): loading a checkpoint (or a small random model of the same
# architecture, for the smoke runs that test this code before it is pointed at a checkpoint), the two places where the Llama-family code computes in float32 although the model is float64 (the
# norm, and the tables of the rotary embedding) replaced by float64 versions, capture of the stream after each layer and of the cache of keys, and the transformations of the weights that the
# experiments apply. A transformation is a function from a model to a new model (a deep copy); none changes its argument. Matrices are (out, in) as in Hugging Face. Nothing is silently skipped:
# an architecture or a library version that does not have what a function needs makes it raise.
import copy, importlib
import numpy as np
import torch

DT = torch.float64


class Pos:
    """The dilation of positions of the patched rotary embedding: the angle of position t is scale * t * theta. 1.0 except in the experiment that dilates."""
    scale = 1.0
    rope_calls = 0
    norm_calls = 0


def rope_theta(cfg):
    for k in ("rope_theta", "rotary_emb_base"):
        if cfg.get(k):
            return float(cfg[k])
    rp = cfg.get("rope_parameters") or cfg.get("rope_scaling") or {}
    if isinstance(rp, dict) and rp.get("rope_theta"):
        return float(rp["rope_theta"])
    return 10000.0


_MODULES = {"llama": ("transformers.models.llama.modeling_llama", "LlamaRMSNorm", "LlamaRotaryEmbedding"),
            "qwen3": ("transformers.models.qwen3.modeling_qwen3", "Qwen3RMSNorm", "Qwen3RotaryEmbedding"),
            "gpt_neox": ("transformers.models.gpt_neox.modeling_gpt_neox", None, "GPTNeoXRotaryEmbedding"),
            "gpt2": (None, None, None)}


def patch_float64(model_type, cfg):
    """Replace the float32 islands of the Llama-family code by float64 ones (the RMSNorm casts to float32 inside; the rotary tables are computed from a float32 inverse frequency). Returns the
    function that restores the originals."""
    modname, norm_cls, rope_cls = _MODULES[model_type]
    if modname is None:
        return lambda: None
    M = importlib.import_module(modname)
    theta = rope_theta(cfg)
    saved = []

    def norm64(self, h):
        Pos.norm_calls += 1
        return self.weight * (h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.variance_epsilon))

    def rope64(self, x, position_ids):
        Pos.rope_calls += 1
        dim = self.inv_freq.shape[0] * 2
        inv = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float64) / dim))
        freqs = position_ids[:, :, None].to(torch.float64) * Pos.scale * inv[None, None, :]
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(x.dtype), emb.sin().to(x.dtype)

    if norm_cls:
        c = getattr(M, norm_cls)
        saved.append((c, c.forward))
        c.forward = norm64
    c = getattr(M, rope_cls)
    saved.append((c, c.forward))
    c.forward = rope64

    def restore():
        for cls, f in saved:
            cls.forward = f
    return restore


def patch_used(model_type):
    """True if the float64 versions of the rotary tables (and of the norm, for the families that have an RMSNorm) have been called since the patch was installed: the claims that depend on them are
    "not run" otherwise, because a float32 table would be measured in place of the theory."""
    if model_type == "gpt2":
        return True
    return Pos.rope_calls > 0 and (Pos.norm_calls > 0 or model_type == "gpt_neox")


def random_small_config(model_type, vocab=64):
    from transformers import GPT2Config, GPTNeoXConfig, LlamaConfig, Qwen3Config
    if model_type == "llama":
        return LlamaConfig(vocab_size=vocab, hidden_size=16, intermediate_size=32, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=4, max_position_embeddings=512,
                           rms_norm_eps=1e-6, rope_theta=10000.0, tie_word_embeddings=True)
    if model_type == "qwen3":
        return Qwen3Config(vocab_size=vocab, hidden_size=16, intermediate_size=32, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=4, max_position_embeddings=512,
                           rms_norm_eps=1e-6, rope_theta=1000000.0, tie_word_embeddings=True)
    if model_type == "gpt2":
        return GPT2Config(vocab_size=vocab, n_embd=16, n_layer=2, n_head=4, n_positions=512, n_inner=64)
    if model_type == "gpt_neox":
        return GPTNeoXConfig(vocab_size=vocab, hidden_size=16, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4, rotary_pct=0.5, max_position_embeddings=512, use_parallel_residual=True)
    raise ValueError(model_type)


def randomize(m, seed):
    """Generic parameters: every tensor redrawn (matrices N(0, 0.5^2 / fan_in), norm weights 1 + 0.3 N(0,1), biases 0.3 N(0,1)); tied tensors stay tied (named_parameters lists them once)."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in m.named_parameters():
            r = torch.randn(p.shape, generator=g, dtype=DT)
            if p.dim() >= 2:
                p.copy_(r * 0.5 / np.sqrt(p.shape[-1] if p.dim() == 2 else p.shape[0]))
            elif "norm" in name or "ln" in name.split(".")[-2]:
                p.copy_(1.0 + 0.3 * r) if name.endswith("weight") else p.copy_(0.3 * r)
            else:
                p.copy_(0.3 * r)


def load_model(model_type, repo=None, rev=None, random_small=False, seed=0):
    """(model in float64, tokenizer or None, config dict, restore function of the float64 patches)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if random_small:
        tok = AutoTokenizer.from_pretrained(repo, revision=rev) if repo else None          # with a repository, only its tokenizer is read (its weights are not): the paths that read text can then be run on a random model
        cfg = random_small_config(model_type, len(tok) if tok is not None else 64)
        torch.manual_seed(seed)
        m = AutoModelForCausalLM.from_config(cfg, attn_implementation="sdpa").to(DT)
        randomize(m, seed)
    else:
        m = AutoModelForCausalLM.from_pretrained(repo, revision=rev, dtype=DT, attn_implementation="sdpa")
        tok = AutoTokenizer.from_pretrained(repo, revision=rev)
    m.eval()
    cfgd = m.config.to_dict()
    return m, tok, cfgd, patch_float64(model_type, cfgd)


def rho(a, b):
    """The relative difference of two arrays: the Frobenius norm of the difference over the norm of the second."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300))


def run(m, ids, positions=None, pos_scale=1.0, cache=False, streams=False, model_type="llama"):
    """Logits (T, V) as numpy; and, if asked, the stream after the embedding and after each layer (list of (T, d)) and the keys of the cache (list of (n_kv, T, hd), after the rotation)."""
    kw = {"input_ids": ids, "attention_mask": torch.ones_like(ids), "use_cache": bool(cache)}
    if positions is not None:
        kw["position_ids"] = positions
    Pos.scale = pos_scale
    caught, hooks = [], []
    if streams:
        base, layers = (m.model, m.model.layers) if model_type in ("llama", "qwen3") else (m.gpt_neox, m.gpt_neox.layers)
        emb = base.embed_tokens if model_type in ("llama", "qwen3") else base.embed_in
        hooks.append(emb.register_forward_hook(lambda mod, i, o: caught.append(o[0].detach().clone())))
        for layer in layers:
            hooks.append(layer.register_forward_hook(lambda mod, i, o: caught.append((o[0] if isinstance(o, tuple) else o)[0].detach().clone())))
    try:
        with torch.no_grad():
            out = m(**kw)
    finally:
        for h in hooks:
            h.remove()
        Pos.scale = 1.0
    res = {"logits": out.logits[0].detach().cpu().numpy()}
    if streams:
        res["streams"] = [c.cpu().numpy() for c in caught]
    if cache:
        pkv = out.past_key_values
        n = m.config.num_hidden_layers
        if hasattr(pkv, "layers"):
            res["keys"] = [pkv.layers[i].keys[0].cpu().numpy() for i in range(n)]
        elif hasattr(pkv, "key_cache"):
            res["keys"] = [pkv.key_cache[i][0].cpu().numpy() for i in range(n)]
        else:
            res["keys"] = [pkv[i][0][0].cpu().numpy() for i in range(n)]
    return res


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# the Llama family: model.embed_tokens, model.layers[i].{self_attn.{q,k,v,o}_proj, mlp.{gate,up,down}_proj, input_layernorm, post_attention_layernorm}, model.norm, lm_head
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def _target(m, inplace):
    """The model that a transformation modifies: the argument itself if `inplace`, else a deep copy. The in-place form exists because a float64 copy of a 0.6-billion-parameter checkpoint is 4.8 GB."""
    return m if inplace else copy.deepcopy(m)


def dims(m):
    c = m.config
    d, nh = c.hidden_size, c.num_attention_heads
    hd = getattr(c, "head_dim", None) or d // nh
    return d, nh, c.num_key_value_heads, hd, c.intermediate_size


def is_tied(m):
    return m.lm_head.weight is m.get_input_embeddings().weight


def fold_untie(m):
    """The same function with every norm weight equal to 1 (each multiplied into the columns of the matrices that read its output) and the head a matrix of its own."""
    m = copy.deepcopy(m)
    d, nh, nkv, hd, dff = dims(m)
    with torch.no_grad():
        for layer in m.model.layers:
            g = layer.input_layernorm.weight.clone()
            for p in (layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj):
                p.weight.mul_(g[None, :])
            layer.input_layernorm.weight.fill_(1.0)
            g = layer.post_attention_layernorm.weight.clone()
            for p in (layer.mlp.gate_proj, layer.mlp.up_proj):
                p.weight.mul_(g[None, :])
            layer.post_attention_layernorm.weight.fill_(1.0)
        head = m.lm_head.weight.detach().clone() * m.model.norm.weight[None, :]
        m.model.norm.weight.fill_(1.0)
        new = torch.nn.Linear(d, head.shape[0], bias=False, dtype=DT)
        new.weight = torch.nn.Parameter(head)
        m.lm_head = new
        m.config.tie_word_embeddings = False
    return m


def rotate(m, Q, inplace=False):
    """x' = Q x on the stream: the embedding rows e -> e Q^T, every matrix that reads the stream W -> W Q^T, every matrix that writes it W -> Q W. Exact only if every norm weight is 1 (fold_untie first)."""
    m = _target(m, inplace)
    Q = Q.to(DT)
    tied = is_tied(m)
    with torch.no_grad():
        E = m.model.embed_tokens.weight
        E.copy_(E @ Q.T)
        for layer in m.model.layers:
            for p in (layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj, layer.mlp.gate_proj, layer.mlp.up_proj):
                p.weight.copy_(p.weight @ Q.T)
            for p in (layer.self_attn.o_proj, layer.mlp.down_proj):
                p.weight.copy_(Q @ p.weight)
        if not tied:
            m.lm_head.weight.copy_(m.lm_head.weight @ Q.T)
    return m


def signed_permutation(m, perm, signs, inplace=False):
    """H1: x'_j = signs[j] * x_{perm[j]}; the norm weights are permuted along (gamma'_j = gamma_{perm[j]}), unsigned."""
    d = dims(m)[0]
    Q = torch.zeros(d, d, dtype=DT)
    for j in range(d):
        Q[j, perm[j]] = float(signs[j])
    m2 = rotate(m, Q, inplace=inplace)
    with torch.no_grad():
        ab = Q.abs()
        for layer in m2.model.layers:
            layer.input_layernorm.weight.copy_(ab @ layer.input_layernorm.weight)
            layer.post_attention_layernorm.weight.copy_(ab @ layer.post_attention_layernorm.weight)
        m2.model.norm.weight.copy_(ab @ m2.model.norm.weight)
    return m2


def random_orthogonal(d, seed):
    q, r = np.linalg.qr(np.random.RandomState(seed).standard_normal((d, d)))
    return torch.tensor(q * np.sign(np.diag(r))[None, :], dtype=DT)


def unit_gauge(m, perm, scale, inplace=False):
    """M1: per layer, the units are permuted (gate rows, up rows, down columns by `perm`), the up row of unit u scaled by scale[u] and its down column by 1 / scale[u]."""
    m = _target(m, inplace)
    p = torch.tensor(perm, dtype=torch.long)
    c = torch.tensor(scale, dtype=DT)
    with torch.no_grad():
        for layer in m.model.layers:
            wg, wu, wd = layer.mlp.gate_proj.weight, layer.mlp.up_proj.weight, layer.mlp.down_proj.weight
            wg2, wu2, wd2 = wg[p].clone(), (wu * c[:, None])[p].clone(), (wd / c[None, :])[:, p].clone()
            wg.copy_(wg2), wu.copy_(wu2), wd.copy_(wd2)
    return m


def value_output_gauge(m, As, inplace=False):
    """A1: per layer and key/value group an invertible hd x hd matrix A: the value rows of the group -> A (rows), the output columns of its heads -> (columns) A^-1. As[g] numpy (hd, hd)."""
    m = _target(m, inplace)
    d, nh, nkv, hd, dff = dims(m)
    rep = nh // nkv
    with torch.no_grad():
        for layer in m.model.layers:
            for g in range(nkv):
                A = torch.tensor(As[g], dtype=DT)
                Ai = torch.linalg.inv(A)
                v = layer.self_attn.v_proj.weight
                v[g * hd:(g + 1) * hd] = A @ v[g * hd:(g + 1) * hd].clone()
                o = layer.self_attn.o_proj.weight
                for h in range(g * rep, (g + 1) * rep):
                    o[:, h * hd:(h + 1) * hd] = o[:, h * hd:(h + 1) * hd].clone() @ Ai
    return m


def head_swap(m, inplace=False):
    """A6: in every layer, the first two query heads of the first key/value group are exchanged (their rows of q_proj and columns of o_proj)."""
    m = _target(m, inplace)
    d, nh, nkv, hd, dff = dims(m)
    if nh // nkv < 2:
        raise ValueError("a group of one head has nothing to exchange")
    with torch.no_grad():
        for layer in m.model.layers:
            q, o = layer.self_attn.q_proj.weight, layer.self_attn.o_proj.weight
            a, b = q[0:hd].clone(), q[hd:2 * hd].clone()
            q[0:hd], q[hd:2 * hd] = b, a
            a, b = o[:, 0:hd].clone(), o[:, hd:2 * hd].clone()
            o[:, 0:hd], o[:, hd:2 * hd] = b, a
    return m


def plane_rows(g_or_h, p, hd):
    h2 = hd // 2
    return g_or_h * hd + p, g_or_h * hd + p + h2


def phase_gauge(m, phi, inplace=False):
    """A3, the phase part: for every layer, key/value group g and rotary plane p, the plane (rows p and p + hd/2) of the keys of the group and of the queries of its heads is rotated by phi[g][p]
    (a 2 x 2 rotation applied to the pair of rows). With phi[g][p] = c * theta_p this is a shift of every position by c."""
    m = _target(m, inplace)
    d, nh, nkv, hd, dff = dims(m)
    rep, h2 = nh // nkv, hd // 2
    with torch.no_grad():
        for layer in m.model.layers:
            for g in range(nkv):
                for p in range(h2):
                    co, si = float(np.cos(phi[g][p])), float(np.sin(phi[g][p]))
                    targets = [(layer.self_attn.k_proj.weight, g)] + [(layer.self_attn.q_proj.weight, h) for h in range(g * rep, (g + 1) * rep)]
                    for W, blk in targets:
                        r1, r2 = plane_rows(blk, p, hd)
                        a, b = W[r1].clone(), W[r2].clone()
                        W[r1], W[r2] = co * a - si * b, si * a + co * b
    return m


def scalar_pair_gauge(m, s, inplace=False):
    """A3, the scale part: the keys of plane p of group g times s[g][p], the queries of the group's heads in that plane times 1 / s[g][p]."""
    m = _target(m, inplace)
    d, nh, nkv, hd, dff = dims(m)
    rep, h2 = nh // nkv, hd // 2
    with torch.no_grad():
        for layer in m.model.layers:
            for g in range(nkv):
                for p in range(h2):
                    for r in plane_rows(g, p, hd):
                        layer.self_attn.k_proj.weight[r] *= float(s[g][p])
                    for h in range(g * rep, (g + 1) * rep):
                        for r in plane_rows(h, p, hd):
                            layer.self_attn.q_proj.weight[r] *= 1.0 / float(s[g][p])
    return m


def rotary_frequencies(m):
    d, nh, nkv, hd, dff = dims(m)
    return np.array([rope_theta(m.config.to_dict()) ** (-2.0 * p / hd) for p in range(hd // 2)])


def head_scalar(m, c, inplace=False):
    """Qwen3 (QK-norm): the query projection of head h times c[h] > 0 in every layer."""
    m = _target(m, inplace)
    d, nh, nkv, hd, dff = dims(m)
    with torch.no_grad():
        for layer in m.model.layers:
            for h in range(nh):
                layer.self_attn.q_proj.weight[h * hd:(h + 1) * hd] *= float(c[h])
    return m


def qk_gain_move(m, s, inplace=False):
    """Qwen3 (QK-norm): the gain of plane p (both coordinates) divided by s[p] on the query norm and multiplied by s[p] on the key norm, in every layer."""
    m = _target(m, inplace)
    d, nh, nkv, hd, dff = dims(m)
    h2 = hd // 2
    with torch.no_grad():
        for layer in m.model.layers:
            for p in range(h2):
                for r in (p, p + h2):
                    layer.self_attn.q_norm.weight[r] /= float(s[p])
                    layer.self_attn.k_norm.weight[r] *= float(s[p])
    return m


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# GPT-2: transformer.{wte, wpe, h[i].{ln_1, attn.{c_attn, c_proj}, ln_2, mlp.{c_fc, c_proj}}, ln_f}; the Conv1D weights are (in, out); the head is tied to wte
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def gpt2_writer_shift(m, u_attn, u_mlp, b_attn, b_mlp):
    """LayerNorm is blind to a multiple of the all-ones vector: the output projection of the attention (weight (in, out), bias (out)) gets u added to every column and b to every bias entry, and
    the same for the output projection of the MLP. u_*: (in,) tensors for every layer: lists indexed by layer."""
    m = copy.deepcopy(m)
    with torch.no_grad():
        for i, blk in enumerate(m.transformer.h):
            blk.attn.c_proj.weight.add_(u_attn[i][:, None])
            blk.attn.c_proj.bias.add_(float(b_attn[i]))
            blk.mlp.c_proj.weight.add_(u_mlp[i][:, None])
            blk.mlp.c_proj.bias.add_(float(b_mlp[i]))
    return m


def gpt2_reader_shift(m, a_attn, a_mlp):
    """The dual shift on the readers: after ln_1 the input of c_attn is gamma * y + beta with y of mean zero, so adding a_j / gamma_i to W[i, j] adds a_j sum(y) = 0 and a_j sum(beta / gamma), which the
    bias cancels. a_attn[i]: (3 n_embd,), a_mlp[i]: (n_inner,)."""
    m = copy.deepcopy(m)
    with torch.no_grad():
        for i, blk in enumerate(m.transformer.h):
            for ln, conv, a in ((blk.ln_1, blk.attn.c_attn, a_attn[i]), (blk.ln_2, blk.mlp.c_fc, a_mlp[i])):
                gam, bet = ln.weight, ln.bias
                conv.weight.add_(gam.reciprocal()[:, None] * a[None, :])
                conv.bias.sub_(a * float((bet / gam).sum()))
    return m


def gpt2_embedding_shift(m, tokens, c):
    """Adds c[k] to every coordinate of the embedding row of tokens[k] (the head is tied to the embedding)."""
    m = copy.deepcopy(m)
    with torch.no_grad():
        for t, v in zip(tokens, c):
            m.transformer.wte.weight[t] += float(v)
    return m


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# GPT-NeoX (Pythia): gpt_neox.{embed_in, layers[i].{input_layernorm, attention.{query_key_value, dense}, post_attention_layernorm, mlp.{dense_h_to_4h, dense_4h_to_h}}, final_layer_norm}, embed_out;
# every Linear has a bias except embed_out; parallel residual: x + attention(ln1(x)) + mlp(ln2(x))
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def neox_head(m):
    """The output matrix of GPT-NeoX: `embed_out` in earlier versions of Transformers, `lm_head` in later ones."""
    for name in ("embed_out", "lm_head"):
        if hasattr(m, name):
            return getattr(m, name)
    raise AttributeError("GPT-NeoX with neither embed_out nor lm_head")


def neox_writer_shift(m, u_attn, u_mlp, c_attn, c_mlp, u_embed):
    """The same blindness, in the layout (out, in): the attention's dense and the MLP's dense_4h_to_h get u[:, ...] (a column vector times the all-ones row) added to the weight, c to the bias; embed_in gets
    u_embed[v] added to every coordinate of the row of token v."""
    m = copy.deepcopy(m)
    with torch.no_grad():
        for i, layer in enumerate(m.gpt_neox.layers):
            layer.attention.dense.weight.add_(torch.ones(layer.attention.dense.weight.shape[0], 1, dtype=DT) * u_attn[i][None, :])
            layer.attention.dense.bias.add_(float(c_attn[i]))
            layer.mlp.dense_4h_to_h.weight.add_(torch.ones(layer.mlp.dense_4h_to_h.weight.shape[0], 1, dtype=DT) * u_mlp[i][None, :])
            layer.mlp.dense_4h_to_h.bias.add_(float(c_mlp[i]))
        m.gpt_neox.embed_in.weight.add_(u_embed[:, None])
    return m


def neox_fold_rotate(m, Q):
    """Folds every LayerNorm (gamma into the readers' columns, beta into their biases), then x' = Q x on the stream. The final LayerNorm's beta cannot be folded (embed_out has no bias): the constant logit
    offset embed_out beta is returned, to be added to the logits of the transformed model. Returns (model, offset)."""
    m = copy.deepcopy(m)
    Q = Q.to(DT)
    with torch.no_grad():
        for layer in m.gpt_neox.layers:
            for ln, lins in ((layer.input_layernorm, [layer.attention.query_key_value]), (layer.post_attention_layernorm, [layer.mlp.dense_h_to_4h])):
                gam, bet = ln.weight.clone(), ln.bias.clone()
                for lin in lins:
                    lin.bias.add_(lin.weight @ bet)
                    lin.weight.mul_(gam[None, :])
                ln.weight.fill_(1.0)
                ln.bias.zero_()
        gam, bet = m.gpt_neox.final_layer_norm.weight.clone(), m.gpt_neox.final_layer_norm.bias.clone()
        offset = (neox_head(m).weight @ bet).clone()
        neox_head(m).weight.mul_(gam[None, :])
        m.gpt_neox.final_layer_norm.weight.fill_(1.0)
        m.gpt_neox.final_layer_norm.bias.zero_()
        E = m.gpt_neox.embed_in.weight
        E.copy_(E @ Q.T)
        for layer in m.gpt_neox.layers:
            for lin in (layer.attention.query_key_value, layer.mlp.dense_h_to_4h):
                lin.weight.copy_(lin.weight @ Q.T)
            for lin in (layer.attention.dense, layer.mlp.dense_4h_to_h):
                lin.weight.copy_(Q @ lin.weight)
                lin.bias.copy_(Q @ lin.bias)
        neox_head(m).weight.copy_(neox_head(m).weight @ Q.T)
    return m, offset


def orthogonal_fixing_ones(d, seed):
    """An orthogonal Q with Q 1 = 1: a product of Householder reflections in vectors orthogonal to the all-ones vector."""
    rs = np.random.RandomState(seed)
    Q = np.eye(d)
    for _ in range(d):
        v = rs.standard_normal(d)
        v = v - v.mean()
        Q = Q @ (np.eye(d) - 2.0 * np.outer(v, v) / (v @ v))
    return torch.tensor(Q, dtype=DT)


def membership(m, tok):
    """The operational test of this batch for "is a language model": the mean next-token loss (nats) of the checkpoint on one fixed short English text (membership.txt), as a fraction of ln(size of the
    output vocabulary), the loss of a model that knows nothing. A checkpoint whose parameters were fitted to text has a fraction well below 1; a function with random parameters has a fraction
    near 1 (above it, if its logits are large). Registered threshold: at most 0.9 (docs/batches/0023.md)."""
    import math, os
    here = os.path.dirname(os.path.abspath(__file__))
    text = open(os.path.join(here, "membership.txt"), encoding="utf-8").read()
    ids = tok(text, return_tensors="pt")["input_ids"][:, :256]
    with torch.no_grad():
        lg = m(input_ids=ids, attention_mask=torch.ones_like(ids)).logits[0, :-1]
    loss = float(torch.nn.functional.cross_entropy(lg, ids[0, 1:]))
    return {"tokens": int(ids.shape[1]), "mean_loss_nats": loss, "ln_output_vocabulary": math.log(lg.shape[-1]), "fraction_of_ln_vocabulary": loss / math.log(lg.shape[-1])}


def texts():
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    return [l for l in open(os.path.join(here, "..", "0013", "sentences.txt"), encoding="utf-8").read().splitlines() if l.strip()][:6]


def encode(tok, model_type, random_small, vocab, seed=0, length=24):
    """The test sequences: the tokens of six sentences (cut to `length`) for a checkpoint; random token ids for a finite instance."""
    if random_small:
        g = torch.Generator().manual_seed(seed)
        return [torch.randint(0, vocab, (1, length), generator=g) for _ in range(3)]
    out = []
    for t in texts():
        ids = tok(t, return_tensors="pt")["input_ids"][:, :length]
        out.append(ids)
    return out
