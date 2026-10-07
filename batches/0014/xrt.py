# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
# The driver of batch 0014: one checkpoint, its variants (variants.py), one runtime, one compute dtype.
#   xrt.py prepare MODELDIR_ORIG WORK        token ids of the test set, the generation prompts and the perplexity windows -> WORK/inputs.json
#   xrt.py one RUNTIME DTYPE VARIANT ROUND WORKDIR   one measurement in its own process (so that peak memory is the variant's): WORKDIR/res/...json, and
#                                                    the logits as WORKDIR/res/...npy for round 0
#   xrt.py all RUNTIME DTYPE WORKDIR ROUNDS  every variant, ROUNDS times, in an order that rotates each round (so that drift of the machine does not
#                                            fall on one variant), then the comparison -> WORKDIR/summary-RUNTIME-DTYPE.json
# WORKDIR holds variants/<name>/ (variants.py), inputs.json, and two sets of reference logits that `all` makes first if they are missing: ref.npy (PyTorch, float32,
# original: what the runtimes are compared with) and ref64.npy (ref64.py, float64, original: the value of the function, from which every run's rounding error is measured). A runtime adapter is a module rt_<name>.py with version() and run(model_dir, ids, wins, dtype, work, tag, full) -> (result dict, logits
# of the test set, positions pooled, float32). The result dict has: load_s, gen {plain, chat: lists of 32 greedy token ids per prompt}, ppl_nll, ppl_tokens,
# prefill_tps (list), decode_tps (list), peak_rss_mb. Some runtimes cannot give all of it; what they cannot give they leave out. Only round 0 is `full` (logits,
# generation, perplexity); the other rounds measure load time, speed and memory alone.
import importlib, json, os, subprocess, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import texts_and_ids, wikitext_windows, logit_metrics, median, first_divergence, dump

VARIANTS = ["orig", "scale", "perm", "all", "canon", "broken", "heads", "units", "units_blk", "resid", "resid_blk", "nullall"]      # the driver runs those that are present in WORK/variants


def eos_ids(orig_dir, tok):
    """The token ids after which generation stops: the generation config's, else the tokenizer's."""
    gc = os.path.join(orig_dir, "generation_config.json")
    e = json.load(open(gc)).get("eos_token_id") if os.path.exists(gc) else None
    if e is None:
        e = tok.eos_token_id
    return [int(x) for x in (e if isinstance(e, list) else [e])]


def prepare(orig_dir, work):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(orig_dir)
    ids = texts_and_ids(tok, chat=True)
    ids["eos"] = eos_ids(orig_dir, tok)
    wins, text = wikitext_windows(tok)
    dump({"ids": ids, "wins": wins}, os.path.join(work, "inputs.json"))
    seqs = open(os.path.join(HERE, "..", "0010", "prompts.txt"), encoding="utf-8").read().splitlines() + open(os.path.join(HERE, "..", "0013", "sentences.txt"), encoding="utf-8").read().splitlines()
    from common import PLAIN_PROMPTS
    texts = seqs[:24] + PLAIN_PROMPTS + [text[:2000]]
    dump({"texts": texts, "ids": [tok(t)["input_ids"] for t in texts]}, os.path.join(work, "texts.json"))
    open(os.path.join(work, "wiki.txt"), "w", encoding="utf-8").write(text)
    print("inputs:", {k: len(v) for k, v in ids.items()}, "windows", len(wins), "of", len(wins[0]), "eos", ids["eos"])


def load_inputs(work):
    j = json.load(open(os.path.join(work, "inputs.json")))
    return j["ids"], j["wins"]


def one(runtime, dtype, variant, rnd, work):
    rt = importlib.import_module(f"rt_{runtime}")
    ids, wins = load_inputs(work)
    os.makedirs(os.path.join(work, "res"), exist_ok=True)
    stem = os.path.join(work, "res", f"{runtime}-{dtype}-{variant}-r{rnd}")
    t = time.time()
    full = rnd == 0 or rnd >= 100              # round 100 is the original run a second time, whole: the determinism control
    res, logits = rt.run(os.path.join(work, "variants", variant), ids, wins, dtype, work, f"{runtime}-{dtype}-{variant}", full=full)
    res.update({"runtime": runtime, "dtype": dtype, "variant": variant, "round": rnd, "version": rt.version(), "wall_s": time.time() - t})
    dump(res, stem + ".json")
    if logits is not None and (rnd == 0 or rnd >= 100):
        np.save(stem + ".npy", logits)
    print(f"{runtime} {dtype} {variant} r{rnd}: {res.get('wall_s', 0):.0f}s", flush=True)


def run_all(runtime, dtype, work, rounds):
    ref = os.path.join(work, "ref.npy")
    if not os.path.exists(ref):
        subprocess.run([sys.executable, __file__, "one", "pytorch", "float32", "orig", "0", work], check=True)
        os.replace(os.path.join(work, "res", "pytorch-float32-orig-r0.npy"), ref)
    if not os.path.exists(os.path.join(work, "ref64.npy")):
        subprocess.run([sys.executable, os.path.join(HERE, "ref64.py"), "logits", work], check=True)
    present = [v for v in VARIANTS if os.path.isdir(os.path.join(work, "variants", v))]
    for r in range(rounds):
        order = present[r % len(present):] + present[:r % len(present)]
        for v in order:
            subprocess.run([sys.executable, __file__, "one", runtime, dtype, v, str(r), work], check=True)
    subprocess.run([sys.executable, __file__, "one", runtime, dtype, "orig", "100", work], check=True)
    subprocess.run([sys.executable, os.path.join(HERE, "ref64.py"), "margins", runtime, dtype, work], check=True)
    summarize(runtime, dtype, work, rounds, present)


def summarize(runtime, dtype, work, rounds, variants):
    ref = np.load(os.path.join(work, "ref.npy"))
    ref64 = np.load(os.path.join(work, "ref64.npy")) if os.path.exists(os.path.join(work, "ref64.npy")) else None
    rs = lambda v, r: json.load(open(os.path.join(work, "res", f"{runtime}-{dtype}-{v}-r{r}.json")))
    lg = lambda v: np.load(os.path.join(work, "res", f"{runtime}-{dtype}-{v}-r0.npy")) if os.path.exists(os.path.join(work, "res", f"{runtime}-{dtype}-{v}-r0.npy")) else None
    base = lg("orig")
    out = {"runtime": runtime, "dtype": dtype, "model": os.environ.get("LMNF_MODEL", ""), "version": rs("orig", 0)["version"], "rounds": rounds, "variants": {},
           "kv": rs("orig", 0).get("kv"), "reference_float32_vs_float64": logit_metrics(ref, ref64) if ref64 is not None else None}
    mp = os.path.join(work, "res", f"margins-{runtime}-{dtype}.json")
    if os.path.exists(mp):
        out["margins"] = json.load(open(mp))
    again = os.path.join(work, "res", f"{runtime}-{dtype}-orig-r100.npy")
    if os.path.exists(again) and base is not None:
        r0, r100 = rs("orig", 0), json.load(open(os.path.join(work, "res", f"{runtime}-{dtype}-orig-r100.json")))
        out["determinism"] = {"logits_bit_identical": bool(np.array_equal(np.load(again), base)), "ppl_equal": r0.get("ppl_nll") == r100.get("ppl_nll"),
                              "generations_equal": r0.get("gen") == r100.get("gen")}
    for v in variants:
        runs = [rs(v, r) for r in range(rounds)]
        pre, dec = [t for x in runs for t in x.get("prefill_tps", [])], [t for x in runs for t in x.get("decode_tps", [])]
        o = {"load_s": median([x["load_s"] for x in runs if "load_s" in x]), "peak_rss_mb": median([x["peak_rss_mb"] for x in runs if "peak_rss_mb" in x]),
             "prefill_tps": median(pre), "decode_tps": median(dec), "prefill_tps_all": pre, "decode_tps_all": dec}
        o["bytes"] = {k: runs[0][k] for k in ("onnx_bytes", "gguf_bytes", "ct2_bytes") if k in runs[0]}
        o["artifact_sha256"] = runs[0].get("artifact_sha256")
        if runs[0].get("ppl_tokens"):
            o["ppl"] = float(np.exp(runs[0]["ppl_nll"] / runs[0]["ppl_tokens"]))
        L = lg(v)
        if L is not None:
            o["vs_reference_pytorch_float32_original"] = logit_metrics(L, ref)
            if ref64 is not None:
                o["vs_float64_original"] = logit_metrics(L, ref64)
            if base is not None:
                o["vs_same_runtime_original"] = logit_metrics(L, base)
        base_gen = rs("orig", 0).get("gen", {})
        g = runs[0].get("gen", {})
        o["generation"] = {k: {"identical": sum(1 for a, b in zip(g[k], base_gen[k]) if a == b), "of": len(g[k]),
                               "first_divergences": [first_divergence(a, b) for a, b in zip(g[k], base_gen[k])]} for k in g if k in base_gen}
        out["variants"][v] = o
    dump(out, os.path.join(work, f"summary-{runtime}-{dtype}.json"))
    print(json.dumps({v: {k: o.get(k) for k in ("ppl", "prefill_tps", "decode_tps", "peak_rss_mb")} for v, o in out["variants"].items()}, indent=1))


if __name__ == "__main__":
    c = sys.argv[1]
    if c == "prepare":
        prepare(sys.argv[2], sys.argv[3])
    elif c == "one":
        one(sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]), sys.argv[6])
    elif c == "all":
        run_all(sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]))
