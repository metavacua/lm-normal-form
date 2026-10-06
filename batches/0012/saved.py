# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
# What ONNX Runtime writes when asked to save the graph it optimized (SessionOptions.optimized_model_filepath), with
# optimization off and with all of it: the header and the census of the saved file, whether onnx.checker takes it, and whether
# ONNX Runtime run again on it, with optimization off, gives the logits of the session that saved it. Each line of the
# result is an id, what was predicted, what was seen. Usage: saved.py MODEL.onnx WORKDIR IDS_FILE VOCAB_FILE OUT.tsv
import collections, csv, os, subprocess, sys
import numpy as np
import onnx
import onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))
LOGITS = os.path.join(HERE, "..", "0010", "logits.py")


def save(model, path, level):
    o = ort.SessionOptions()
    o.graph_optimization_level = level
    o.optimized_model_filepath = path
    ort.InferenceSession(model, o, providers=["CPUExecutionProvider"])


def census(path):
    m = onnx.load(path)
    ops = collections.Counter((n.domain, n.op_type) for n in m.graph.node)
    return m, ops


def same_logits(a, b, n, vocab):
    return all(np.array_equal(np.fromfile(f"{a}/{k}.f32", np.float32).reshape(-1, vocab), np.fromfile(f"{b}/{k}.f32", np.float32).reshape(-1, vocab)) for k in range(n))


def main(model, work, ids_file, vocab_file, out):
    ids, vocab = open(ids_file).read().split(), int(open(vocab_file).read())
    os.makedirs(work, exist_ok=True)
    base = onnx.load(model, load_external_data=False)
    res = []

    def line(i, predicted, seen, ok):
        res.append([i, predicted, str(seen), "yes" if ok else "no"])
        print(i, "predicted", predicted, "seen", seen, "ok" if ok else "MISS", flush=True)
    for level, tag in ((ort.GraphOptimizationLevel.ORT_DISABLE_ALL, "none"), (ort.GraphOptimizationLevel.ORT_ENABLE_ALL, "all")):
        path = f"{work}/saved-{tag}.onnx"
        save(model, path, level)
        m, ops = census(path)
        domains = sorted({e.domain for e in m.opset_import})
        if tag == "none":
            line("F1a", "ir_version and producer as the original's", (m.ir_version, m.producer_name), (m.ir_version, m.producer_name) == (base.ir_version, base.producer_name))
            line("F1b", "more than one opset_import entry", len(domains), len(domains) > 1)
            line("F1c", "no Constant nodes", ops[("", "Constant")], ops[("", "Constant")] == 0)
            line("F1d", "more initializers than the original's", len(m.graph.initializer), len(m.graph.initializer) > len(base.graph.initializer))
            line("F1e", "ReduceMean has noop_with_empty_axes written out",
                 [a.name for n in m.graph.node if n.op_type == "ReduceMean" for a in n.attribute][:1],
                 any(a.name == "noop_with_empty_axes" for n in m.graph.node if n.op_type == "ReduceMean" for a in n.attribute))
        else:
            ms = sum(v for (d, _), v in ops.items() if d == "com.microsoft")
            line("F2a", "at least one node in com.microsoft", ms, ms >= 1)
            line("F2b", "30 QuickGelu", ops[("com.microsoft", "QuickGelu")], ops[("com.microsoft", "QuickGelu")] == 30)
            line("F2c", "SimplifiedLayerNormalization in the default domain, at most 61", ops[("", "SimplifiedLayerNormalization")], 0 <= ops[("", "SimplifiedLayerNormalization")] <= 61)
            try:
                onnx.checker.check_model(path)
                seen = "accept"
            except Exception as e:
                seen = "refuse: " + str(e).strip().splitlines()[-1][:160]
            line("F3", "refuse if a default-domain SimplifiedLayerNormalization is present, else accept",
                 seen, (seen == "accept") == (ops[("", "SimplifiedLayerNormalization")] == 0))
            os.makedirs(f"{work}/mem", exist_ok=True)
            os.makedirs(f"{work}/reload", exist_ok=True)
            subprocess.run([sys.executable, LOGITS, model, f"{work}/mem", "all"] + ids, check=True, capture_output=True)
            q = subprocess.run([sys.executable, LOGITS, path, f"{work}/reload", "none"] + ids, capture_output=True, text=True)
            line("F4a", "the saved file is accepted again", "accept" if q.returncode == 0 else "refuse: " + q.stderr.strip().splitlines()[-1][:160], q.returncode == 0)
            if q.returncode == 0:
                eq = same_logits(f"{work}/mem", f"{work}/reload", len(ids), vocab)
                line("F4b", "its logits are the saving session's, byte for byte", eq, eq)
        print(tag, "census:", dict(ops.most_common(8)), "domains", domains, "nodes", len(m.graph.node), "initializers", len(m.graph.initializer), flush=True)
        os.remove(path)
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["id", "predicted", "seen", "as_predicted"])
        w.writerows(res)


if __name__ == "__main__":
    main(*sys.argv[1:6])
