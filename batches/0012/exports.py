# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
# The export repeated under other arguments and other environments, as the rows of predictions-export.tsv say: what
# header the file has (ir_version, opset_import), whether ONNX Runtime takes it and how far its logits are from the first
# export's (`same` is byte for byte, `near` is within 1e-3), whether its sha256 is the first export's, and whether the
# exporter's output has the sentence the row expects. Then the observation against the prediction.
# Usage: exports.py MODEL_ID WORKDIR BASELINE.onnx IDS_FILE VOCAB_FILE PREDICTIONS.tsv OUT.tsv
import csv, hashlib, os, subprocess, sys
import numpy as np
import onnx

HERE = os.path.dirname(os.path.abspath(__file__))
LOGITS = os.path.join(HERE, "..", "0010", "logits.py")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def main(model, work, baseline, ids_file, vocab_file, predictions, out):
    ids, vocab = open(ids_file).read().split(), int(open(vocab_file).read())
    base_sha = sha(baseline)
    os.makedirs(f"{work}/base", exist_ok=True)
    subprocess.run([sys.executable, LOGITS, baseline, f"{work}/base", "all"] + ids, check=True, capture_output=True)
    res = []
    for r in csv.DictReader(open(predictions, encoding="utf-8"), delimiter="\t"):
        d = f"{work}/{r['id']}"
        env = dict(os.environ)
        if r["env"] != "-":
            env[r["env"].split("=")[0]] = r["env"].split("=", 1)[1]
        p = subprocess.run(["optimum-cli", "export", "onnx", "--model", model, "--task", "text-generation"] + r["args"].split() + [d],
                           capture_output=True, text=True, env=env)
        o = {"id": r["id"], "args": r["args"], "env": r["env"], "exit": p.returncode, "ir_version": "", "opset": "", "ort": "", "sha_equal": "", "log": "no"}
        o["log"] = "yes" if r["log"] != "-" and r["log"].lower() in (p.stdout + p.stderr).lower() else ("-" if r["log"] == "-" else "no")
        path = f"{d}/model.onnx"
        if p.returncode == 0 and os.path.exists(path):
            m = onnx.load(path, load_external_data=False)
            o["ir_version"], o["opset"] = str(m.ir_version), ",".join(str(e.version) for e in m.opset_import if e.domain in ("", "ai.onnx"))
            o["sha_equal"] = "yes" if sha(path) == base_sha else "no"
            os.makedirs(f"{d}/logits", exist_ok=True)
            q = subprocess.run([sys.executable, LOGITS, path, f"{d}/logits", "all"] + ids, capture_output=True, text=True)
            if q.returncode:
                o["ort"] = "refuse"
            else:
                worst, same = 0.0, True
                for k in range(len(ids)):
                    a = np.fromfile(f"{work}/base/{k}.f32", np.float32).reshape(-1, vocab)
                    b = np.fromfile(f"{d}/logits/{k}.f32", np.float32).reshape(-1, vocab)
                    same = same and a.shape == b.shape and bool(np.array_equal(a, b))
                    worst = max(worst, float(np.abs(a - b).max()) if a.shape == b.shape else float("inf"))
                o["ort"] = "same" if same else "near" if worst <= 1e-3 else "different"
        o["ir_ok"], o["opset_ok"] = o["ir_version"] == r["ir_version"], o["opset"] == r["opset"]
        o["ort_ok"] = o["ort"] in r["ort"].split("/")
        o["sha_ok"] = o["sha_equal"] == r["sha_equal"]
        o["log_ok"] = o["log"] == ("yes" if r["log"] != "-" else "-")
        res.append(o)
        print({k: o[k] for k in ("id", "exit", "ir_version", "opset", "ort", "sha_equal", "log")}, "hits:", [k for k in ("ir_ok", "opset_ok", "ort_ok", "sha_ok", "log_ok") if o[k]], flush=True)
        subprocess.run(["rm", "-rf", d])
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(res[0]), delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(res)


if __name__ == "__main__":
    main(*sys.argv[1:8])
